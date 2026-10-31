//! Entity-configured N-1 watches compiled once into a compact shared rule plan.
//! Numerical evaluation consumes solver-order voltages and a local outage mask.
use super::postprocess::{BranchCatalog, BranchIdentity};
use crate::basic::{
    branch::BranchFlowResult,
    ecs::{
        elements::{
            BusMatrixIndex, NodeAggRes,
            bus::{BusID, VmLimit},
            units::PerUnit,
        },
        powerflow::systems::PowerFlowMat,
    },
};
use bevy_ecs::prelude::*;
use num_complex::Complex64;
use soa_rs::{Soa, Soars};
use std::collections::{HashMap, HashSet};

/// Monitoring rules attached to a source branch or bus Entity.
///
/// Insert this component before preparation. [`WatchPlan::from_world`] resolves
/// source Entities into solver indices and freezes the rules; workers never query
/// the source ECS while solving.
#[derive(Component, Clone, Debug, Default)]
pub struct N1Watch {
    pub rules: Vec<WatchRule>,
}
/// A caller-defined rule ID and the condition evaluated for each scenario.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WatchRule {
    /// Unique within one Entity's [`N1Watch`] and retained in [`WatchHit`].
    pub rule_id: u64,
    pub condition: WatchCondition,
}
/// Security constraints and diagnostic triggers supported by the compiled plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WatchCondition {
    /// Emit a diagnostic hit for every scenario, including solve failure.
    Always,
    /// Branch loading in percent; valid on line and transformer Entities.
    BranchLoadingAbove(f64),
    /// Maximum terminal current in kA; valid on branch Entities.
    BranchCurrentAbove(f64),
    /// Emit a diagnostic hit when DCPF or Newton fails.
    SolveFailed,
    /// Bounds are captured from `VmLimit<PerUnit>` on this bus Entity.
    BusVoltageOutsideLimits,
}

/// Resolved location of a rule hit in the prepared numerical model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchTarget {
    Branch {
        index: usize,
        identity: BranchIdentity,
    },
    Bus {
        index: usize,
        id: i64,
    },
}
impl WatchTarget {
    pub fn branch(self) -> Option<(usize, BranchIdentity)> {
        match self {
            Self::Branch { index, identity } => Some((index, identity)),
            Self::Bus { .. } => None,
        }
    }
    pub fn bus(self) -> Option<(usize, i64)> {
        match self {
            Self::Bus { index, id } => Some((index, id)),
            Self::Branch { .. } => None,
        }
    }
}
#[derive(Debug, Clone, Copy, Soars)]
#[soa_derive(Debug)]
pub struct WatchHit {
    pub rule_id: u64,
    pub source_entity: Entity,
    pub target: WatchTarget,
    pub value: Option<f64>,
    /// Index into this scenario's shared result list. Failed solves carry none.
    pub result_index: Option<usize>,
}

/// Compact hit columns plus one full result per matched branch, shared by its rules.
#[derive(Debug, Default)]
pub struct WatchHits {
    records: Soa<WatchHit>,
    branches: Vec<BranchFlowResult>,
    violation_count: usize,
    first_violation: usize,
}
impl WatchHits {
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn violation_count(&self) -> usize {
        self.violation_count
    }
    /// The first matched numerical constraint, excluding diagnostic rules.
    pub fn first_violation(&self) -> Option<WatchHitRef<'_>> {
        if self.violation_count == 0 {
            None
        } else {
            self.get(self.first_violation)
        }
    }
    pub fn capacity(&self) -> usize {
        self.records.capacity()
    }
    pub fn records(&self) -> &soa_rs::Slice<WatchHit> {
        &self.records
    }
    pub fn iter(&self) -> soa_rs::Iter<'_, WatchHit> {
        self.records.iter()
    }
    pub fn get(&self, index: usize) -> Option<WatchHitRef<'_>> {
        self.records.get(index)
    }
    pub fn branch_for_hit(&self, index: usize) -> Option<&BranchFlowResult> {
        let slot = *self.records.get(index)?.result_index;
        self.branches.get(slot?)
    }
    pub fn branch_results(&self) -> &[BranchFlowResult] {
        &self.branches
    }
    fn reserve_record(&mut self, limit: usize) {
        if self.records.len() == self.records.capacity() {
            let capacity = self.records.capacity().saturating_mul(2).max(8).min(limit);
            self.records.reserve_exact(capacity - self.records.len());
        }
    }
}
struct TargetRules {
    branch_index: usize,
    rules: Vec<WatchRule>,
}
#[derive(Clone, Copy)]
enum BusRule {
    OutsideLimits,
    Always,
    SolveFailed,
}
#[derive(Soars)]
struct BusWatch {
    source_entity: Entity,
    bus_id: i64,
    solver_index: usize,
    rule_id: u64,
    condition: BusRule,
    min_squared: f64,
    max_squared: f64,
}
/// One immutable plan shared across all worker Apps. Each target's branch
/// values are computed once per scenario, regardless of the number of rules.
pub struct WatchPlan {
    catalog_id: u64,
    targets: Vec<TargetRules>,
    rule_count: usize,
    buses: Soa<BusWatch>,
    bus_count: usize,
    bus_status_rules: bool,
}
impl WatchPlan {
    /// Freeze branch rules and bus voltage limits with final solver indices.
    pub fn from_world(world: &mut World, catalog: &BranchCatalog) -> Result<Self, String> {
        if world.id() != catalog.source_world_id() {
            return Err("N1 watches and branch catalog belong to different Worlds".into());
        }
        let indices: HashMap<_, _> = catalog
            .source_entities()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, e)| (e, i))
            .collect();
        let mut targets = Vec::new();
        let mut rule_count = 0;
        let mut buses = Soa::<BusWatch>::new();
        let mut bus_count = 0;
        let mut bus_status_rules = false;
        for (entity, watch) in world.query::<(Entity, &N1Watch)>().iter(world) {
            if watch.rules.is_empty() {
                continue;
            }
            let branch_index = indices.get(&entity).copied();
            let mut ids = HashSet::new();
            for rule in &watch.rules {
                if !ids.insert(rule.rule_id) {
                    return Err(format!(
                        "duplicate N1 rule ID {} on {entity:?}",
                        rule.rule_id
                    ));
                }
                match rule.condition {
                    WatchCondition::BranchLoadingAbove(value)
                    | WatchCondition::BranchCurrentAbove(value)
                        if !value.is_finite() =>
                    {
                        return Err(format!("N1 rule {} threshold must be finite", rule.rule_id));
                    }
                    _ => {}
                }
            }
            rule_count += watch.rules.len();
            if let Some(branch_index) = branch_index {
                if watch
                    .rules
                    .iter()
                    .any(|r| r.condition == WatchCondition::BusVoltageOutsideLimits)
                {
                    return Err(format!("voltage rule on {entity:?} requires a bus entity"));
                }
                targets.push(TargetRules {
                    branch_index,
                    rules: watch.rules.clone(),
                });
            } else {
                let id = world
                    .get::<BusID>(entity)
                    .ok_or_else(|| {
                        format!("N1Watch on {entity:?} requires a cataloged branch or prepared bus")
                    })?
                    .0;
                let original = world
                    .get::<BusMatrixIndex>(entity)
                    .ok_or("bus watch requires BusMatrixIndex")?
                    .0;
                let mat = world
                    .get_resource::<PowerFlowMat>()
                    .ok_or("bus watch requires PowerFlowMat")?;
                let merged = match world.get_resource::<NodeAggRes>() {
                    Some(agg) => *agg
                        .original_to_merged
                        .get(original)
                        .ok_or("bus watch merged mapping is missing")?,
                    None => original,
                };
                let index = *mat
                    .to_perm
                    .get(merged)
                    .ok_or("bus watch permutation mapping is missing")?;
                bus_count = mat.y_bus.nrows();
                if index >= bus_count {
                    return Err("bus watch solver index is out of bounds".into());
                }
                for rule in &watch.rules {
                    let (condition, min_squared, max_squared) = match rule.condition {
                        WatchCondition::BusVoltageOutsideLimits => {
                            let limits = world
                                .get::<VmLimit<PerUnit>>(entity)
                                .ok_or("voltage rule requires VmLimit<PerUnit>")?;
                            let (min, max) = (limits.min(), limits.max());
                            if !min.is_finite()
                                || !max.is_finite()
                                || min < 0.
                                || max < min
                                || !max.powi(2).is_finite()
                            {
                                return Err(format!(
                                    "bus {id} voltage limits must be finite, nonnegative and ordered"
                                ));
                            }
                            (BusRule::OutsideLimits, min * min, max * max)
                        }
                        WatchCondition::Always => {
                            bus_status_rules = true;
                            (BusRule::Always, 0., 0.)
                        }
                        WatchCondition::SolveFailed => {
                            bus_status_rules = true;
                            (BusRule::SolveFailed, 0., 0.)
                        }
                        _ => {
                            return Err(format!(
                                "branch loading/current rule on bus {id} requires a branch entity"
                            ));
                        }
                    };
                    buses.push(BusWatch {
                        source_entity: entity,
                        bus_id: id,
                        solver_index: index,
                        rule_id: rule.rule_id,
                        condition,
                        min_squared,
                        max_squared,
                    });
                }
            }
        }
        targets.sort_unstable_by_key(|t| t.branch_index);
        Ok(Self {
            catalog_id: catalog.catalog_id(),
            targets,
            rule_count,
            buses,
            bus_count,
            bus_status_rules,
        })
    }
    pub fn rule_count(&self) -> usize {
        self.rule_count
    }
    pub fn is_empty(&self) -> bool {
        self.rule_count == 0
    }
    pub(super) fn max_output_bytes(&self) -> Option<usize> {
        self.rule_count
            .checked_mul(std::mem::size_of::<WatchHit>())
            .and_then(|hits| {
                self.targets
                    .len()
                    .checked_mul(std::mem::size_of::<BranchFlowResult>())
                    .and_then(|results| hits.checked_add(results))
            })
    }

    /// Screening mode: stop at the first violated numerical constraint.
    /// Diagnostic Always/SolveFailed rules are handled by the run's outcome policy.
    /// This path allocates no WatchHit or BranchFlowResult storage.
    pub fn is_violated(
        &self,
        catalog: &BranchCatalog,
        voltage: &[Complex64],
        outage: Option<usize>,
    ) -> Result<bool, String> {
        if self.catalog_id != catalog.catalog_id() {
            return Err("N1 WatchPlan belongs to a different branch catalog".into());
        }
        if !self.buses.is_empty() && voltage.len() != self.bus_count {
            return Err("bus watch voltage dimensions differ from the prepared model".into());
        }
        // Voltage comparisons are cheaper than branch current products.
        for i in 0..self.buses.len() {
            if !matches!(self.buses.condition()[i], BusRule::OutsideLimits) {
                continue;
            }
            let squared = voltage[self.buses.solver_index()[i]].norm_sqr();
            if !squared.is_finite()
                || squared < self.buses.min_squared()[i]
                || squared > self.buses.max_squared()[i]
            {
                return Ok(true);
            }
        }
        for target in &self.targets {
            if !target.rules.iter().any(|r| {
                matches!(
                    r.condition,
                    WatchCondition::BranchLoadingAbove(_) | WatchCondition::BranchCurrentAbove(_)
                )
            }) {
                continue;
            }
            let metrics = catalog.check_metrics(target.branch_index, voltage, outage)?;
            for rule in &target.rules {
                let (loading, threshold) = match rule.condition {
                    WatchCondition::BranchLoadingAbove(v) => (true, v),
                    WatchCondition::BranchCurrentAbove(v) => (false, v),
                    _ => continue,
                };
                let value = if loading {
                    metrics.loading_squared
                } else {
                    metrics.current_squared
                };
                if threshold < 0. || value > threshold * threshold {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Zero hits means no output message is needed. Failure watches inspect only
    /// the solve status; numerical predicates never read a failed voltage vector.
    pub fn evaluate(
        &self,
        catalog: &BranchCatalog,
        voltage: &[Complex64],
        outage: Option<usize>,
        converged: bool,
    ) -> Result<WatchHits, String> {
        if self.catalog_id != catalog.catalog_id() {
            return Err("N1 WatchPlan belongs to a different branch catalog".into());
        }
        let mut hits = WatchHits::default();
        for target in &self.targets {
            let branch_index = target.branch_index;
            // A target with only failure rules needs no ordinary postprocessing.
            let needs_numbers = converged
                && target
                    .rules
                    .iter()
                    .any(|r| r.condition != WatchCondition::SolveFailed);
            let result = if needs_numbers {
                Some(catalog.compute_one(branch_index, voltage, outage)?)
            } else {
                None
            };
            let first_hit = hits.len();
            let result_index = result.as_ref().map(|_| hits.branches.len());
            for rule in &target.rules {
                let (matched, value) = match rule.condition {
                    WatchCondition::Always => (true, None),
                    WatchCondition::SolveFailed => (!converged, None),
                    WatchCondition::BranchLoadingAbove(threshold) => {
                        let value = result.as_ref().map(|r| r.loading_percent);
                        (value.is_some_and(|v| v > threshold), value)
                    }
                    WatchCondition::BranchCurrentAbove(threshold) => {
                        let value = result.as_ref().map(|r| r.i_ka);
                        (value.is_some_and(|v| v > threshold), value)
                    }
                    WatchCondition::BusVoltageOutsideLimits => {
                        unreachable!("validated during preparation")
                    }
                };
                if matched {
                    if matches!(
                        rule.condition,
                        WatchCondition::BranchLoadingAbove(_)
                            | WatchCondition::BranchCurrentAbove(_)
                    ) {
                        if hits.violation_count == 0 {
                            hits.first_violation = hits.len();
                        }
                        hits.violation_count += 1;
                    }
                    hits.reserve_record(self.rule_count);
                    hits.records.push(WatchHit {
                        rule_id: rule.rule_id,
                        source_entity: catalog.source_entities()[branch_index],
                        target: WatchTarget::Branch {
                            index: branch_index,
                            identity: catalog.identities()[branch_index],
                        },
                        value,
                        result_index,
                    });
                }
            }
            if hits.len() > first_hit {
                if let Some(result) = result {
                    if hits.branches.len() == hits.branches.capacity() {
                        let capacity = (hits.branches.capacity().saturating_mul(2))
                            .max(8)
                            .min(self.targets.len());
                        hits.branches.reserve_exact(capacity - hits.branches.len());
                    }
                    hits.branches.push(result);
                }
            }
        }
        if converged && !self.buses.is_empty() && voltage.len() != self.bus_count {
            return Err("bus watch voltage dimensions differ from the prepared model".into());
        }
        if converged || self.bus_status_rules {
            for i in 0..self.buses.len() {
                let condition = self.buses.condition()[i];
                if !converged && matches!(condition, BusRule::OutsideLimits) {
                    continue;
                }
                let squared = if converged {
                    Some(voltage[self.buses.solver_index()[i]].norm_sqr())
                } else {
                    None
                };
                let matched = match condition {
                    BusRule::OutsideLimits => squared.is_some_and(|v| {
                        !v.is_finite()
                            || v < self.buses.min_squared()[i]
                            || v > self.buses.max_squared()[i]
                    }),
                    BusRule::Always => true,
                    BusRule::SolveFailed => !converged,
                };
                if matched {
                    if matches!(condition, BusRule::OutsideLimits) {
                        if hits.violation_count == 0 {
                            hits.first_violation = hits.len();
                        }
                        hits.violation_count += 1;
                    }
                    hits.reserve_record(self.rule_count);
                    hits.records.push(WatchHit {
                        rule_id: self.buses.rule_id()[i],
                        source_entity: self.buses.source_entity()[i],
                        target: WatchTarget::Bus {
                            index: self.buses.solver_index()[i],
                            id: self.buses.bus_id()[i],
                        },
                        value: squared.map(f64::sqrt),
                        result_index: None,
                    });
                }
            }
        }
        Ok(hits)
    }
}
