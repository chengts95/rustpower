//! Entity-oriented N-1 orchestration over a borrowed, immutable prepared baseline.
use super::{
    BranchIdentity, BridgeCut, N1Config, N1EventBatch, N1Initialization, N1Reporting, N1Summary,
    PreparedN1, execution::run_round,
};
use bevy_ecs::entity::Entity;
use bevy_ecs::world::World;
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

/// Independent single-device outages. Each entity defines one scenario in which
/// that device is OutOfService; the source World keeps its baseline state.
#[derive(Clone, Debug, Default)]
pub struct N1FaultSet {
    outages: Vec<Entity>,
}
impl N1FaultSet {
    /// Preserve the first occurrence of each entity, in input order.
    pub fn out_of_service(entities: impl IntoIterator<Item = Entity>) -> Self {
        let mut seen = HashSet::new();
        Self {
            outages: entities
                .into_iter()
                .filter(|&entity| seen.insert(entity))
                .collect(),
        }
    }
    pub fn entities(&self) -> &[Entity] {
        &self.outages
    }
    pub fn len(&self) -> usize {
        self.outages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.outages.is_empty()
    }
}
impl FromIterator<Entity> for N1FaultSet {
    fn from_iter<T: IntoIterator<Item = Entity>>(iter: T) -> Self {
        Self::out_of_service(iter)
    }
}

/// Options belong to this run; the shared baseline is reused by later runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct N1RunOptions {
    pub config: N1Config,
    /// None inherits the baseline preparation's default policy.
    pub initialization: Option<N1Initialization>,
    /// Include successful selected scenarios in the callback, with the requested
    /// payload level. Useful for a small retry batch and explicit detail requests.
    pub emit_all: bool,
}

#[derive(Clone, Debug)]
pub struct N1Island {
    pub outage_index: usize,
    pub outage: BranchIdentity,
    pub outage_entity: Entity,
    pub cut: BridgeCut,
}

/// Four disjoint terminal categories cover the supplied fault set. Detailed
/// numerical events travel through the callback; topology records remain here.
#[derive(Debug, Default)]
pub struct N1StudyReport {
    pub solve: N1Summary,
    pub islands: Vec<N1Island>,
}
impl N1StudyReport {
    pub fn processed(&self) -> usize {
        self.solve.processed + self.islands.len()
    }
    pub fn passed(&self) -> usize {
        self.solve.passed()
    }
    pub fn violated(&self) -> usize {
        self.solve.violated_cases
    }
    pub fn numerical_failed(&self) -> usize {
        self.solve.failed
    }
    pub fn islanded(&self) -> usize {
        self.islands.len()
    }
    pub fn rejected(&self) -> usize {
        self.solve.rejected() + self.islands.len()
    }
}

#[derive(Debug)]
pub enum N1StudyError {
    Baseline(String),
    UnknownBranch(Entity),
    AlreadyOutOfService(Entity),
    Execution(String),
}
impl fmt::Display for N1StudyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Baseline(error) => write!(f, "N-1 baseline: {error}"),
            Self::UnknownBranch(entity) => write!(
                f,
                "fault entity {entity:?} is absent from the prepared branch catalog"
            ),
            Self::AlreadyOutOfService(entity) => write!(
                f,
                "fault entity {entity:?} is already out of service in the baseline"
            ),
            Self::Execution(error) => write!(f, "N-1 execution: {error}"),
        }
    }
}
impl std::error::Error for N1StudyError {}

/// High-level behavior borrows PreparedN1. It owns neither a World nor an App.
/// Entity lookup is built once; workers borrow the same immutable prepared data.
pub struct N1Study<'base> {
    prepared: &'base PreparedN1,
    branches: HashMap<Entity, usize>,
}
impl<'base> N1Study<'base> {
    /// Check the base once, then reuse this study for screening, DC retries and
    /// detailed runs. Entity identities refer to the prepared source World.
    pub fn new(prepared: &'base PreparedN1) -> Result<Self, N1StudyError> {
        prepared.check_baseline().map_err(N1StudyError::Baseline)?;
        let branches = prepared
            .base()
            .catalog()
            .source_entities()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, entity)| (entity, index))
            .collect();
        Ok(Self { prepared, branches })
    }
    pub fn prepared(&self) -> &'base PreparedN1 {
        self.prepared
    }
    pub fn all_branch_outages(&self) -> N1FaultSet {
        let catalog = self.prepared.base().catalog();
        N1FaultSet {
            outages: self
                .prepared
                .active_branches()
                .map(|i| catalog.source_entities()[i])
                .collect(),
        }
    }

    /// Execute the fault set loaded during preparation. An unbound baseline
    /// defaults to all initially in-service branches.
    pub fn execute<F>(
        &self,
        options: N1RunOptions,
        collect: F,
    ) -> Result<N1StudyReport, N1StudyError>
    where
        F: FnMut(N1EventBatch) + Send,
    {
        if let Some(faults) = self.prepared.faults() {
            self.run(faults, options, collect)
        } else {
            self.run(&self.all_branch_outages(), options, collect)
        }
    }

    /// Resolve the entity set before dispatch, classify bridge outages, and run
    /// the remaining independent scenarios on the existing scoped workers.
    pub fn run<F>(
        &self,
        faults: &N1FaultSet,
        options: N1RunOptions,
        collect: F,
    ) -> Result<N1StudyReport, N1StudyError>
    where
        F: FnMut(N1EventBatch) + Send,
    {
        let catalog = self.prepared.base().catalog();
        let mut candidates = Vec::with_capacity(faults.len());
        let mut islands = Vec::new();
        for &entity in faults.entities() {
            let &index = self
                .branches
                .get(&entity)
                .ok_or(N1StudyError::UnknownBranch(entity))?;
            if !catalog.in_service()[index] {
                return Err(N1StudyError::AlreadyOutOfService(entity));
            }
            if let Some(cut) = self.prepared.topology().bridge_cut(index) {
                if cut.slackless_bus_count() > 0 {
                    islands.push(N1Island {
                        outage_index: index,
                        outage: catalog.identities()[index],
                        outage_entity: entity,
                        cut,
                    });
                    continue;
                }
            }
            candidates.push(index);
        }
        let initialization = options
            .initialization
            .unwrap_or(self.prepared.initialization());
        let solve = run_round(
            self.prepared,
            initialization,
            options.config,
            candidates,
            options.emit_all,
            collect,
        )
        .map_err(N1StudyError::Execution)?;
        Ok(N1StudyReport { solve, islands })
    }

    /// Return every selected numerical scenario, retaining the configured
    /// Detailed rule payloads. Successful scenarios can have an empty hit list.
    pub fn details<F>(
        &self,
        faults: &N1FaultSet,
        mut options: N1RunOptions,
        collect: F,
    ) -> Result<N1StudyReport, N1StudyError>
    where
        F: FnMut(N1EventBatch) + Send,
    {
        options.config.reporting = N1Reporting::Detailed;
        options.emit_all = true;
        self.run(faults, options, collect)
    }
}

impl PreparedN1 {
    /// Bind a specific set atomically; failed validation preserves the previous set.
    pub fn load_faults(&mut self, faults: N1FaultSet) -> Result<(), N1StudyError> {
        let catalog = self.base().catalog();
        let indices: HashMap<_, _> = catalog
            .source_entities()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, entity)| (entity, index))
            .collect();
        for &entity in faults.entities() {
            let &index = indices
                .get(&entity)
                .ok_or(N1StudyError::UnknownBranch(entity))?;
            if !catalog.in_service()[index] {
                return Err(N1StudyError::AlreadyOutOfService(entity));
            }
        }
        self.faults = Some(faults);
        Ok(())
    }
    pub fn faults(&self) -> Option<&N1FaultSet> {
        self.faults.as_ref()
    }
    pub fn study(&self) -> Result<N1Study<'_>, N1StudyError> {
        N1Study::new(self)
    }
}

/// Prepare resources and bind the supplied entity faults for subsequent execute().
/// The source World remains caller-owned; each entity is an independent outage.
pub fn prepare_n1_for(world: &mut World, faults: N1FaultSet) -> Result<PreparedN1, N1StudyError> {
    let mut prepared = super::prepare_n1(world).map_err(N1StudyError::Baseline)?;
    prepared.load_faults(faults)?;
    Ok(prepared)
}
