//! Switch models are prepared before core Ybus assembly, outside the solver loop.
use super::{super::powerflow::systems::PowerFlowMat, *};
use crate::io::pandapower::SwitchType;
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use derive_more::{Deref, DerefMut};
use nalgebra::{Complex, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use std::collections::HashMap;

mod comps;
pub use comps::*;

/// Transient merge projection. Indices refer to physical matrix order, not BusID.
#[derive(Default, Debug, Clone, Resource, serde::Serialize, serde::Deserialize)]
pub struct NodeAggRes {
    pub expand_mat: CscMatrix<f64>,
    /// Selects one controlling/initial voltage per merged bus (not result expansion).
    pub expand_mat_v: CscMatrix<f64>,
    pub original_to_merged: Vec<usize>,
    pub representatives: Vec<usize>,
}
impl NodeAggRes {
    pub fn expand_voltage(&self, merged: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
        DVector::from_iterator(
            self.original_to_merged.len(),
            self.original_to_merged.iter().map(|&i| merged[i]),
        )
    }
}
#[derive(Default, Debug, Clone, Component, serde::Serialize, serde::Deserialize)]
pub struct MergeNode(pub usize, pub usize);

/// Legacy arbitrary-node union-find API. The plugin uses dense vectors internally.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeMerge {
    pub parent: HashMap<u64, u64>,
    pub rank: HashMap<u64, u64>,
}
#[derive(
    Default, Debug, Clone, Deref, DerefMut, Resource, serde::Serialize, serde::Deserialize,
)]
pub struct NodeMapping(HashMap<u64, u64>);
impl NodeMerge {
    pub fn new(nodes: &[u64]) -> Self {
        Self {
            parent: nodes.iter().map(|&n| (n, n)).collect(),
            rank: nodes.iter().map(|&n| (n, 0)).collect(),
        }
    }
    fn find(&mut self, node: u64) -> u64 {
        let mut root = node;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        let mut n = node;
        while n != root {
            let next = self.parent[&n];
            self.parent.insert(n, root);
            n = next;
        }
        root
    }
    pub fn union(&mut self, a: u64, b: u64) {
        let a = self.find(a);
        let b = self.find(b);
        if a == b {
            return;
        }
        if self.rank[&a] < self.rank[&b] {
            self.parent.insert(a, b);
        } else {
            self.parent.insert(b, a);
            if self.rank[&a] == self.rank[&b] {
                *self.rank.get_mut(&a).unwrap() += 1;
            }
        }
    }
    pub fn get_node_mapping(&self, starting_idx: u64) -> HashMap<u64, u64> {
        let mut roots = HashMap::new();
        let mut result = HashMap::new();
        let mut nodes: Vec<_> = self.parent.keys().copied().collect();
        nodes.sort_unstable();
        for node in nodes {
            let mut root = node;
            while self.parent[&root] != root {
                root = self.parent[&root];
            }
            let next = starting_idx + roots.len() as u64;
            let merged = *roots.entry(root).or_insert(next);
            result.insert(node, merged);
        }
        result
    }
}
#[derive(Resource, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SwitchModel {
    Merge,
    Admittance,
}

/// Last prepared state lets Changed queries ignore startup writes already consumed.
#[derive(Component, Clone, PartialEq)]
pub(crate) struct PreparedSwitch {
    bus: i64,
    element: i64,
    et: SwitchType,
    z: f64,
    closed: bool,
    disabled: bool,
}
fn prepared(s: &Switch, state: &SwitchState, disabled: bool) -> PreparedSwitch {
    PreparedSwitch {
        bus: s.bus,
        element: s.element,
        et: s.et.clone(),
        z: s.z_ohm,
        closed: state.0,
        disabled,
    }
}
#[derive(Component)]
struct SwitchStamp;

pub(crate) fn detect_switch_changes(
    q: Query<
        (
            &Switch,
            &SwitchState,
            Option<&OutOfService>,
            Option<&PreparedSwitch>,
        ),
        Or<(Changed<Switch>, Changed<SwitchState>, Added<OutOfService>)>,
    >,
    mut removed: RemovedComponents<Switch>,
    mut removed_state: RemovedComponents<SwitchState>,
    mut removed_service: RemovedComponents<OutOfService>,
    policy: Option<Res<OpenLinePolicy>>,
    prepared_policy: Option<Res<PreparedOpenLinePolicy>>,
    mut events: MessageWriter<crate::basic::ecs::powerflow::structure_update::FullRebuildEvent>,
) {
    let changed = q
        .iter()
        .any(|(s, state, off, old)| old.cloned() != Some(prepared(s, state, off.is_some())));
    // Service removal may belong to another device; rebuilding is still correct.
    let removed =
        removed.read().count() + removed_state.read().count() + removed_service.read().count() > 0;
    let policy_changed = policy.as_deref().copied() != prepared_policy.as_deref().map(|p| p.0);
    if changed || removed || policy_changed {
        events.write_default();
    }
}
fn root(parent: &mut [usize], n: usize) -> usize {
    let mut r = n;
    while parent[r] != r {
        r = parent[r];
    }
    let mut x = n;
    while x != r {
        let next = parent[x];
        parent[x] = r;
        x = next;
    }
    r
}
fn prepare_switch_model(world: &mut World, model: SwitchModel) {
    world.remove_resource::<NodeAggRes>();
    // Stamps are derived and must disappear when switches open, change type or are removed.
    let stale: Vec<_> = world
        .query_filtered::<Entity, With<SwitchStamp>>()
        .iter(world)
        .collect();
    for e in stale {
        world.entity_mut(e).remove::<(
            SwitchStamp,
            Port4MatPatch,
            FromBus,
            ToBus,
            FromBusRef,
            ToBusRef,
            FromBusMatrixIndex,
            ToBusMatrixIndex,
        )>();
    }
    let n = world.resource::<NodeLookup>().len();
    let mut parent: Vec<_> = (0..n).collect();
    let mut rank = vec![0u8; n];
    let switches: Vec<_> = world
        .query::<(
            Entity,
            &Switch,
            &SwitchState,
            Option<&OutOfService>,
            &SwitchBusRef,
            Option<&SwitchOtherBusRef>,
        )>()
        .iter(world)
        .map(|(e, s, state, off, a, b)| (e, s.clone(), state.0, off.is_some(), a.0, b.map(|b| b.0)))
        .collect();
    for (e, s, closed, off, a, b) in switches {
        world
            .entity_mut(e)
            .insert(prepared(&s, &SwitchState(closed), off));
        if s.et != SwitchType::SwitchTwoBuses || !closed || off {
            continue;
        }
        let b = b.expect("prepared bus-bus switch must have both references");
        assert!(
            s.z_ohm.is_finite() && s.z_ohm >= 0.0,
            "switch impedance must be finite and nonnegative"
        );
        let va = world.get::<VNominal>(a.entity).unwrap().0.0;
        let vb = world.get::<VNominal>(b.entity).unwrap().0.0;
        assert!(
            (va - vb).abs() <= 1e-9 * va.abs().max(vb.abs()),
            "bus-bus switch connects different nominal voltage levels"
        );
        if model == SwitchModel::Merge && s.z_ohm == 0.0 {
            let ra = root(&mut parent, a.index);
            let rb = root(&mut parent, b.index);
            if ra != rb {
                if rank[ra] < rank[rb] {
                    parent[ra] = rb;
                } else {
                    parent[rb] = ra;
                    if rank[ra] == rank[rb] {
                        rank[ra] += 1;
                    }
                }
            }
        } else {
            // Existing RustPower model uses a purely resistive finite switch.
            // Type B approximates an ideal switch with a high conductance.
            let y = Complex::new(if s.z_ohm == 0.0 { 1e6 } else { 1.0 / s.z_ohm }, 0.0);
            let patch = nalgebra::Matrix2::new(y, -y, -y, y);
            world.entity_mut(e).insert((
                SwitchStamp,
                Port4MatPatch(patch),
                FromBusMatrixIndex(a.index),
                ToBusMatrixIndex(b.index),
            ));
        }
    }
    let mut groups = vec![usize::MAX; n];
    let mut count = 0;
    let mapping: HashMap<_, _> = (0..n)
        .map(|i| {
            let r = root(&mut parent, i);
            if groups[r] == usize::MAX {
                groups[r] = count;
                count += 1;
            }
            (i as u64, groups[r] as u64)
        })
        .collect();
    world.insert_resource(NodeMapping(mapping));
}
/// Compatibility entry point; normal preparation is dispatched by the installed plugin.
pub fn process_switch_state(world: &mut World) {
    prepare_switch_model(world, SwitchModel::Merge);
}
pub fn process_switch_state_admit(world: &mut World) {
    prepare_switch_model(world, SwitchModel::Admittance);
}
pub(crate) fn prepare_switches(world: &mut World) {
    prepare_open_lines(world);
    if let Some(model) = world.get_resource::<SwitchModel>().copied() {
        prepare_switch_model(world, model);
    }
    if world.contains_resource::<OpenLinePolicy>() {
        let states: Vec<_> = world
            .query::<(Entity, &Switch, &SwitchState, Has<OutOfService>)>()
            .iter(world)
            .map(|(e, s, state, off)| (e, prepared(s, state, off)))
            .collect();
        for (e, state) in states {
            world.entity_mut(e).insert(state);
        }
    }
}

fn prepare_open_lines(world: &mut World) {
    // Restore only previously modified lines. Recompute from current parameters,
    // so repeated preparation and parameter edits cannot compound reductions.
    let old: Vec<_> = world
        .query_filtered::<Entity, With<OpenLineModel>>()
        .iter(world)
        .collect();
    if old.is_empty() && !world.contains_resource::<OpenLinePolicy>() {
        return;
    }
    let wbase = world.resource::<PFCommonData>().wbase;
    for e in old {
        if let Some(params) = world.get::<LineParams>(e) {
            let patch = super::line::line_systems::closed_line_patch(params, wbase);
            world.entity_mut(e).insert(Port4MatPatch(patch));
        }
        world
            .entity_mut(e)
            .remove::<(OpenLineModel, OpenLineTerminals)>();
    }
    let Some(policy) = world.get_resource::<OpenLinePolicy>().copied() else {
        world.remove_resource::<PreparedOpenLinePolicy>();
        return;
    };
    let mut open: HashMap<Entity, OpenLineTerminals> = HashMap::new();
    for (connection, state) in world
        .query_filtered::<(&SwitchLineRef, &SwitchState), (Without<OutOfService>, With<Switch>)>()
        .iter(world)
    {
        if state.0 {
            continue;
        }
        let ends = open.entry(connection.entity).or_default();
        match connection.terminal {
            LineTerminal::From => ends.from = true,
            LineTerminal::To => ends.to = true,
        }
    }
    for (e, terminals) in open {
        let params = world
            .get::<LineParams>(e)
            .expect("validated switch line reference");
        let base = super::line::line_systems::closed_line_patch(params, wbase);
        let mut patch = nalgebra::Matrix2::zeros();
        let mut voltage_ratio = Complex::new(0.0, 0.0);
        if policy == OpenLinePolicy::PreserveCharging && terminals.from != terminals.to {
            let connected = if terminals.from { 1 } else { 0 };
            let series = -base[(0, 1)];
            // Factored Kron reduction avoids subtracting nearly equal series admittances.
            let shunt = 0.5
                * Complex::new(
                    1e-6 * params.g_us_per_km * params.length_km * params.parallel as f64,
                    wbase * 1e-9 * params.c_nf_per_km * params.length_km * params.parallel as f64,
                );
            patch[(connected, connected)] = shunt * (2.0 * series + shunt) / (series + shunt);
            voltage_ratio = series / (series + shunt);
        }
        world.entity_mut(e).insert((
            terminals,
            OpenLineModel {
                terminals,
                policy,
                voltage_ratio,
            },
            Port4MatPatch(patch),
        ));
    }
    world.insert_resource(PreparedOpenLinePolicy(policy));
}

/// Deterministic voltage priority: Slack > PV > PQ, then physical bus order.
pub fn node_aggregation_system(
    mapping: Res<NodeMapping>,
    mat: Res<PowerFlowMat>,
) -> (CscMatrix<f64>, CscMatrix<f64>) {
    let n = mapping.len();
    let m = mapping
        .values()
        .copied()
        .max()
        .map_or(0, |x| x as usize + 1);
    let mut priority = vec![0u8; n];
    for (i, &original) in mat.from_perm.iter().enumerate() {
        priority[original] = if i >= mat.npq + mat.npv {
            2
        } else if i >= mat.npq {
            1
        } else {
            0
        };
    }
    let mut representatives = vec![usize::MAX; m];
    let mut full = CooMatrix::new(n, m);
    for i in 0..n {
        let group = mapping[&(i as u64)] as usize;
        full.push(i, group, 1.0);
        let old = representatives[group];
        if old == usize::MAX || priority[i] > priority[old] {
            representatives[group] = i;
        }
    }
    let mut selected = CooMatrix::new(n, m);
    for (group, &i) in representatives.iter().enumerate() {
        selected.push(i, group, 1.0);
    }
    (CscMatrix::from(&full), CscMatrix::from(&selected))
}
pub fn handle_node_merge(
    In((full, selected)): In<(CscMatrix<f64>, CscMatrix<f64>)>,
    mapping: Res<NodeMapping>,
    mut mat: ResMut<PowerFlowMat>,
    mut cmd: Commands,
) {
    let n = full.nrows();
    let m = full.ncols();
    if n == m {
        cmd.remove_resource::<NodeAggRes>();
        return;
    }
    assert_eq!(
        mat.y_bus.nrows(),
        n,
        "switch reduction must start in physical bus space"
    );
    let indices: Vec<_> = (0..n).map(|i| mapping[&(i as u64)] as usize).collect();
    let reps: Vec<_> = (0..m)
        .map(|g| selected.row_indices()[selected.col_offsets()[g]])
        .collect();
    let mut priority = vec![0u8; n];
    for (i, &original) in mat.from_perm.iter().enumerate() {
        priority[original] = if i >= mat.npq + mat.npv {
            2
        } else if i >= mat.npq {
            1
        } else {
            0
        };
    }
    let mut pq = Vec::new();
    let mut pv = Vec::new();
    let mut slack = Vec::new();
    for (g, &r) in reps.iter().enumerate() {
        match priority[r] {
            2 => slack.push(g),
            1 => pv.push(g),
            _ => pq.push(g),
        }
    }
    let mut y = CooMatrix::new(m, m);
    for (r, c, &value) in mat.y_bus.triplet_iter() {
        y.push(indices[r], indices[c], value);
    }
    let mut s = DVector::zeros(m);
    for i in 0..n {
        s[indices[i]] += mat.s_bus[i];
    }
    let v = DVector::from_iterator(m, reps.iter().map(|&i| mat.v_bus_init[i]));
    mat.npq = pq.len();
    mat.npv = pv.len();
    mat.from_perm = pq.into_iter().chain(pv).chain(slack).collect();
    mat.to_perm = vec![0; m];
    for i in 0..m {
        let g = mat.from_perm[i];
        mat.to_perm[g] = i;
    }
    mat.y_bus = CscMatrix::from(&y);
    mat.s_bus = s;
    mat.v_bus_init = v;
    cmd.insert_resource(NodeAggRes {
        expand_mat: full,
        expand_mat_v: selected,
        original_to_merged: indices,
        representatives: reps,
    });
}
pub(crate) fn reduce_switch_nodes(world: &mut World) {
    if world.get_resource::<SwitchModel>() == Some(&SwitchModel::Merge) {
        if world
            .resource::<NodeMapping>()
            .iter()
            .all(|(original, merged)| original == merged)
        {
            world.remove_resource::<NodeAggRes>();
            return;
        }
        world
            .run_system_once(node_aggregation_system.pipe(handle_node_merge))
            .unwrap();
    }
}

#[cfg(test)]
#[path = "switch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "open_line_tests.rs"]
mod open_line_tests;
