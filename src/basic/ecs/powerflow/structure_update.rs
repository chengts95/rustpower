use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use crate::basic::ecs::network::PowerFlowSolver;
use crate::basic::ecs::{elements::*, network::apply_permutation};

use super::systems::{PowerFlowMat, init_states};
use crate::prelude::ecs::network::SolverStage::*;

/// Fired when the voltage (VBusPu) of one or more nodes has changed.
/// Triggers voltage vector update in the solver matrix.
#[derive(Message, Default, Debug, Clone, Copy)]
pub struct VoltageChangeEvent;

/// Fired when the SBus injection (SBusInjPu) has changed at any node.
/// Indicates active/reactive power has been updated.
#[derive(Message, Default, Debug, Clone, Copy)]
pub struct SBusChangeEvent;

/// Forces a complete structure rebuild, including YBus, node tags, etc.
/// Typically triggered by initialization or topology changes.
#[derive(Message, Default, Debug, Clone, Copy)]
pub struct FullRebuildEvent;

/// Indicates that the bus type (PV/PQ/Slack) of one or more nodes has changed.
/// Requires matrix structure update (e.g., PV to PQ downgrades).
#[derive(Message, Default, Debug, Clone, Copy)]
pub struct NodeTypeChangeEvent;

/// Flags representing which parts of the simulation state are dirty and need update.
///
/// Set by [`event_update`] and consumed by [`structure_update`] to determine minimal work needed.
#[derive(Default, Debug, Clone, Copy)]
pub struct SimStateFlags {
    /// Run the full PFInit schedule (topology changed: ingestion, element
    /// setup, re-labeling, matrices). Triggered by [`FullRebuildEvent`].
    pub full_dirty: bool,
    /// Re-derive matrices from the CURRENT node tags (no relabeling — this
    /// must preserve e.g. qlim's PV->PQ demotions). Triggered by
    /// [`NodeTypeChangeEvent`].
    pub structure_dirty: bool,
    /// Rebuild admittance (YBus) matrix only.
    pub admit_dirty: bool,
    /// Update SBus power injection vector.
    pub injection_dirty: bool,
    /// Update VBus voltage vector.
    pub voltage_dirty: bool,
}

/// What the last `structure_update` invocation actually did. Read by callers
/// (e.g. the Python SolveReport) for observability.
#[derive(Resource, Default, Clone, Copy)]
pub struct LastStructureAction {
    pub full_rebuild: bool,
}

/// Aggregates all recent event types into a unified [`SimStateFlags`] structure.
/// Clears all event queues after reading them.
pub fn event_update(
    mut e_sbus: MessageReader<SBusChangeEvent>,
    mut e_vbus: MessageReader<VoltageChangeEvent>,
    mut e_full: MessageReader<FullRebuildEvent>,
    mut e_node_type: MessageReader<NodeTypeChangeEvent>,
) -> SimStateFlags {
    let mut flags = SimStateFlags::default();

    if !e_full.is_empty() {
        flags.full_dirty = true;
    }
    if !e_node_type.is_empty() {
        flags.structure_dirty = true;
    }
    if !e_sbus.is_empty() {
        flags.injection_dirty = true;
    }
    if !e_vbus.is_empty() {
        flags.voltage_dirty = true;
    }

    e_full.clear();
    e_node_type.clear();
    e_sbus.clear();
    e_vbus.clear();

    flags
}

pub fn reset_solvers(world: &mut World) {
    use crate::basic::solver::*;
    world.remove_resource::<super::prepare::CachedNewtonResult>();

    if let Some(mut solver) = world.get_resource_mut::<PowerFlowSolver>() {
        solver.solver.reset();
    }
    if let Some(mut cache) = world.get_resource_mut::<crate::basic::newtonpf::NewtonCache>() {
        *cache = Default::default();
    }
    super::prepare::invalidate_dcpf_model(world);
}
/// Re-syncs the full `s_bus` vector in [`PowerFlowMat`] from the `SBusInjPu`
/// components. Triggered by the coarse [`SBusChangeEvent`] (native writers,
/// e.g. time series, update the components themselves and fire the event).
/// Deliberately tick-free: a full O(n) copy is cheap and always correct,
/// whereas `Changed<T>` semantics depend on observer tick bookkeeping.
pub fn sbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    agg: Option<Res<NodeAggRes>>,
    sbus: Query<(&BusMatrixIndex, &SBusInjPu, Option<&super::qlim::QLimited>)>,
) {
    if let Some(a) = agg {
        pfmat.s_bus.fill(num_complex::Complex64::new(0.0, 0.0));
        for (bus_id, s, bound) in sbus {
            let idx = pfmat.reorder_index(a.original_to_merged[bus_id.0]);
            pfmat.s_bus[idx] +=
                s.0 + num_complex::Complex64::new(0.0, bound.map_or(0.0, |q| q.q_pu));
        }
    } else {
        for (bus_id, s, bound) in sbus {
            let idx = pfmat.reorder_index(bus_id.0);
            pfmat.s_bus[idx] =
                s.0 + num_complex::Complex64::new(0.0, bound.map_or(0.0, |q| q.q_pu));
        }
    }
}
/// Re-syncs the full `v_bus` vector in [`PowerFlowMat`] from the `VBusPu`
/// components. Triggered by the coarse [`VoltageChangeEvent`]. Tick-free for
/// the same reason as [`sbus_pu_update`].
pub fn vbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    agg: Option<Res<NodeAggRes>>,
    vbus: Query<(&BusMatrixIndex, &VBusPu)>,
) {
    for (bus_id, s) in vbus {
        let original = if let Some(a) = &agg {
            let group = a.original_to_merged[bus_id.0];
            if a.representatives[group] != bus_id.0 {
                continue;
            }
            group
        } else {
            bus_id.0
        };
        let idx = pfmat.reorder_index(original);
        pfmat.v_bus_init[idx] = s.0;
    }
}

pub fn structure_update(world: &mut World) {
    let mut flags = world.run_system_cached(event_update).unwrap();
    flags.full_dirty |= !world.contains_resource::<PowerFlowMat>();
    let earlier_rebuild = world
        .get_resource::<super::prepare::SolvePass>()
        .is_some_and(|pass| pass.continuation)
        && world.resource::<LastStructureAction>().full_rebuild;
    world.insert_resource(LastStructureAction {
        full_rebuild: earlier_rebuild || flags.full_dirty,
    });

    // Topology changed: run the single full-rebuild pipeline.
    if flags.full_dirty {
        world.run_schedule(super::pf_init::PFInit);
        return;
    }
    if flags.structure_dirty || flags.admit_dirty {
        world.run_system_cached(reset_solvers).unwrap();
        if flags.admit_dirty {
            world.run_system_cached(init_states).unwrap();
        } else {
            world
                .run_system_cached(super::systems::repartition_states)
                .unwrap();
        }
        world.run_system_cached(apply_permutation).unwrap();
        // Q-limit continuation voltages are in original bus order. Apply the
        // new permutation explicitly after rebuilding the partition.
        let continuation = world
            .get_resource_mut::<super::prepare::SolvePass>()
            .and_then(|mut pass| pass.voltage.take());
        if let Some(v) = continuation {
            let mut mat = world.resource_mut::<PowerFlowMat>();
            for idx in 0..mat.from_perm.len() {
                mat.v_bus_init[idx] = v[mat.from_perm[idx]];
            }
        }
    } else {
        if flags.injection_dirty {
            world.run_system_cached(sbus_pu_update).unwrap();
        }
        if flags.voltage_dirty {
            world.run_system_cached(vbus_pu_update).unwrap();
            // The DC right-hand-side contains the slack-angle contribution.
            super::prepare::invalidate_dcpf_model(world);
        }
    }
}

/// Plugin responsible for maintaining matrix consistency in response to state changes.
///
/// Tracks and responds to structural events such as:
/// - Changes in bus types (PV → PQ, etc)
/// - Power injection changes
/// - Voltage setpoint updates
/// - Full topology/matrix rebuild triggers
///
/// # Added Events:
/// - [`VoltageChangeEvent`]
/// - [`SBusChangeEvent`]
/// - [`FullRebuildEvent`]
/// - [`NodeTypeChangeEvent`]
///
/// # System Registration:
/// Adds [`structure_update`] system to `Update` stage, between [`BeforeSolve`] and [`Solve`].
#[derive(Default)]
pub struct StructureUpdatePlugin;

impl Plugin for StructureUpdatePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<VoltageChangeEvent>();
        app.add_message::<SBusChangeEvent>();
        app.add_message::<FullRebuildEvent>();
        app.add_message::<NodeTypeChangeEvent>();
        app.add_message::<super::mutation::ParamDiff>();
        app.init_resource::<LastStructureAction>();
        // The single definition of "full rebuild", runnable on demand.
        app.add_schedule(super::pf_init::build_pf_init_schedule());
        app.init_resource::<super::prepare::DcpfModelRevision>();
        app.add_schedule(super::prepare::build_prepare_schedule());
        app.add_systems(
            Update,
            super::prepare::prepare_solver_inputs
                .after(BeforeSolve)
                .before(Solve),
        );
    }
}
