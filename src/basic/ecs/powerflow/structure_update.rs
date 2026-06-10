use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

#[cfg(feature = "rsparse")]
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

    if let Some(mut solver) = world.get_resource_mut::<PowerFlowSolver>() {
        solver.solver.reset();
    }
}
/// Re-syncs the full `s_bus` vector in [`PowerFlowMat`] from the `SBusInjPu`
/// components. Triggered by the coarse [`SBusChangeEvent`] (native writers,
/// e.g. time series, update the components themselves and fire the event).
/// Deliberately tick-free: a full O(n) copy is cheap and always correct,
/// whereas `Changed<T>` semantics depend on observer tick bookkeeping.
pub fn sbus_pu_update(mut pfmat: ResMut<PowerFlowMat>, sbus: Query<(&BusID, &SBusInjPu)>) {
    for (bus_id, s) in sbus {
        let idx = pfmat.reorder_index(bus_id.0 as usize);
        pfmat.s_bus[idx] = s.0;
    }
}
/// Re-syncs the full `v_bus` vector in [`PowerFlowMat`] from the `VBusPu`
/// components. Triggered by the coarse [`VoltageChangeEvent`]. Tick-free for
/// the same reason as [`sbus_pu_update`].
pub fn vbus_pu_update(mut pfmat: ResMut<PowerFlowMat>, vbus: Query<(&BusID, &VBusPu)>) {
    for (bus_id, s) in vbus {
        let idx = pfmat.reorder_index(bus_id.0 as usize);
        pfmat.v_bus_init[idx] = s.0;
    }
}

pub fn structure_update(world: &mut World) {
    let flags = world.run_system_cached(event_update).unwrap();
    world.insert_resource(LastStructureAction { full_rebuild: flags.full_dirty });

    // Topology changed: run the single full-rebuild pipeline.
    if flags.full_dirty {
        let _ = world.try_run_schedule(super::pf_init::PFInit);
        return;
    }
    // Until init has produced the solver matrices there is nothing to patch;
    // the pending changes are captured by the full rebuild instead.
    if !world.contains_resource::<PowerFlowMat>() {
        return;
    }
    if flags.structure_dirty || flags.admit_dirty {
        //TODO: this should only update ybus or node structure
        world.run_system_cached(reset_solvers).unwrap();
        world.run_system_cached(init_states).unwrap();
        world.run_system_cached(apply_permutation).unwrap();
    } else {
        if flags.injection_dirty {
            world.run_system_cached(sbus_pu_update).unwrap();
        }
        if flags.voltage_dirty {
            world.run_system_cached(vbus_pu_update).unwrap();
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
        // The mutation-bus consumer runs as an ordinary system right before
        // structure_update, so the change events it emits are seen this frame.
        app.add_systems(
            Update,
            super::mutation::consume_param_diffs
                .after(BeforeSolve)
                .before(structure_update),
        );
        app.add_systems(Update, structure_update.after(BeforeSolve).before(Solve));
    }
}
