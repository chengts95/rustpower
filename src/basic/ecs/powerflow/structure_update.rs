use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};

use crate::basic::ecs::{elements::*, network::apply_permutation};

use super::systems::{PowerFlowMat, init_states};
use crate::prelude::ecs::network::SolverStage::*;

/// Fired when the voltage (VBusPu) of one or more nodes has changed.
/// Triggers voltage vector update in the solver matrix.
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct VoltageChangeEvent;

/// Fired when the SBus injection (SBusInjPu) has changed at any node.
/// Indicates active/reactive power has been updated.
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct SBusChangeEvent;

/// Forces a complete structure rebuild, including YBus, node tags, etc.
/// Typically triggered by initialization or topology changes.
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct FullRebuildEvent;

/// Indicates that the bus type (PV/PQ/Slack) of one or more nodes has changed.
/// Requires matrix structure update (e.g., PV to PQ downgrades).
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct NodeTypeChangeEvent;

/// Flags representing which parts of the simulation state are dirty and need update.
///
/// Set by [`event_update`] and consumed by [`structure_update`] to determine minimal work needed.
#[derive(Default, Debug, Clone, Copy)]
pub struct SimStateFlags {
    /// Rebuild entire structure including topology, bus types, admittance matrix.
    pub structure_dirty: bool,
    /// Rebuild admittance (YBus) matrix only.
    pub admit_dirty: bool,
    /// Update SBus power injection vector.
    pub injection_dirty: bool,
    /// Update VBus voltage vector.
    pub voltage_dirty: bool,
}

/// Aggregates all recent event types into a unified [`SimStateFlags`] structure.
/// Clears all event queues after reading them.
pub fn event_update(
    mut e_sbus: EventReader<SBusChangeEvent>,
    mut e_vbus: EventReader<VoltageChangeEvent>,
    mut e_full: EventReader<FullRebuildEvent>,
    mut e_node_type: EventReader<NodeTypeChangeEvent>,
) -> SimStateFlags {
    let mut flags = SimStateFlags::default();

    if !e_full.is_empty() {
        flags.structure_dirty = true;
        flags.admit_dirty = true;
        flags.injection_dirty = true;
        flags.voltage_dirty = true;
    } else {
        if !e_node_type.is_empty() {
            flags.structure_dirty = true;
        }
        if !e_sbus.is_empty() {
            flags.injection_dirty = true;
        }
        if !e_vbus.is_empty() {
            flags.voltage_dirty = true;
        }
    }

    e_full.clear();
    e_node_type.clear();
    e_sbus.clear();
    e_vbus.clear();

    flags
}

/// Updates the `s_bus` vector in [`PowerFlowMat`] when [`SBusInjPu`] values have changed.
pub fn sbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    sbus: Query<(&BusID, &SBusInjPu), Changed<SBusInjPu>>,
) {
    println!("test sbus:{}", sbus.iter().count());
    for (bus_id, s) in sbus.iter() {
        let idx = pfmat.reorder_index(bus_id.0 as usize);
        pfmat.s_bus[idx] = s.0;
    }
}

/// Updates the `v_bus` vector in [`PowerFlowMat`] when [`VBusPu`] values have changed.
/// Note: this assumes target voltage values are directly applied as injected power.
pub fn vbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    sbus: Query<(&TargetBus, &VBusPu), Changed<VBusPu>>,
) {
    for (bus_id, v) in sbus.iter() {
        let idx = pfmat.reorder_index(bus_id.0 as usize); // 原始 → 排序后的索引
        pfmat.v_bus_init[idx] = v.0;
    }
}

pub fn structure_update(world: &mut World) {
    let flags = world.run_system_once(event_update).unwrap();
    if flags.structure_dirty || flags.admit_dirty {
        //TODO: this should only update ybus or node structure
        world.run_system_once(init_states).unwrap();
        world.run_system_once(apply_permutation).unwrap();
    } else {
        if flags.injection_dirty {
            world.run_system_once(sbus_pu_update).unwrap();
        }
        if flags.voltage_dirty {
            world.run_system_once(vbus_pu_update).unwrap();
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
        app.add_event::<VoltageChangeEvent>();
        app.add_event::<SBusChangeEvent>();
        app.add_event::<FullRebuildEvent>();
        app.add_event::<NodeTypeChangeEvent>();
        app.add_systems(Update, structure_update.after(BeforeSolve).before(Solve));
    }
}
