use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};

use crate::basic::ecs::{elements::*, network::apply_permutation};

use super::systems::{init_bus_status, init_states, PowerFlowMat};
use crate::prelude::ecs::network::SolverStage::*;

#[derive(Event, Default, Debug, Clone, Copy)]
pub struct VoltageChangeEvent;
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct SBusChangeEvent;
#[derive(Event, Default, Debug, Clone, Copy)]
pub struct FullRebuildEvent;

#[derive(Event, Default, Debug, Clone, Copy)]
pub struct NodeTypeChangeEvent;

#[derive(Default, Debug, Clone, Copy)]
pub struct SimStateFlags {
    pub structure_dirty: bool, // Rebuild everything
    pub admit_dirty: bool,     // Admittance matrix changed
    pub injection_dirty: bool, // Sbus injection changed
    pub voltage_dirty: bool,   // Voltage changed
}

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
pub fn sbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    sbus: Query<(&BusID, &SBusInjPu), Changed<SBusInjPu>>,
) {
    println!("test sbus:{}",sbus.iter().count());
    for (bus_id, s) in sbus.iter() {
        let idx = pfmat.reorder_index(bus_id.0 as usize);
        pfmat.s_bus[idx] = s.0;
    }
  
}
pub fn vbus_pu_update(
    mut pfmat: ResMut<PowerFlowMat>,
    sbus: Query<(&TargetBus, &VBusPu), Changed<VBusPu>>,
) {
    for (bus_id, s) in sbus.iter() {
        let idx = pfmat.reorder_index(bus_id.0 as usize); // 原始 → 排序后的索引
        pfmat.s_bus[idx] = s.0;
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
