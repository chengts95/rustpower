use crate::basic::ecs::elements::bus::SnaptShotRegGroup;
use crate::io::pandapower::SwitchType;
use crate::prelude::ecs::defer_builder::DeferBundle;
use crate::prelude::ecs::defer_builder::DeferredBundleBuilder;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::component::Component;
use bevy_ecs::name::Name;
use derive_more::derive::{Deref, DerefMut};
use rustpower_proc_marco::DeferBundle;
/// Represents a network switch in the power flow network.
///
/// A switch connects two buses or a bus and an element, and can have a given impedance (z_ohm).
/// The switch state is defined by its type (`SwitchType`), the connected buses, and its impedance.
#[derive(Default, Debug, Clone, Component, serde::Serialize, serde::Deserialize)]
#[require(SwitchState)]
pub struct Switch {
    pub bus: i64,       // Identifier for the bus connected by the switch.
    pub element: i64,   // Identifier for the element connected by the switch.
    pub et: SwitchType, // Switch type that defines its behavior.
    pub z_ohm: f64,     // Impedance in ohms for the switch connection.
}
/// Represents the state of a switch (either open or closed).
///
/// The state (`true` for closed and `false` for open) is wrapped in the `SwitchState` component.
#[derive(
    Default, Debug, Clone, Component, Deref, DerefMut, serde::Serialize, serde::Deserialize,
)]
pub struct SwitchState(pub bool);

#[derive(DeferBundle, Default, Debug, Clone)]
pub struct SwitchBundle {
    pub switch: Switch,
    pub state: SwitchState,
    pub name: Option<Name>,
}
use crate::io::pandapower::Switch as PSwitch;

impl From<&PSwitch> for SwitchBundle {
    fn from(value: &PSwitch) -> Self {
        let switch = Switch {
            bus: value.bus,
            element: value.element,
            et: value.et.clone(),
            z_ohm: value.z_ohm,
        };
        let state = SwitchState(value.closed);
        Self {
            switch,
            state,
            name: value.name.clone().map(Name::new),
        }
    }
}
pub struct SwitchSnapShotReg;
impl SnaptShotRegGroup for SwitchSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<SwitchState>();
        reg.register::<Switch>();
    }
}

pub fn register_switch_snapshot(reg: &mut SnapshotRegistry) {
    SwitchSnapShotReg::register_snap_shot(reg);
}
