use crate::basic::ecs::elements::bus::SnaptShotRegGroup;
use crate::io::pandapower::SwitchType;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::derive::{Deref, DerefMut};
/// Represents a network switch in the power flow network.
///
/// A switch connects two buses or a bus and an element, and can have a given impedance (z_ohm).
/// The switch state is defined by its type (`SwitchType`), the connected buses, and its impedance.
#[derive(Default, Debug, Clone, Component, serde::Serialize, serde::Deserialize)]
#[require(SwitchState)]
pub struct Switch {
    pub bus: i64,     // Identifier for the bus connected by the switch.
    pub element: i64, // Identifier for the element connected by the switch.
    #[serde(with = "switch_type_label")]
    pub et: SwitchType, // Switch type that defines its behavior.
    pub z_ohm: f64,   // Impedance in ohms for the switch connection.
}

// Use the same text labels as pandapower JSON. Arrow enum unions cannot be
// written to Parquet, whereas a string column round-trips through snapshots.
mod switch_type_label {
    use super::SwitchType;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &SwitchType, serializer: S) -> Result<S::Ok, S::Error> {
        let label = match value {
            SwitchType::SwitchBusLine => "l",
            SwitchType::SwitchBusTransformer => "t",
            SwitchType::SwitchBusTransformer3w => "t3",
            SwitchType::SwitchTwoBuses => "b",
            SwitchType::Unknown => "Unknown",
        };
        serializer.serialize_str(label)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<SwitchType, D::Error> {
        let label = String::deserialize(deserializer)?;
        Ok(SwitchType::from(label.as_str()))
    }
}
/// Represents the state of a switch (either open or closed).
///
/// The state (`true` for closed and `false` for open) is wrapped in the `SwitchState` component.
#[derive(
    Default, Debug, Clone, Component, Deref, DerefMut, serde::Serialize, serde::Deserialize,
)]
pub struct SwitchState(pub bool);

use rustpower_proc_marco::DeferBundle;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineTerminal {
    From,
    To,
}
/// Resolved switch-to-line connection. Reconstructed from labels after archive loading.
#[derive(Component, Debug, Clone, Copy)]
pub struct SwitchLineRef {
    pub entity: Entity,
    pub terminal: LineTerminal,
    pub(crate) label: i64,
    pub(crate) world_id: u64,
}
/// Effective terminal state; present only on lines with an active open switch.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenLineTerminals {
    pub from: bool,
    pub to: bool,
}
/// Policy applied by the optional line-switch plugin before core Ybus assembly.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenLinePolicy {
    #[default]
    PreserveCharging,
    DisableBranch,
}
/// Derived line-side voltage reconstruction. Never serialized.
#[derive(Component, Debug, Clone, Copy)]
pub(crate) struct OpenLineModel {
    pub terminals: OpenLineTerminals,
    pub policy: OpenLinePolicy,
    pub voltage_ratio: num_complex::Complex64,
}
#[derive(Resource, Clone, Copy)]
pub(crate) struct PreparedOpenLinePolicy(pub OpenLinePolicy);
