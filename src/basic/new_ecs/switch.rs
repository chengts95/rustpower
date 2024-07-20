use crate::io::pandapower::SwitchType;
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};

/// Represents a switch in the network.
#[derive(Default, Debug, Clone, Component)]
pub struct Switch {
    pub bus: i64,
    pub element: i64,
    pub et: SwitchType,
    pub z_ohm: f64,
}

/// Represents a switch state in the network.
#[derive(Default, Debug, Clone, Component, Deref, DerefMut)]
pub struct SwitchState(pub bool);

/// Merge 2 nodes
#[derive(Default, Debug, Clone, Component)]
pub struct MergeNode(pub usize, pub usize);
