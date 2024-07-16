
use crate::io::pandapower;
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
use nalgebra::Complex;
use serde::{Deserialize, Serialize};


#[derive(Debug, Component, Deref, DerefMut)]
#[derive(Default)]
pub struct VBase(pub f64);


/// Represents an admittance value in a power system.
///
/// `Admittance` is a wrapper around a complex number representing the admittance value.
#[derive(Component, Clone, Default, PartialEq, Debug)]
pub struct Admittance(pub Complex<f64>);

/// Represents a port with two integer values.
///
/// `Port2` is a structure holding two integer values typically used to denote a port in a system.
#[derive(Component, Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Port2(pub nalgebra::Vector2<i64>);

/// Represents a branch with admittance and port information.
#[derive(Debug, Default, Bundle)]
pub struct AdmittanceBranch {
    /// The admittance value of the branch.
    pub y: Admittance,
    /// The port information of the branch.
    pub port: Port2,
    /// base voltage for per-unit values
    pub v_base: VBase,
}


/// Represents a node with specified power and bus information in a power system.
#[derive(Debug, Clone, Copy, Default, Component)]
pub struct PQNode {
    /// The complex power injected at the node.
    pub s: Complex<f64>,
    /// The bus identifier of the node.
    pub bus: i64,
}

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct PPNetwork(pub pandapower::Network);


