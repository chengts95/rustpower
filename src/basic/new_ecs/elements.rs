use std::collections::HashMap;

pub use super::switch::*;
use crate::io::pandapower;
pub use crate::prelude::ExtGridNode;
pub use crate::prelude::PQNode;
pub use crate::prelude::PVNode;
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
use nalgebra::Complex;
use serde::{Deserialize, Serialize};
#[derive(Debug, Component, Deref, DerefMut, Default)]
pub struct VBase(pub f64);

/// Represents an admittance value in a power system.
///
/// `Admittance` is a wrapper around a complex number representing the admittance value.
#[derive(Component, Clone, Default, PartialEq, Debug)]
pub struct Admittance(pub Complex<f64>);

/// Represents a port with two integer values.
///
/// `Port2` is a structure holding two integer values typically used to denote a port in a system.
#[derive(Component, Deref, DerefMut, Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Resource, Deref, DerefMut)]
pub struct PPNetwork(pub pandapower::Network);

#[derive(Debug, Component, Deref, DerefMut)]
pub struct ElemIdx(pub usize);
#[derive(Debug, Component, Deref, DerefMut)]
pub struct PFNode(pub usize);

#[derive(Default, Debug, Resource)]
pub struct NodeLookup(pub HashMap<i64, Entity>);
#[derive(Debug, Component)]
pub struct AuxNode {
    pub bus: i64,
}
#[derive(Debug, Component)]
pub struct Line;
#[derive(Debug, Component)]
pub struct Transformer;
#[derive(Debug, Component)]
pub struct EShunt;

#[derive(Debug, Resource)]
pub struct PFCommonData {
    pub wbase: f64,
    pub sbase: f64,
}

#[derive(Debug, Component)]
pub enum NodeType {
    PQ(PQNode),
    PV(PVNode),
    EXT(ExtGridNode),
    AUX(AuxNode),
}
impl Default for NodeType {
    fn default() -> Self {
        NodeType::PQ(PQNode::default())
    }
}

impl From<PQNode> for NodeType {
    fn from(node: PQNode) -> Self {
        NodeType::PQ(node)
    }
}

impl From<PVNode> for NodeType {
    fn from(node: PVNode) -> Self {
        NodeType::PV(node)
    }
}

impl From<ExtGridNode> for NodeType {
    fn from(node: ExtGridNode) -> Self {
        NodeType::EXT(node)
    }
}

impl From<AuxNode> for NodeType {
    fn from(node: AuxNode) -> Self {
        NodeType::AUX(node)
    }
}
