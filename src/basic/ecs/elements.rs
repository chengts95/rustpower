use std::collections::HashMap;

pub use super::switch::*;
use crate::io::pandapower;
pub use crate::prelude::ExtGridNode;
pub use crate::prelude::PQNode;
pub use crate::prelude::PVNode;
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
use nalgebra::Complex;

/// Base voltage for a bus or system node.
///
/// `VBase` is a wrapper around a `f64` value representing the base voltage of a node.
#[derive(Debug, Component, Deref, DerefMut, Default)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct VBase(pub f64);

/// Represents an admittance value in a power system.
///
/// `Admittance` is a wrapper around a complex number representing the admittance value,
/// which is essential for modeling the impedance in electrical systems.
#[derive(Component, Clone, Default, PartialEq, Debug)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct Admittance(pub Complex<f64>);

/// Represents a port with two integer values.
///
/// `Port2` holds two integer values (typically bus or node indices) used to define
/// the connectivity between two entities in the power grid (like branches).
#[derive(Component, Deref, DerefMut, Default, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct Port2(pub nalgebra::Vector2<i64>);

/// Represents a branch with admittance and port information.
///
/// `AdmittanceBranch` bundles together an admittance value, port information,
/// and base voltage, which are essential to define electrical branches like lines or transformers.
#[derive(Debug, Default, Bundle)]
pub struct AdmittanceBranch {
    /// The admittance value of the branch.
    pub y: Admittance,
    /// The port information (node indices) of the branch.
    pub port: Port2,
    /// Base voltage for per-unit system calculations.
    pub v_base: VBase,
}

/// Wrapper around a `pandapower::Network` structure.
///
/// This resource contains the complete power network data, loaded from the external Pandapower library.
#[derive(Debug, Resource, Deref, DerefMut)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PPNetwork(pub pandapower::Network);

/// Component that stores an index, typically referring to an element within the power network.
#[derive(Debug, Component, Deref, DerefMut)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct ElemIdx(pub usize);

/// Component that stores an index, typically referring to a power flow node within the network.
#[derive(Debug, Component, Deref, DerefMut)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PFNode(pub usize);

/// Resource that maps node indices (i64) to ECS entities.
///
/// `NodeLookup` helps in quickly finding the ECS entity corresponding to a node in the power flow network.
#[derive(Default, Debug, Resource)]
pub struct NodeLookup(pub HashMap<i64, Entity>);

/// Component representing an auxiliary node in the network.
///
/// `AuxNode` typically refers to a node with a special function, defined by its bus index.
#[derive(Debug, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct AuxNode {
    pub bus: i64,
}

/// Marker component for a line element in the power system.
#[derive(Debug, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct Line;

/// Marker component for a transformer element in the power system.
#[derive(Debug, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct Transformer;

/// Marker component for a shunt element in the power system.
#[derive(Debug, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct EShunt;

/// Resource holding common base values for the power flow calculation.
///
/// `PFCommonData` contains the base frequency (`wbase`) and base power (`sbase`) for per-unit system calculations.
#[derive(Debug, Resource)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PFCommonData {
    pub wbase: f64, // Base frequency (typically in rad/s).
    pub sbase: f64, // Base power (typically in MVA).
}

/// Enum representing different types of nodes in the power flow network.
///
/// `NodeType` differentiates between various node types such as PQ nodes, PV nodes, external grid nodes, and auxiliary nodes.
#[derive(Debug, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeType {
    PQ(PQNode),       // Load bus (PQ bus)
    PV(PVNode),       // Generator bus (PV bus)
    EXT(ExtGridNode), // External grid node
    AUX(AuxNode),     // Auxiliary node
}

/// Default implementation for `NodeType`, which defaults to a `PQNode`.
impl Default for NodeType {
    fn default() -> Self {
        NodeType::PQ(PQNode::default())
    }
}

/// Allows converting a `PQNode` into a `NodeType`.
impl From<PQNode> for NodeType {
    fn from(node: PQNode) -> Self {
        NodeType::PQ(node)
    }
}

/// Allows converting a `PVNode` into a `NodeType`.
impl From<PVNode> for NodeType {
    fn from(node: PVNode) -> Self {
        NodeType::PV(node)
    }
}

/// Allows converting an `ExtGridNode` into a `NodeType`.
impl From<ExtGridNode> for NodeType {
    fn from(node: ExtGridNode) -> Self {
        NodeType::EXT(node)
    }
}

/// Allows converting an `AuxNode` into a `NodeType`.
impl From<AuxNode> for NodeType {
    fn from(node: AuxNode) -> Self {
        NodeType::AUX(node)
    }
}
