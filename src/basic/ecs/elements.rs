pub mod bus;
pub mod ele_process;
pub mod generator;
pub mod line;
pub mod load;
pub mod sgen;
pub mod shunt;
pub mod switch;
pub mod trans;
pub mod units;
use crate::io::pandapower;

use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
pub use ele_process::*;
use nalgebra::Complex;

/// Base voltage for a bus or system node.
///
/// `VBase` is a wrapper around a `f64` value representing the base voltage of a node.
#[derive(Debug, Component, Deref, DerefMut, Default, serde::Serialize, serde::Deserialize)]
pub struct VBase(pub f64);

/// Represents an admittance value in a power system.
///
/// `Admittance` is a wrapper around a complex number representing the admittance value,
/// which is essential for modeling the impedance in electrical systems.
#[derive(Component, Clone, Default, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Admittance(pub Complex<f64>);

/// Represents a port with two integer values.
///
/// `Port2` holds two integer values (typically bus or node indices) used to define
/// the connectivity between two entities in the power grid (like branches).
#[derive(
    Component,
    Deref,
    DerefMut,
    Default,
    Debug,
    Clone,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Port2(pub nalgebra::Vector2<i64>);

impl Port2 {
    pub fn new(a: i64, b: i64) -> Self {
        Self(nalgebra::Vector2::new(a, b))
    }
}

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
#[derive(Debug, Resource, Deref, DerefMut, serde::Serialize, serde::Deserialize)]
pub struct PPNetwork(pub pandapower::Network);

/// Resource that maps node indices (i64) to ECS entities.
///
/// `NodeLookup` helps in quickly finding the ECS entity corresponding to a node in the power flow network.
#[derive(Default, Debug, Resource, Clone)]
pub struct NodeLookup {
    /// bus_id → entity 映射
    pub forward: Vec<Option<Entity>>,
    count: usize,
}

/// Component representing an auxiliary node in the network.
///
/// `AuxNode` typically refers to a node with a special function, defined by its bus index.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize)]
pub struct AuxNode {
    pub bus: i64,
}

/// Marker component for a line element in the power system.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize, Clone)]
pub struct Line;

/// Marker component for a transformer element in the power system.
#[derive(Debug, Component, Clone, serde::Serialize, serde::Deserialize)]
pub struct Transformer;

/// Dense flat index of a branch (Line or Trafo) in the BranchCollection pool.
#[derive(
    Debug, Component, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct BranchIndex(pub usize);

/// Marker component for a shunt element in the power system.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize)]
pub struct EShunt;

/// Resource holding common base values for the power flow calculation.
///
/// `PFCommonData` contains the base frequency (`wbase`) and base power (`sbase`) for per-unit system calculations.
#[derive(Debug, Resource, serde::Serialize, serde::Deserialize)]
pub struct PFCommonData {
    pub wbase: f64, // Base frequency (typically in rad/s).
    pub f_hz: f64,  // Base frequency (typically in Hz).
    pub sbase: f64, // Base power (typically in MVA).
}

impl NodeLookup {
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = (i64, Entity)> + '_ {
        self.forward
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|e| (i as i64, e)))
    }
    pub fn insert(&mut self, bus_id: i64, entity: Entity) {
        let idx = bus_id as usize;
        if self.forward.len() <= idx {
            self.forward.resize_with(idx + 1, || None);
        }
        if self.forward[idx].is_none() {
            self.count += 1;
        }
        self.forward[idx] = Some(entity);
    }
    pub fn remove_entity(&mut self, entity: Entity) {
        for slot in self.forward.iter_mut() {
            if *slot == Some(entity) {
                *slot = None;
                self.count = self.count.saturating_sub(1);
            }
        }
    }

    pub fn remove_id(&mut self, bus_id: i64) {
        if let Some(slot) = self.forward.get_mut(bus_id as usize) {
            if slot.is_some() {
                *slot = None;
                self.count = self.count.saturating_sub(1);
            }
        }
    }

    pub fn get_entity(&self, bus_id: i64) -> Option<Entity> {
        self.forward.get(bus_id as usize).and_then(|x| *x)
    }

    pub fn get_id(&self, entity: Entity) -> Option<i64> {
        self.forward
            .iter()
            .position(|&x| x == Some(entity))
            .map(|idx| idx as i64)
    }

    pub fn contains_id(&self, bus_id: i64) -> bool {
        self.forward
            .get(bus_id as usize)
            .is_some_and(|e| e.is_some())
    }

    pub fn contains_entity(&self, entity: Entity) -> bool {
        self.forward.iter().any(|&e| e == Some(entity))
    }
}
