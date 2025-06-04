use std::collections::HashMap;
mod bus;
mod ele_process;
mod generator;
mod line;
mod load;
mod sgen;
mod shunt;
mod switch;
mod trans;
mod units;
use crate::io::pandapower;

use bevy_ecs::entity::EntityHash;
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

/// Component that stores an index, typically referring to an element within the power network.
#[derive(Debug, Component, Deref, DerefMut, serde::Serialize, serde::Deserialize)]
pub struct ElemIdx(pub usize);

/// Component that stores an index, typically referring to a power flow node within the network.
#[derive(Debug, Component, Deref, DerefMut, serde::Serialize, serde::Deserialize)]
pub struct PFNode(pub usize);

/// Resource that maps node indices (i64) to ECS entities.
///
/// `NodeLookup` helps in quickly finding the ECS entity corresponding to a node in the power flow network.
#[derive(Default, Debug, Resource)]
pub struct NodeLookup {
    /// bus_id → entity 映射
    pub forward: Vec<Option<Entity>>,
    /// entity → bus_id 映射
    pub reverse: HashMap<Entity, i64, EntityHash>,
}

/// Component representing an auxiliary node in the network.
///
/// `AuxNode` typically refers to a node with a special function, defined by its bus index.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize)]
pub struct AuxNode {
    pub bus: i64,
}

/// Marker component for a line element in the power system.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize)]
pub struct Line;

/// Marker component for a transformer element in the power system.
#[derive(Debug, Component, serde::Serialize, serde::Deserialize)]
pub struct Transformer;

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
        self.reverse.len()
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

        if let Some(old_id) = self.reverse.insert(entity, bus_id) {
            if let Some(e) = self.forward.get_mut(old_id as usize) {
                if *e == Some(entity) {
                    *e = None;
                }
            }
        }

        self.forward[idx] = Some(entity);
    }
    pub fn remove_entity(&mut self, entity: Entity) {
        if let Some(id) = self.reverse.remove(&entity) {
            if let Some(slot) = self.forward.get_mut(id as usize) {
                if *slot == Some(entity) {
                    *slot = None;
                }
            }
        }
    }

    pub fn remove_id(&mut self, bus_id: i64) {
        if let Some(Some(entity)) = self.forward.get_mut(bus_id as usize) {
            self.reverse.remove(entity);
        }
    }

    pub fn get_entity(&self, bus_id: i64) -> Option<Entity> {
        self.forward.get(bus_id as usize).and_then(|x| *x)
    }

    pub fn get_id(&self, entity: Entity) -> Option<i64> {
        self.reverse.get(&entity).copied()
    }

    pub fn contains_id(&self, bus_id: i64) -> bool {
        self.forward
            .get(bus_id as usize)
            .map_or(false, |e| e.is_some())
    }

    pub fn contains_entity(&self, entity: Entity) -> bool {
        self.reverse.contains_key(&entity)
    }
}
