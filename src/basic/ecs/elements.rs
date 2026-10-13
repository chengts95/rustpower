pub mod bus;
pub mod ele_process;
pub mod generator;
pub mod line;
pub mod load;
pub mod sgen;
pub mod shunt;
pub mod switch;
pub mod topology;
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
#[derive(Debug, Resource, Clone)]
pub struct NodeLookup {
    pub forward: std::collections::HashMap<i64, Entity>,
    reverse: std::collections::HashMap<Entity, i64>,
    dense: Vec<Entity>,
    indices: std::collections::HashMap<Entity, usize>,
    pub revision: u64,
    pub(crate) world_id: u64,
}

/// Physical matrix offset; independent of the API label and solver permutation.
#[derive(Component, Debug, Clone, Copy)]
pub struct BusMatrixIndex(pub usize);

/// A resolved reference and its cached physical matrix offset.
/// Not serialized: archives store labels and resolve references in the new world.
#[derive(Debug, Clone, Copy)]
pub struct BusConnection {
    pub entity: Entity,
    pub index: usize,
    pub label: i64,
    pub(crate) world_id: u64,
}
macro_rules! bus_reference {
    ($($name:ident),*) => { $(
        #[derive(Component, Debug, Clone, Copy)]
        pub struct $name(pub BusConnection);
    )* };
}
bus_reference!(
    FromBusRef,
    ToBusRef,
    TargetBusRef,
    SwitchBusRef,
    SwitchOtherBusRef
);

/// Compact numerical caches: hot loops stream usize offsets without Entity payloads.
#[derive(Component, Debug, Clone, Copy)]
pub struct FromBusMatrixIndex(pub usize);
#[derive(Component, Debug, Clone, Copy)]
pub struct ToBusMatrixIndex(pub usize);
#[derive(Component, Debug, Clone, Copy)]
pub struct TargetBusMatrixIndex(pub usize);
#[derive(Component, Debug, Clone, Copy)]
pub struct Port2MatrixIndex(pub [Option<usize>; 2]);

/// Ground is explicit in prepared ports, never encoded as a matrix offset.
#[derive(Component, Debug, Clone, Copy)]
pub struct Port2Ref(pub [Option<BusConnection>; 2]);

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

impl Default for NodeLookup {
    fn default() -> Self {
        static NEXT_WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            forward: Default::default(),
            reverse: Default::default(),
            dense: Vec::new(),
            indices: Default::default(),
            revision: 0,
            world_id: NEXT_WORLD.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl NodeLookup {
    pub fn len(&self) -> usize {
        self.dense.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (i64, Entity)> + '_ {
        self.dense.iter().map(|&e| (self.reverse[&e], e))
    }
    pub fn dense_iter(&self) -> impl Iterator<Item = (usize, Entity)> + '_ {
        self.dense.iter().copied().enumerate()
    }
    pub fn get_by_index(&self, index: usize) -> Option<Entity> {
        self.dense.get(index).copied()
    }
    /// Bind a world-local bus Entity for native construction of connectivity.
    pub fn connection(&self, entity: Entity) -> Option<BusConnection> {
        Some(BusConnection {
            entity,
            label: self.get_id(entity)?,
            index: self.matrix_index(entity)?,
            world_id: self.world_id,
        })
    }
    pub fn matrix_index(&self, entity: Entity) -> Option<usize> {
        self.indices.get(&entity).copied()
    }
    pub fn insert(&mut self, bus_id: i64, entity: Entity) {
        if let Some(existing) = self.forward.get(&bus_id) {
            assert_eq!(*existing, entity, "duplicate BusID {bus_id}");
            return;
        }
        assert!(
            !self.reverse.contains_key(&entity),
            "entity already has a bus label"
        );
        self.forward.insert(bus_id, entity);
        self.reverse.insert(entity, bus_id);
        self.indices.insert(entity, self.dense.len());
        self.dense.push(entity);
    }
    pub fn remove_entity(&mut self, entity: Entity) {
        if let Some(id) = self.reverse.remove(&entity) {
            self.forward.remove(&id);
            self.dense.retain(|&e| e != entity);
            self.indices.clear();
            for (i, &e) in self.dense.iter().enumerate() {
                self.indices.insert(e, i);
            }
        }
    }
    pub fn remove_id(&mut self, bus_id: i64) {
        if let Some(entity) = self.get_entity(bus_id) {
            self.remove_entity(entity);
        }
    }
    pub fn get_entity(&self, bus_id: i64) -> Option<Entity> {
        self.forward.get(&bus_id).copied()
    }
    pub fn get_id(&self, entity: Entity) -> Option<i64> {
        self.reverse.get(&entity).copied()
    }
    pub fn contains_id(&self, bus_id: i64) -> bool {
        self.forward.contains_key(&bus_id)
    }
    pub fn contains_entity(&self, entity: Entity) -> bool {
        self.reverse.contains_key(&entity)
    }
}
