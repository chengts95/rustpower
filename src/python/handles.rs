#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use bevy_ecs::prelude::Entity;

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct SwitchHandle(u64);

#[cfg(feature = "python")]
impl From<Entity> for SwitchHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl SwitchHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl SwitchHandle { fn __repr__(&self) -> String { format!("SwitchHandle({})", self.0) } }

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct BusHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct LineHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct TrafoHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct LoadHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct GenHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct ExtGridHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct ShuntHandle(u64);

#[cfg(feature = "python")]
#[pyclass]
#[derive(Clone, Copy)]
pub struct SGenHandle(u64);

#[cfg(feature = "python")]
impl From<Entity> for BusHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for LineHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for TrafoHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for LoadHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for GenHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for ExtGridHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for ShuntHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }
#[cfg(feature = "python")]
impl From<Entity> for SGenHandle { fn from(e: Entity) -> Self { Self(e.to_bits()) } }

#[cfg(feature = "python")]
impl BusHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl LineHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl TrafoHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl LoadHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl GenHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl ExtGridHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl ShuntHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }
#[cfg(feature = "python")]
impl SGenHandle { pub fn entity(&self) -> Entity { Entity::from_bits(self.0) } }

#[cfg(feature = "python")]
#[pymethods]
impl BusHandle { fn __repr__(&self) -> String { format!("BusHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl LineHandle { fn __repr__(&self) -> String { format!("LineHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl TrafoHandle { fn __repr__(&self) -> String { format!("TrafoHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl LoadHandle { fn __repr__(&self) -> String { format!("LoadHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl GenHandle { fn __repr__(&self) -> String { format!("GenHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl ExtGridHandle { fn __repr__(&self) -> String { format!("ExtGridHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl ShuntHandle { fn __repr__(&self) -> String { format!("ShuntHandle({})", self.0) } }
#[cfg(feature = "python")]
#[pymethods]
impl SGenHandle { fn __repr__(&self) -> String { format!("SGenHandle({})", self.0) } }
