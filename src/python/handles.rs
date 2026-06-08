use pyo3::prelude::*;
use bevy_ecs::prelude::Entity;

use crate::basic::ecs::elements::BusID;
use crate::basic::ecs::elements::generator::{TargetPMW, TargetQMVar, TargetVmPu};
use crate::basic::ecs::network::DataOps;
use super::grid::PowerGrid;

macro_rules! define_handle {
    ($name:ident) => {
        #[pyclass]
        pub struct $name {
            pub(crate) entity: u64,
            pub(crate) grid: Py<PowerGrid>,
        }

        impl Clone for $name {
            fn clone(&self) -> Self {
                Python::with_gil(|py| {
                    Self {
                        entity: self.entity,
                        grid: self.grid.clone_ref(py),
                    }
                })
            }
        }

        impl $name {
            pub fn new(entity: Entity, grid: Py<PowerGrid>) -> Self {
                Self { entity: entity.to_bits(), grid }
            }
            pub fn entity(&self) -> Entity {
                Entity::from_bits(self.entity)
            }
        }
    };
}

define_handle!(BusHandle);
define_handle!(LineHandle);
define_handle!(TrafoHandle);
define_handle!(LoadHandle);
define_handle!(GenHandle);
define_handle!(ExtGridHandle);
define_handle!(ShuntHandle);
define_handle!(SGenHandle);
define_handle!(SwitchHandle);

#[pymethods]
impl BusHandle { 
    fn __repr__(&self) -> String { format!("BusHandle({})", self.entity) } 
    
    /// Update all loads attached to this bus.
    ///
    /// p_mw: Total active power consumption (positive for consumption).
    /// q_mvar: Total reactive power consumption (positive for consumption).
    fn set_load(&self, py: Python<'_>, p_mw: f64, q_mvar: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let bus_id = {
            let world = grid_py.inner.world();
            world.get::<BusID>(self.entity()).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a bus entity"))?.0
        };
        
        let PowerGrid { inner, bus_to_elements, .. } = &mut *grid_py;
        if let Some(entities) = bus_to_elements.get(&bus_id) {
            let world = inner.world_mut();
            for &e in entities {
                if let Some(mut p) = world.get_mut::<TargetPMW>(e) { p.0 = -p_mw; }
                if let Some(mut q) = world.get_mut::<TargetQMVar>(e) { q.0 = -q_mvar; }
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No loads found at bus {}", bus_id)))
        }
    }

    /// Update all generators attached to this bus.
    ///
    /// p_mw: Total active power production.
    /// vm_pu: Voltage magnitude setpoint (p.u.).
    fn set_gen(&self, py: Python<'_>, p_mw: f64, vm_pu: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let bus_id = {
            let world = grid_py.inner.world();
            world.get::<BusID>(self.entity()).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a bus entity"))?.0
        };

        let PowerGrid { inner, bus_to_elements, .. } = &mut *grid_py;
        if let Some(entities) = bus_to_elements.get(&bus_id) {
            let world = inner.world_mut();
            for &e in entities {
                if let Some(mut p) = world.get_mut::<TargetPMW>(e) { p.0 = p_mw; }
                if let Some(mut vm) = world.get_mut::<TargetVmPu>(e) { vm.0 = vm_pu; }
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No generators found at bus {}", bus_id)))
        }
    }
}

#[pymethods]
impl LoadHandle { 
    fn __repr__(&self) -> String { format!("LoadHandle({})", self.entity) } 
    
    /// Set the active power consumption (MW).
    /// Positive value means the bus consumes power.
    fn set_p(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let world = grid_py.inner.world_mut();
        let entity = self.entity();
        
        if let Some(mut p) = world.get_mut::<TargetPMW>(entity) {
            p.0 = -value; Ok(()) 
        } else { 
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Entity {:?} missing TargetPMW component", entity)))
        }
    }
    
    /// Set the reactive power consumption (MVar).
    /// Positive value means the bus consumes reactive power.
    fn set_q(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let world = grid_py.inner.world_mut();
        if let Some(mut q) = world.get_mut::<TargetQMVar>(self.entity()) {
            q.0 = -value; Ok(())
        } else { Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity (missing TargetQMVar)")) }
    }
}

#[pymethods]
impl GenHandle { 
    fn __repr__(&self) -> String { format!("GenHandle({})", self.entity) } 
    
    /// Set the active power production (MW).
    fn set_p(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let world = grid_py.inner.world_mut();
        if let Some(mut p) = world.get_mut::<TargetPMW>(self.entity()) {
            p.0 = value; Ok(())
        } else { Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity")) }
    }
    
    /// Set the voltage magnitude setpoint (p.u.).
    fn set_vm(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        let world = grid_py.inner.world_mut();
        if let Some(mut vm) = world.get_mut::<TargetVmPu>(self.entity()) {
            vm.0 = value; Ok(())
        } else { Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity")) }
    }
}

#[pymethods]
impl LineHandle { fn __repr__(&self) -> String { format!("LineHandle({})", self.entity) } }
#[pymethods]
impl TrafoHandle { fn __repr__(&self) -> String { format!("TrafoHandle({})", self.entity) } }
#[pymethods]
impl ExtGridHandle { fn __repr__(&self) -> String { format!("ExtGridHandle({})", self.entity) } }
#[pymethods]
impl ShuntHandle { fn __repr__(&self) -> String { format!("ShuntHandle({})", self.entity) } }
#[pymethods]
impl SGenHandle { fn __repr__(&self) -> String { format!("SGenHandle({})", self.entity) } }
#[pymethods]
impl SwitchHandle { fn __repr__(&self) -> String { format!("SwitchHandle({})", self.entity) } }
