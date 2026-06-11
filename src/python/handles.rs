use pyo3::prelude::*;
use bevy_ecs::prelude::Entity;

use crate::basic::ecs::elements::*;
use crate::basic::ecs::elements::generator::{GeneratorCfg, Slack, TargetPMW, TargetQMVar, TargetVmPu};
use crate::basic::ecs::network::DataOps;
use crate::basic::ecs::post_processing::{LineResultData, SBusResult, VBusResult};
// All parameter changes route through the standardized mutation pipeline.
// Handles are pure views: no propagation logic, no sign conventions.
use crate::basic::ecs::powerflow::mutation;
use super::grid::PowerGrid;

/// Toggle the OutOfService marker on an element entity and post a
/// FullRebuildEvent: in_service changes are Topology-class, consumed by
/// structure_update at the next solve.
fn set_entity_in_service(grid_py: &mut PowerGrid, entity: Entity, on: bool) {
    let world = grid_py.inner.world_mut();
    if on {
        world.entity_mut(entity).remove::<OutOfService>();
    } else {
        world.entity_mut(entity).insert(OutOfService);
    }
    let _ = world.write_message(crate::basic::ecs::powerflow::structure_update::FullRebuildEvent);
}

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

    /// Bus ID (from the case data).
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<i64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<BusID>(self.entity()).map(|b| b.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a bus entity"))
    }

    /// Nominal voltage in kV.
    #[getter]
    fn vn_kv(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<VNominal>(self.entity()).map(|v| v.0.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a bus entity"))
    }

    /// Voltage magnitude (p.u.) from the last solve.
    #[getter]
    fn vm_pu(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<VBusResult>(self.entity()).map(|v| v.0.norm())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result at this bus: call solve() first"))
    }

    /// Voltage angle (degrees) from the last solve.
    #[getter]
    fn va_degree(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<VBusResult>(self.entity()).map(|v| v.0.arg().to_degrees())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result at this bus: call solve() first"))
    }

    /// Net injected active power (MW) from the last solve. Positive for production.
    #[getter]
    fn p_mw(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<SBusResult>(self.entity()).map(|s| s.0.re)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result at this bus: call solve() first"))
    }

    /// Net injected reactive power (MVar) from the last solve.
    #[getter]
    fn q_mvar(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<SBusResult>(self.entity()).map(|s| s.0.im)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result at this bus: call solve() first"))
    }

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
        let entities = bus_to_elements.get(&bus_id).cloned().unwrap_or_default();
        let world = inner.world_mut();
        let mut found = false;
        for e in entities {
            // Only touch load entities; gens/sgens at the same bus also carry TargetPMW
            if world.get::<LoadCfg>(e).is_none() { continue; }
            found = true;
            mutation::set_load_p(world, e, p_mw);
            mutation::set_load_q(world, e, q_mvar);
        }
        if !found {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No loads found at bus {}", bus_id)));
        }
        Ok(())
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
        let entities = bus_to_elements.get(&bus_id).cloned().unwrap_or_default();
        let world = inner.world_mut();
        let mut found = false;
        for e in entities {
            // Only touch PV generators; skip loads and the slack machine
            if world.get::<GeneratorCfg>(e).is_none() || world.get::<Slack>(e).is_some() { continue; }
            found = true;
            mutation::set_gen_p(world, e, p_mw);
            mutation::set_gen_vm(world, e, vm_pu);
        }
        if !found {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No generators found at bus {}", bus_id)));
        }
        Ok(())
    }
}

#[pymethods]
impl LoadHandle {
    fn __repr__(&self) -> String { format!("LoadHandle({})", self.entity) }

    #[getter]
    fn bus(&self, py: Python<'_>) -> PyResult<i64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetBus>(self.entity()).map(|b| b.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity"))
    }

    /// Active power consumption (MW). Assignment = immediate-mode command.
    #[getter]
    fn p_mw(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetPMW>(self.entity()).map(|p| -p.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity"))
    }

    #[setter]
    fn set_p_mw(&self, py: Python<'_>, value: f64) -> PyResult<()> { self.set_p(py, value) }

    /// Reactive power consumption (MVar). Assignment = immediate-mode command.
    #[getter]
    fn q_mvar(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetQMVar>(self.entity()).map(|q| -q.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity"))
    }

    #[setter]
    fn set_q_mvar(&self, py: Python<'_>, value: f64) -> PyResult<()> { self.set_q(py, value) }

    #[getter]
    fn in_service(&self, py: Python<'_>) -> bool {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<OutOfService>(self.entity()).is_none()
    }

    /// Topology-class change: triggers a full rebuild at the next solve.
    #[setter]
    fn set_in_service(&self, py: Python<'_>, on: bool) {
        let mut grid_py = self.grid.borrow_mut(py);
        set_entity_in_service(&mut grid_py, self.entity(), on);
    }

    /// Set the active power consumption (MW).
    /// Positive value means the bus consumes power.
    fn set_p(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        if mutation::set_load_p(grid_py.inner.world_mut(), self.entity(), value) {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity (missing TargetPMW)"))
        }
    }

    /// Set the reactive power consumption (MVar).
    /// Positive value means the bus consumes reactive power.
    fn set_q(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        if mutation::set_load_q(grid_py.inner.world_mut(), self.entity(), value) {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Load entity (missing TargetQMVar)"))
        }
    }
}

#[pymethods]
impl GenHandle {
    fn __repr__(&self) -> String { format!("GenHandle({})", self.entity) }

    #[getter]
    fn bus(&self, py: Python<'_>) -> PyResult<i64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetBus>(self.entity()).map(|b| b.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity"))
    }

    /// Active power production (MW). Assignment = immediate-mode command.
    #[getter]
    fn p_mw(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetPMW>(self.entity()).map(|p| p.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity"))
    }

    #[setter]
    fn set_p_mw(&self, py: Python<'_>, value: f64) -> PyResult<()> { self.set_p(py, value) }

    /// Voltage magnitude setpoint (p.u.). Assignment = immediate-mode command.
    #[getter]
    fn vm_pu(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<TargetVmPu>(self.entity()).map(|v| v.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity"))
    }

    #[setter]
    fn set_vm_pu(&self, py: Python<'_>, value: f64) -> PyResult<()> { self.set_vm(py, value) }

    #[getter]
    fn in_service(&self, py: Python<'_>) -> bool {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<OutOfService>(self.entity()).is_none()
    }

    /// Topology-class change: triggers a full rebuild at the next solve.
    #[setter]
    fn set_in_service(&self, py: Python<'_>, on: bool) {
        let mut grid_py = self.grid.borrow_mut(py);
        set_entity_in_service(&mut grid_py, self.entity(), on);
    }

    /// Set the active power production (MW).
    fn set_p(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        if mutation::set_gen_p(grid_py.inner.world_mut(), self.entity(), value) {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity"))
        }
    }

    /// Set the voltage magnitude setpoint (p.u.).
    fn set_vm(&self, py: Python<'_>, value: f64) -> PyResult<()> {
        let mut grid_py = self.grid.borrow_mut(py);
        if mutation::set_gen_vm(grid_py.inner.world_mut(), self.entity(), value) {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Generator entity"))
        }
    }
}

#[pymethods]
impl LineHandle {
    fn __repr__(&self) -> String { format!("LineHandle({})", self.entity) }

    #[getter]
    fn from_bus(&self, py: Python<'_>) -> PyResult<i64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<FromBus>(self.entity()).map(|b| b.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Line entity"))
    }

    #[getter]
    fn to_bus(&self, py: Python<'_>) -> PyResult<i64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<ToBus>(self.entity()).map(|b| b.0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Not a Line entity"))
    }

    #[getter]
    fn in_service(&self, py: Python<'_>) -> bool {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<OutOfService>(self.entity()).is_none()
    }

    /// Topology-class change: triggers a full rebuild at the next solve.
    #[setter]
    fn set_in_service(&self, py: Python<'_>, on: bool) {
        let mut grid_py = self.grid.borrow_mut(py);
        set_entity_in_service(&mut grid_py, self.entity(), on);
    }

    /// Active power flow at the from-side (MW), from the last solve.
    #[getter]
    fn p_from_mw(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<LineResultData>(self.entity()).map(|d| d.p_from_mw)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result on this line: call solve() first"))
    }

    /// Reactive power flow at the from-side (MVar), from the last solve.
    #[getter]
    fn q_from_mvar(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<LineResultData>(self.entity()).map(|d| d.q_from_mvar)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result on this line: call solve() first"))
    }

    /// Current magnitude at the from-side (kA), from the last solve.
    #[getter]
    fn i_ka(&self, py: Python<'_>) -> PyResult<f64> {
        let grid_py = self.grid.borrow(py);
        grid_py.inner.world().get::<LineResultData>(self.entity()).map(|d| d.i_from_ka)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No result on this line: call solve() first"))
    }
}
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
