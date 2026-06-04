#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use numpy::IntoPyArray;
#[cfg(feature = "python")]
use crate::prelude::*;
#[cfg(feature = "python")]
use crate::basic::ecs::elements::{BusID, Line, LineParams, TransformerDevice, TapChanger};
#[cfg(feature = "python")]
use crate::basic::ecs::powerflow::prelude::{PowerFlowMat, PowerFlowResult};
#[cfg(feature = "python")]
use crate::basic::ecs::post_processing::{VBusResult, SBusResult};
#[cfg(feature = "python")]
use crate::io::pandapower::load_csv_zip;
#[cfg(feature = "python")]
use num_complex::ComplexFloat;
#[cfg(feature = "python")]
use pyo3::types::PyDictMethods;
#[cfg(feature = "python")]
use bevy_ecs::prelude::Entity;

#[cfg(feature = "python")]
use crate::basic::ecs::factory::GridFactory;

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

#[cfg(feature = "python")]
#[pyclass(unsendable)]
pub struct PowerGrid {
    inner: crate::prelude::PowerGrid,
    buffer: crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer,
}

#[cfg(feature = "python")]
#[pymethods]
impl PowerGrid {
    #[new]
    #[pyo3(signature = (case_path=None, _qlim=false, **kwargs))]
    fn new(case_path: Option<String>, _qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        let buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        
        if let Some(args) = kwargs {
            if let Some(branch_analysis) = args.get_item("branch_analysis")? {
                if branch_analysis.extract::<bool>()? {
                    inner.app_mut().add_plugins(crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin);
                }
            }
        }

        if let Some(path) = case_path {
            let net = load_csv_zip(&path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            inner.world_mut().insert_resource(PPNetwork(net));
        }

        Ok(Self { inner, buffer })
    }

    /// Set base frequency and base power.
    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        self.inner.set_base(f_hz, sn_mva);
    }

    /// Add a standard line type to the library.
    fn add_std_line_type(&mut self, name: String, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, max_i_ka: f64) {
        self.inner.add_std_line_type(name, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka);
    }

    /// Add a standard transformer type to the library.
    fn add_std_trafo_type(&mut self, name: String, sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64) {
        self.inner.add_std_trafo_type(name, sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent);
    }

    /// Add a bus to the grid.
    #[pyo3(signature = (id, vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(&mut self, id: i64, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> BusHandle {
        self.inner.add_bus(&mut self.buffer, id, vn_kv, name, vm_min, vm_max, zone).into()
    }

    /// Add a line to the grid.
    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(&mut self, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> LineHandle {
        let params = if std_type.is_none() {
            Some(LineParams {
                r_ohm_per_km,
                x_ohm_per_km,
                g_us_per_km,
                c_nf_per_km,
                length_km,
                df: 1.0,
                parallel,
                max_i_ka,
            })
        } else {
            None
        };
        self.inner.add_line(&mut self.buffer, from_bus, to_bus, length_km, std_type, params, name).into()
    }

    /// Add a load to the grid.
    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_load(&mut self, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> LoadHandle {
        self.inner.add_load(&mut self.buffer, bus, p_mw, q_mvar, name).into()
    }

    /// Add a generator to the grid.
    #[pyo3(signature = (bus, p_mw, vm_pu=1.0, p_min=-1000.0, p_max=1000.0, q_min=-1000.0, q_max=1000.0, name=None))]
    fn add_gen(&mut self, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> GenHandle {
        self.inner.add_gen(&mut self.buffer, bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name).into()
    }

    /// Add an external grid (slack) to the grid.
    #[pyo3(signature = (bus, vm_pu=1.0, va_degree=0.0, name=None))]
    fn add_ext_grid(&mut self, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> ExtGridHandle {
        self.inner.add_ext_grid(&mut self.buffer, bus, vm_pu, va_degree, name).into()
    }

    /// Add a transformer to the grid.
    #[pyo3(signature = (hv_bus, lv_bus, std_type=None, sn_mva=100.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=0.1, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, tap_pos=0.0, tap_neutral=0.0, tap_step_percent=1.25, parallel=1, name=None))]
    fn add_trafo(&mut self, hv_bus: i64, lv_bus: i64, std_type: Option<String>, sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64, shift_degree: f64, tap_pos: f64, tap_neutral: f64, tap_step_percent: f64, parallel: i32, name: Option<String>) -> TrafoHandle {
        let params = if std_type.is_none() {
            Some(TransformerDevice {
                df: 1.0,
                i0_percent,
                pfe_kw,
                vk_percent,
                vkr_percent,
                shift_degree,
                sn_mva,
                vn_hv_kv,
                vn_lv_kv,
                max_loading_percent: None,
                parallel,
                tap: Some(TapChanger {
                    side: Some("hv".to_string()),
                    neutral: Some(tap_neutral),
                    max: Some(tap_neutral + 10.0),
                    min: Some(tap_neutral - 10.0),
                    pos: Some(tap_pos),
                    step_degree: Some(0.0),
                    step_percent: Some(tap_step_percent),
                    is_phase_shifter: false,
                }),
            })
        } else {
            None
        };
        self.inner.add_trafo(&mut self.buffer, hv_bus, lv_bus, std_type, params, name).into()
    }

    /// Add a shunt to the grid.
    #[pyo3(signature = (bus, p_mw, q_mvar, vn_kv, step=1, name=None))]
    fn add_shunt(&mut self, bus: i64, p_mw: f64, q_mvar: f64, vn_kv: f64, step: i32, name: Option<String>) -> ShuntHandle {
        self.inner.add_shunt(&mut self.buffer, bus, p_mw, q_mvar, vn_kv, step, name).into()
    }

    /// Add a static generator (sgen) to the grid.
    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_sgen(&mut self, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> SGenHandle {
        self.inner.add_sgen(&mut self.buffer, bus, p_mw, q_mvar, name).into()
    }

    /// Add a switch to the grid.
    #[pyo3(signature = (bus, element, et, closed=true, name=None, z_ohm=0.0))]
    fn add_switch(&mut self, bus: i64, element: i64, et: String, closed: bool, name: Option<String>, z_ohm: f64) -> SwitchHandle {
        self.inner.add_switch(&mut self.buffer, bus, element, et, closed, name, z_ohm).into()
    }

    /// Initialize the power flow network (runs Bevy Startup systems).
    fn init_pf(&mut self) {
        // Apply programmatic grid construction commands
        self.buffer.apply(self.inner.world_mut());
        self.inner.init_pf_net();
    }

    /// Run a single power flow calculation.
    fn run_pf(&mut self) {
        self.inner.run_pf();
    }

    /// Run post-processing to extract results.
    fn post_process(&mut self) {
        self.inner.post_process();
    }

    /// Run both power flow and post-processing.
    fn solve(&mut self) {
        self.inner.run_pf();
        self.inner.post_process();
    }

    /// Returns the number of iterations for the last power flow run.
    #[getter]
    fn iterations(&self) -> PyResult<usize> {
        let results = self.inner.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        Ok(results.iterations)
    }

    /// Returns whether the last power flow run converged.
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        let results = self.inner.world().get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        Ok(results.converged)
    }

    /// Returns the number of buses in the grid.
    #[getter]
    fn n_bus(&self) -> usize {
        let world = self.inner.world();
        if let Some(id) = world.components().get_id(std::any::TypeId::of::<BusID>()) {
            world.archetypes().iter()
                .filter(|a| a.contains(id))
                .map(|a| a.len() as usize)
                .sum()
        } else {
            0
        }
    }

    /// Returns the number of lines in the grid.
    #[getter]
    fn n_line(&self) -> usize {
        let world = self.inner.world();
        if let Some(id) = world.components().get_id(std::any::TypeId::of::<Line>()) {
            world.archetypes().iter()
                .filter(|a| a.contains(id))
                .map(|a| a.len() as usize)
                .sum()
        } else {
            0
        }
    }

    /// Returns the complex voltage vector (reordered) as a Numpy array.
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.inner.world();
        let res = world.get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?;
        Ok(res.v.as_slice().to_vec().into_pyarray(py))
    }

    /// Returns the Y-bus matrix components as a dictionary of Numpy arrays.
    #[getter]
    fn y_bus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mat = world.get_resource::<PowerFlowMat>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow matrices not initialized. Did you call init_pf()?"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        let (offsets, indices, values) = mat.y_bus.csc_data();
        dict.set_item("indptr", offsets.to_vec().into_pyarray(py))?;
        dict.set_item("indices", indices.to_vec().into_pyarray(py))?;
        dict.set_item("data", values.to_vec().into_pyarray(py))?;
        dict.set_item("shape", (mat.y_bus.nrows(), mat.y_bus.ncols()))?;
        Ok(dict)
    }

    /// Returns the S-bus injection vector (reordered) as a Numpy array.
    #[getter]
    fn s_bus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.inner.world();
        let mat = world.get_resource::<PowerFlowMat>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow matrices not initialized"))?;
        Ok(mat.s_bus.as_slice().to_vec().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let n_bus = self.n_bus();
        let n_line = self.n_line();
        let converged = self.inner.world().get_resource::<PowerFlowResult>().map(|r| r.converged);
        format!("PowerGrid(buses={}, lines={}, converged={:?})", n_bus, n_line, converged)
    }

    /// Returns bus results (Vm, Va, P, Q) as a dictionary of Numpy arrays.
    fn get_bus_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        
        let mut bus_ids = Vec::new();
        let mut vms = Vec::new();
        let mut vas = Vec::new();
        let mut ps = Vec::new();
        let mut qs = Vec::new();

        let mut query = world.query::<(&BusID, &VBusResult, &SBusResult)>();
        for (id, v, s) in query.iter(world) {
            bus_ids.push(id.0 as i32);
            vms.push(v.0.norm());
            vas.push(v.0.arg().to_degrees());
            ps.push(s.0.re());
            qs.push(s.0.im());
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?;
        dict.set_item("vm_pu", vms.into_pyarray(py))?;
        dict.set_item("va_degree", vas.into_pyarray(py))?;
        dict.set_item("p_mw", ps.into_pyarray(py))?;
        dict.set_item("q_mvar", qs.into_pyarray(py))?;
        Ok(dict)
    }

    /// Returns line results as a dictionary of Numpy arrays.
    fn get_line_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        
        let mut p_f = Vec::new();
        let mut q_f = Vec::new();
        let mut p_t = Vec::new();
        let mut q_t = Vec::new();
        let mut loading = Vec::new();

        let mut query = world.query::<&crate::basic::ecs::post_processing::LineResultData>();
        for data in query.iter(world) {
            p_f.push(data.p_from_mw);
            q_f.push(data.q_from_mvar);
            p_t.push(data.p_to_mw);
            q_t.push(data.q_to_mvar);
            loading.push(data.loading_percent);
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("p_from_mw", p_f.into_pyarray(py))?;
        dict.set_item("q_from_mvar", q_f.into_pyarray(py))?;
        dict.set_item("p_to_mw", p_t.into_pyarray(py))?;
        dict.set_item("q_to_mvar", q_t.into_pyarray(py))?;
        dict.set_item("loading_percent", loading.into_pyarray(py))?;
        Ok(dict)
    }

    /// Update load at a specific bus.
    /// Note: This updates ALL loads at the bus.
    fn set_load(&mut self, bus_id: i64, p_mw: f64, q_mvar: f64) -> PyResult<()> {
        use crate::basic::ecs::elements::generator::{TargetPMW, TargetQMVar};
        use crate::basic::ecs::elements::TargetBus;

        let world = self.inner.world_mut();
        let mut query = world.query::<(Entity, &TargetBus, &mut TargetPMW, &mut TargetQMVar)>();
        let mut found = false;
        for (_, bus, mut p, mut q) in query.iter_mut(world) {
            if bus.0 == bus_id {
                p.0 = -p_mw; // Loads are negative in TargetPMW
                q.0 = -q_mvar;
                found = true;
            }
        }
        if !found {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No load found at bus {}", bus_id)));
        }
        Ok(())
    }

    /// Batch update loads at multiple buses.
    fn set_loads(&mut self, bus_ids: Vec<i64>, p_mws: Vec<f64>, q_mvars: Vec<f64>) -> PyResult<()> {
        use crate::basic::ecs::elements::generator::{TargetPMW, TargetQMVar};
        use crate::basic::ecs::elements::TargetBus;
        use std::collections::HashMap;

        if bus_ids.len() != p_mws.len() || bus_ids.len() != q_mvars.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Input arrays must have the same length"));
        }

        let map: HashMap<i64, (f64, f64)> = bus_ids.into_iter().zip(p_mws.into_iter().zip(q_mvars.into_iter()))
            .map(|(id, (p, q))| (id, (p, q)))
            .collect();

        let world = self.inner.world_mut();
        let mut query = world.query::<(&TargetBus, &mut TargetPMW, &mut TargetQMVar)>();
        for (bus, mut p, mut q) in query.iter_mut(world) {
            if let Some((new_p, new_q)) = map.get(&bus.0) {
                p.0 = -*new_p;
                q.0 = -*new_q;
            }
        }
        Ok(())
    }

    /// Update generator at a specific bus.
    fn set_gen(&mut self, bus_id: i64, p_mw: f64, vm_pu: f64) -> PyResult<()> {
        use crate::basic::ecs::elements::generator::{TargetPMW, TargetVmPu};
        use crate::basic::ecs::elements::TargetBus;

        let world = self.inner.world_mut();
        let mut query = world.query::<(Entity, &TargetBus, &mut TargetPMW, &mut TargetVmPu)>();
        let mut found = false;
        for (_, bus, mut p, mut vm) in query.iter_mut(world) {
            if bus.0 == bus_id {
                p.0 = p_mw;
                vm.0 = vm_pu;
                found = true;
            }
        }
        if !found {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("No generator found at bus {}", bus_id)));
        }
        Ok(())
    }

    /// Batch update generators at multiple buses.
    fn set_gens(&mut self, bus_ids: Vec<i64>, p_mws: Vec<f64>, vm_pus: Vec<f64>) -> PyResult<()> {
        use crate::basic::ecs::elements::generator::{TargetPMW, TargetVmPu};
        use crate::basic::ecs::elements::TargetBus;
        use std::collections::HashMap;

        if bus_ids.len() != p_mws.len() || bus_ids.len() != vm_pus.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Input arrays must have the same length"));
        }

        let map: HashMap<i64, (f64, f64)> = bus_ids.into_iter().zip(p_mws.into_iter().zip(vm_pus.into_iter()))
            .map(|(id, (p, v))| (id, (p, v)))
            .collect();

        let world = self.inner.world_mut();
        let mut query = world.query::<(&TargetBus, &mut TargetPMW, &mut TargetVmPu)>();
        for (bus, mut p, mut v) in query.iter_mut(world) {
            if let Some((new_p, new_v)) = map.get(&bus.0) {
                p.0 = *new_p;
                v.0 = *new_v;
            }
        }
        Ok(())
    }

    /// Get incidence matrix if BranchAnalysisPlugin was added.
    fn get_incidence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let res = world.get_resource::<crate::basic::ecs::powerflow::branch_data::BranchAnalysisRes>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("BranchAnalysisPlugin not added"))?;
        
        let dict = pyo3::types::PyDict::new(py);
        let (offsets, indices, values) = res.incidence.csr_data();
        dict.set_item("indptr", offsets.to_vec().into_pyarray(py))?;
        dict.set_item("indices", indices.to_vec().into_pyarray(py))?;
        dict.set_item("data", values.to_vec().into_pyarray(py))?;
        dict.set_item("shape", (res.incidence.nrows(), res.incidence.ncols()))?;
        Ok(dict)
    }

    /// Save simulation results (Vm, Va, P, Q) to a Parquet ZIP archive.
    #[cfg(feature = "archive")]
    fn save_results(&self, path: String) -> PyResult<()> {
        use bevy_archive::binary_archive::WorldArrowSnapshot;
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        use std::io::Write;
        
        let world = self.inner.world();
        let archive_res = world.get_resource::<ArchiveSnapshotRes>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Archive feature enabled but ArchivePlugin not added"))?;
        
        let output_reg = &archive_res.0.output_reg;
        let arrow_snap = WorldArrowSnapshot::from_world_reg(world, output_reg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create Arrow snapshot: {}", e)))?;
        
        let zip_data = arrow_snap.to_zip(None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to convert to zip: {}", e)))?;
            
        let mut f = std::fs::File::create(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e)))?;
        f.write_all(&zip_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write data: {}", e)))?;
            
        Ok(())
    }

    /// Get the case configuration as a ZIP archive (bytes) containing Parquet files.
    /// This includes network topology and electrical parameters.
    #[cfg(feature = "archive")]
    fn get_parquet_case(&self) -> PyResult<Vec<u8>> {
        use bevy_archive::binary_archive::WorldArrowSnapshot;
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        
        let world = self.inner.world();
        let archive_res = world.get_resource::<ArchiveSnapshotRes>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Archive feature enabled but ArchivePlugin not added"))?;
        
        let case_reg = &archive_res.0.case_file_reg;
        let arrow_snap = WorldArrowSnapshot::from_world_reg(world, case_reg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create Arrow snapshot: {}", e)))?;
        
        let zip_data = arrow_snap.to_zip(None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to convert to zip: {}", e)))?;
            
        Ok(zip_data)
    }

    /// Get simulation results as a ZIP archive (bytes) containing Parquet files.
    /// In Python, use io.BytesIO(data) and zipfile.ZipFile to read the contents.
    #[cfg(feature = "archive")]
    fn get_parquet_results(&self) -> PyResult<Vec<u8>> {
        use bevy_archive::binary_archive::WorldArrowSnapshot;
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        
        let world = self.inner.world();
        let archive_res = world.get_resource::<ArchiveSnapshotRes>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Archive feature enabled but ArchivePlugin not added"))?;
        
        let output_reg = &archive_res.0.output_reg;
        let arrow_snap = WorldArrowSnapshot::from_world_reg(world, output_reg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create Arrow snapshot: {}", e)))?;
        
        let zip_data = arrow_snap.to_zip(None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to convert to zip: {}", e)))?;
            
        Ok(zip_data)
    }
}

#[cfg(feature = "python")]
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(feature = "python")]
#[pyfunction]
fn features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "klu") { f.push("klu"); }
    if cfg!(feature = "faer") { f.push("faer"); }
    if cfg!(feature = "rsparse") { f.push("rsparse"); }
    if cfg!(feature = "archive") { f.push("archive"); }
    if cfg!(feature = "arrow") { f.push("arrow"); }
    if cfg!(feature = "python") { f.push("python"); }
    f
}

#[cfg(feature = "python")]
#[pymodule]
fn rustpower(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PowerGrid>()?;
    m.add_class::<BusHandle>()?;
    m.add_class::<LineHandle>()?;
    m.add_class::<TrafoHandle>()?;
    m.add_class::<LoadHandle>()?;
    m.add_class::<GenHandle>()?;
    m.add_class::<ExtGridHandle>()?;
    m.add_class::<ShuntHandle>()?;
    m.add_class::<SGenHandle>()?;
    m.add_class::<SwitchHandle>()?;
    
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    Ok(())
}
