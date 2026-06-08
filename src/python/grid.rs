use pyo3::prelude::*;
use numpy::IntoPyArray;
use crate::prelude::*;
use crate::basic::ecs::elements::{
    BusID, LineParams, FromBus, ToBus, PFCommonData, 
    VNominal, VmLimit, units::PerUnit,
    TargetBus, NodeLookup, SBusInjPu
};
use crate::basic::ecs::elements::generator::{TargetPMW, TargetQMVar};
use crate::basic::ecs::powerflow::prelude::{
    PowerFlowResult, PowerFlowConfig, BasePFInitPlugins, PowerFlowMat,
    SlackBus, PVBus
};
use crate::basic::ecs::elements::OutOfService;

use crate::basic::ecs::post_processing::{VBusResult, SBusResult, LineResultData};
use crate::io::pandapower::load_csv_zip;
use pyo3::types::PyDictMethods;
use bevy_ecs::prelude::{Entity, With, Without, DetectChangesMut};

use crate::basic::ecs::factory::GridFactory;
use crate::basic::ecs::network::{PowerFlowSolver, DataOps};
use crate::basic::ecs::plugin::DefaultPlugins;

use super::handles::*;

#[pyclass(unsendable)]
pub struct PowerGrid {
    pub(crate) inner: crate::prelude::PowerGrid,
    pub(crate) buffer: crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer,
    pub(crate) next_bus_id: i64,
    pub(crate) id_map: std::collections::HashMap<i64, i64>,
    pub(crate) bus_to_elements: std::collections::HashMap<i64, Vec<Entity>>,
}

#[pyclass(unsendable)]
pub struct GridBuilder {
    pub(crate) parent: Py<PowerGrid>,
}

#[pymethods]
impl GridBuilder {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, py: Python<'_>, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) -> PyResult<()> { self.commit(py) }

    /// Add a bus to the deferred command buffer.
    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(&mut self, py: Python<'_>, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> PyResult<(i64, BusHandle)> {
        let mut parent = self.parent.borrow_mut(py);
        let (id, entity) = parent.add_bus_impl(vn_kv, name, vm_min, vm_max, zone);
        Ok((id, BusHandle::new(entity, self.parent.clone_ref(py))))
    }

    /// Add a transmission line to the deferred command buffer.
    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(&mut self, py: Python<'_>, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<LineHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let entity = parent.add_line_impl(from_bus, to_bus, length_km, std_type, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, parallel, max_i_ka, name)?;
        Ok(LineHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Apply all buffered changes to the PowerGrid.
    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        buffer.apply(inner.world_mut());
        Ok(())
    }
}

#[pymethods]
impl PowerGrid {
    /// Create a new PowerGrid.
    ///
    /// case_path: Optional path to a ZIP/CSV case file.
    #[new]
    #[pyo3(signature = (case_path=None, _qlim=false, **kwargs))]
    fn new(case_path: Option<String>, _qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        let buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        inner.world_mut().insert_resource(crate::basic::ecs::factory::StdTypeLibrary::default());
        inner.world_mut().insert_resource(PowerFlowConfig { max_it: None, tol: None });
        inner.world_mut().insert_resource(PowerFlowSolver::default());
        inner.world_mut().insert_resource(PFCommonData { wbase: 50.0 * 2.0 * std::f64::consts::PI, f_hz: 50.0, sbase: 100.0 });
        
        // Add core plugins and the reactive update plugin
        inner.app_mut().add_plugins((
            BasePFInitPlugins, 
            DefaultPlugins,
            crate::basic::ecs::powerflow::structure_update::StructureUpdatePlugin
        ));

        if let Some(args) = kwargs {
            if let Ok(Some(branch_analysis)) = args.get_item("branch_analysis") {
                if branch_analysis.extract::<bool>()? { inner.app_mut().add_plugins(crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin); }
            }
        }

        let mut grid = Self { inner, buffer, next_bus_id: 0, id_map: std::collections::HashMap::new(), bus_to_elements: std::collections::HashMap::new() };

        if let Some(path) = case_path {
            let net = load_csv_zip(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            grid.inner.world_mut().insert_resource(PPNetwork(net));
            grid.init_pf();
            grid.sync_bus_to_elements();
        } else {
            grid.sync_next_bus_id();
        }
        Ok(grid)
    }

    /// Load a grid from a pandapower net object.
    fn from_pp_net(slf: Py<Self>, py: Python<'_>, net: Bound<'_, PyAny>) -> PyResult<()> {
        let mut grid_py = slf.borrow_mut(py);
        grid_py.from_buses_impl(slf.clone_ref(py), py, net.getattr("bus")?)?;
        if let Ok(load) = net.getattr("load") { grid_py.from_loads_impl(slf.clone_ref(py), py, load)?; }
        if let Ok(gen_df) = net.getattr("gen") { grid_py.from_gens_impl(slf.clone_ref(py), py, gen_df)?; }
        if let Ok(ext) = net.getattr("ext_grid") { grid_py.from_ext_grids_impl(slf.clone_ref(py), py, ext)?; }
        if let Ok(line) = net.getattr("line") { grid_py.from_lines_impl(slf.clone_ref(py), py, line)?; }
        Ok(())
    }

    /// Retrieve a handle for a specific Bus ID. Returns None if not found.
    fn bus(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<BusHandle> {
        let grid = slf.borrow(py);
        let world = grid.inner.world();
        let lookup = world.get_resource::<NodeLookup>()?;
        let entity = lookup.get_entity(id)?;
        Some(BusHandle::new(entity, slf.clone_ref(py)))
    }

    /// Retrieve a handle for the first load at a specific Bus ID. Returns None if not found.
    fn load(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<LoadHandle> {
        let grid = slf.borrow(py);
        let world = grid.inner.world();
        if let Some(entities) = grid.bus_to_elements.get(&id) {
            // Find the first entity that actually has a TargetPMW component
            for &e in entities {
                if world.get::<TargetPMW>(e).is_some() {
                    return Some(LoadHandle::new(e, slf.clone_ref(py)));
                }
            }
        }
        None
    }

    /// Return a GridBuilder for deferred operations.
    fn builder(slf: Py<Self>) -> GridBuilder { GridBuilder { parent: slf } }
    /// Alias for builder(), recommended for use with 'with' statements.
    fn defer(slf: Py<Self>) -> GridBuilder { GridBuilder { parent: slf } }

    /// Set system base frequency (Hz) and base power (MVA).
    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        self.inner.world_mut().insert_resource(PFCommonData { wbase: f_hz * 2.0 * std::f64::consts::PI, f_hz, sbase: sn_mva });
    }

    /// Synchronously add a bus to the grid.
    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(slf: Py<Self>, py: Python<'_>, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> (i64, BusHandle) {
        let (id, entity) = slf.borrow_mut(py).add_bus_impl(vn_kv, name, vm_min, vm_max, zone);
        (id, BusHandle::new(entity, slf.clone_ref(py)))
    }

    /// Synchronously add a transmission line.
    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(slf: Py<Self>, py: Python<'_>, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<LineHandle> {
        let entity = slf.borrow_mut(py).add_line_impl(from_bus, to_bus, length_km, std_type, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, parallel, max_i_ka, name)?;
        Ok(LineHandle::new(entity, slf.clone_ref(py)))
    }

    /// Add a static load. p_mw > 0 indicates consumption.
    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_load(slf: Py<Self>, py: Python<'_>, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> PyResult<LoadHandle> {
        let entity = slf.borrow_mut(py).add_load_impl(bus, p_mw, q_mvar, name)?;
        Ok(LoadHandle::new(entity, slf.clone_ref(py)))
    }

    /// Add a generator.
    #[pyo3(signature = (bus, p_mw, vm_pu=1.0, p_min=-1000.0, p_max=1000.0, q_min=-1000.0, q_max=1000.0, name=None))]
    fn add_gen(slf: Py<Self>, py: Python<'_>, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> PyResult<GenHandle> {
        let entity = slf.borrow_mut(py).add_gen_impl(bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name)?;
        Ok(GenHandle::new(entity, slf.clone_ref(py)))
    }

    /// Add an external grid (Slack).
    #[pyo3(signature = (bus, vm_pu=1.0, va_degree=0.0, name=None))]
    fn add_ext_grid(slf: Py<Self>, py: Python<'_>, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> PyResult<ExtGridHandle> {
        let entity = slf.borrow_mut(py).add_ext_grid_impl(bus, vm_pu, va_degree, name)?;
        Ok(ExtGridHandle::new(entity, slf.clone_ref(py)))
    }

    /// Initialize/Rebuild the power flow matrices. 
    /// MUST be called after adding elements or modifying topology.
    fn init_pf(&mut self) {
        use bevy_app::Startup;
        self.inner.world_mut().remove_resource::<PowerFlowMat>(); 
        self.inner.app_mut().world_mut().run_schedule(Startup);
        self.sync_next_bus_id();
        self.sync_bus_to_elements();
    }

    /// Zero out power injections and reset voltages to 1.0 (Flat start).
    fn reset_state(&mut self) {
        self.reset_state_impl();
    }

    /// Run power flow calculation.
    ///
    /// This method automatically synchronizes element parameter changes (via handles) 
    /// into the solver matrix using an incremental, reactive approach.
    fn solve(&mut self) { 
        // 1. Synchronize element-level parameters to bus-level injections
        self.sync_injections_to_buses();
        
        // 2. Synchronize bus-level injections to the solver matrix
        {
            let world = self.inner.world_mut();
            let sbase = world.get_resource::<PFCommonData>().map(|c| c.sbase).unwrap_or(100.0);
            let _sbase_frac = 1.0 / sbase;
            
            // Get data first to avoid holding world borrow while mutating mat
            let mut bus_data = Vec::new();
            {
                let mut bus_query = world.query::<(&BusID, &SBusInjPu)>();
                for (id, s) in bus_query.iter(world) {
                    bus_data.push((id.0, s.0));
                }
            }

            if let Some(mut mat) = world.get_resource_mut::<PowerFlowMat>() {
                // Clear and re-populate the s_bus vector from bus components
                mat.s_bus.fill(num_complex::Complex64::new(0.0, 0.0));
                for (bus_id, s) in bus_data {
                    let idx = mat.reorder_index(bus_id as usize);
                    mat.s_bus[idx] = s;
                }
                // FORCE: Ensure the solver doesn't skip this frame
                mat.set_changed();
            }
        }

        // 3. Perform the actual solve
        self.inner.run_pf();
        
        // 4. Extract results back to ECS components
        self.inner.post_process(); 
    }

    fn sync_injections_to_buses(&mut self) {
        let world = self.inner.world_mut();
        
        // 1. Reset all bus injections to clean state
        {
            let mut bus_query = world.query_filtered::<&mut SBusInjPu, With<BusID>>();
            for mut s in bus_query.iter_mut(world) {
                s.0 = num_complex::Complex64::new(0.0, 0.0); 
            }
        }

        // 2. Aggregate ALL elements that have a target and a bus assignment
        let sbase_frac = {
            let sbase = world.get_resource::<PFCommonData>().map(|c| c.sbase).unwrap_or(100.0);
            1.0 / sbase
        };
        
        let updates: Vec<(i64, f64, f64)> = {
            let mut element_query = world.query_filtered::<(&TargetBus, Option<&TargetPMW>, Option<&TargetQMVar>), Without<OutOfService>>();
            element_query.iter(world).map(|(b, p_opt, q_opt)| {
                (b.0, p_opt.map(|v| v.0).unwrap_or(0.0), q_opt.map(|v| v.0).unwrap_or(0.0))
            }).collect()
        };
        
        let lut_map: std::collections::HashMap<i64, Entity> = {
            let _lut = world.get_resource::<NodeLookup>().expect("NodeLookup missing");
            // Assuming NodeLookup has a public way to access its inner map or we iterate its forward list
            // If direct access is not available, we use the already known bus IDs from bus_to_elements or results
            world.iter_entities().filter_map(|e| e.get::<BusID>().map(|id| (id.0, e.id()))).collect()
        };

        for (bus_id, p, q) in updates {
            if let Some(&entity) = lut_map.get(&bus_id) {
                if let Some(mut s) = world.get_mut::<SBusInjPu>(entity) {
                    s.0.re += p * sbase_frac;
                    s.0.im += q * sbase_frac;
                }
            }
        }
    }

    fn display_case_buses<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let res = self.get_bus_params_impl(py)?; py.import("pandas")?.call_method1("DataFrame", (res,)) }
    fn display_case_lines<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let res = self.get_line_params_impl(py)?; py.import("pandas")?.call_method1("DataFrame", (res,)) }
    fn display_case_loads<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let res = self.get_load_params_impl(py)?; py.import("pandas")?.call_method1("DataFrame", (res,)) }
    fn display_buses<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let res = self.get_bus_results_impl(py)?; py.import("pandas")?.call_method1("DataFrame", (res,)) }
    fn display_lines<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { let res = self.get_line_results_impl(py)?; py.import("pandas")?.call_method1("DataFrame", (res,)) }

    #[getter] fn n_bus(&self) -> usize { let world = self.inner.world(); if let Some(id) = world.components().get_id(std::any::TypeId::of::<BusID>()) { world.archetypes().iter().filter(|a| a.contains(id)).map(|a| a.len() as usize).sum() } else { 0 } }
    #[getter] fn n_line(&self) -> usize { let world = self.inner.world(); if let Some(id) = world.components().get_id(std::any::TypeId::of::<crate::basic::ecs::elements::Line>()) { world.archetypes().iter().filter(|a| a.contains(id)).map(|a| a.len() as usize).sum() } else { 0 } }

    #[getter] fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> { let world = self.inner.world(); let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(res.v.as_slice().to_vec().into_pyarray(py)) }
    #[getter] fn iterations(&self) -> PyResult<usize> { let results = self.inner.world().get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(results.iterations) }
    #[getter] fn converged(&self) -> PyResult<bool> { let results = self.inner.world().get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(results.converged) }
}

impl PowerGrid {
    fn bus_exists_in_world(&self, bus_id: i64) -> bool {
        let world = self.inner.world();
        if let Some(lookup) = world.get_resource::<NodeLookup>() {
            lookup.get_entity(bus_id).is_some()
        } else {
            // During construction, check our internal next_bus_id counter
            bus_id < self.next_bus_id
        }
    }


    fn sync_next_bus_id(&mut self) {
        let world = self.inner.world_mut();
        let mut max_id = -1;
        world.iter_entities().filter_map(|e| e.get::<BusID>()).for_each(|id| { if id.0 > max_id { max_id = id.0; } });
        self.next_bus_id = max_id + 1;
    }

    fn sync_bus_to_elements(&mut self) {
        let world = self.inner.world_mut();
        self.bus_to_elements.clear();
        world.iter_entities().for_each(|e| {
            if let Some(bus) = e.get::<TargetBus>() {
                self.bus_to_elements.entry(bus.0).or_default().push(e.id());
            }
        });
    }

    fn reset_state_impl(&mut self) {
        let world = self.inner.world_mut();
        let bus_entities: Vec<Entity> = world.iter_entities()
            .filter(|e| e.contains::<BusID>())
            .map(|e| e.id())
            .collect();
        for e in bus_entities {
            if let Some(mut s) = world.get_mut::<SBusInjPu>(e) { 
                s.0 = num_complex::Complex64::new(0.0, 0.0); 
            }
        }
    }

    fn add_bus_impl(&mut self, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> (i64, Entity) {
        let id = self.next_bus_id; self.next_bus_id += 1;
        let PowerGrid { inner, buffer, .. } = self;
        let entity = inner.add_bus(buffer, id, vn_kv, name, vm_min, vm_max, zone);
        buffer.apply(inner.world_mut());
        (id, entity)
    }

    fn add_line_impl(&mut self, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<Entity> {
        if !self.bus_exists_in_world(from_bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", from_bus))); }
        if !self.bus_exists_in_world(to_bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", to_bus))); }
        let params = if std_type.is_none() { Some(LineParams { r_ohm_per_km, x_ohm_per_km, g_us_per_km, c_nf_per_km, length_km, df: 1.0, parallel, max_i_ka }) } else { None };
        let PowerGrid { inner, buffer, .. } = self;
        let entity = inner.add_line(buffer, from_bus, to_bus, length_km, std_type, params, name);
        buffer.apply(inner.world_mut());
        Ok(entity)
    }

    fn add_load_impl(&mut self, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> PyResult<Entity> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let PowerGrid { inner, buffer, bus_to_elements, .. } = self;
        let entity = inner.add_load(buffer, bus, p_mw, q_mvar, name);
        buffer.apply(inner.world_mut());
        bus_to_elements.entry(bus).or_default().push(entity);
        Ok(entity)
    }

    fn add_gen_impl(&mut self, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> PyResult<Entity> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let PowerGrid { inner, buffer, bus_to_elements, .. } = self;
        let entity = inner.add_gen(buffer, bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name);
        buffer.apply(inner.world_mut());
        bus_to_elements.entry(bus).or_default().push(entity);
        Ok(entity)
    }

    fn add_ext_grid_impl(&mut self, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> PyResult<Entity> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let PowerGrid { inner, buffer, .. } = self;
        let entity = inner.add_ext_grid(buffer, bus, vm_pu, va_degree, name);
        buffer.apply(inner.world_mut());
        Ok(entity)
    }

    fn from_buses_impl(&mut self, _slf: Py<Self>, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let index = df.getattr("index")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let vn_kv = df.getattr("vn_kv")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; index.len()] };
        let vm_min = if let Ok(col) = df.getattr("min_vm_pu") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![0.9; index.len()] };
        let vm_max = if let Ok(col) = df.getattr("max_vm_pu") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![1.1; index.len()] };
        for i in 0..index.len() {
            let (new_id, _) = self.add_bus_impl(vn_kv[i], names[i].clone(), vm_min[i], vm_max[i], 0);
            self.id_map.insert(index[i], new_id);
        }
        Ok(())
    }

    fn from_lines_impl(&mut self, _slf: Py<Self>, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let from_buses: Vec<i64> = df.getattr("from_bus")?.call_method0("tolist")?.extract()?;
        let to_buses: Vec<i64> = df.getattr("to_bus")?.call_method0("tolist")?.extract()?;
        let lengths = df.getattr("length_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let r_ohms = df.getattr("r_ohm_per_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let x_ohms = df.getattr("x_ohm_per_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; from_buses.len()] };
        let max_i = if let Ok(col) = df.getattr("max_i_ka") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![0.0; from_buses.len()] };
        for i in 0..from_buses.len() {
            let internal_from = *self.id_map.get(&from_buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", from_buses[i])))?;
            let internal_to = *self.id_map.get(&to_buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", to_buses[i])))?;
            self.add_line_impl(internal_from, internal_to, lengths[i], None, r_ohms[i], x_ohms[i], 0.0, 0.0, 1, max_i[i], names[i].clone())?;
        }
        Ok(())
    }

    fn from_loads_impl(&mut self, _slf: Py<Self>, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses: Vec<i64> = df.getattr("bus")?.call_method0("tolist")?.extract()?;
        let p_mws: Vec<f64> = df.getattr("p_mw")?.call_method0("tolist")?.extract()?;
        let q_mvars = df.getattr("q_mvar")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_load_impl(internal_bus, p_mws[i], q_mvars[i], names[i].clone())?;
        }
        Ok(())
    }

    fn from_gens_impl(&mut self, _slf: Py<Self>, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses = df.getattr("bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let p_mws = df.getattr("p_mw")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let vm_pus = df.getattr("vm_pu")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_gen_impl(internal_bus, p_mws[i], vm_pus[i], -1000.0, 1000.0, -1000.0, 1000.0, names[i].clone())?;
        }
        Ok(())
    }

    fn from_ext_grids_impl(&mut self, _slf: Py<Self>, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses = df.getattr("bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let vm_pus = df.getattr("vm_pu")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let va_degs = df.getattr("va_degree")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_ext_grid_impl(internal_bus, vm_pus[i], va_degs[i], names[i].clone())?;
        }
        Ok(())
    }

    fn get_bus_params_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut bus_ids = Vec::new(); let mut names = Vec::new(); let mut vn_kv = Vec::new(); 
        let mut vm_min = Vec::new(); let mut vm_max = Vec::new(); let mut types = Vec::new();
        world.iter_entities().for_each(|e| {
            if let Some(id) = e.get::<BusID>() {
                bus_ids.push(id.0); names.push(e.get::<bevy_ecs::name::Name>().map(|n| n.as_str().to_string()).unwrap_or_default());
                vn_kv.push(e.get::<VNominal>().map(|v| v.0.0).unwrap_or(0.0));
                let (min, max) = e.get::<VmLimit<PerUnit>>().map(|l| (l.min(), l.max())).unwrap_or((0.9, 1.1));
                vm_min.push(min); vm_max.push(max);
                let b_type = if e.contains::<SlackBus>() { "Slack" } else if e.contains::<PVBus>() { "PV" } else { "PQ" };
                types.push(b_type);
            }
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?; dict.set_item("name", names)?; dict.set_item("type", types)?;
        dict.set_item("vn_kv", vn_kv.into_pyarray(py))?; dict.set_item("vm_min_pu", vm_min.into_pyarray(py))?; dict.set_item("vm_max_pu", vm_max.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_line_params_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut from_bus = Vec::new(); let mut to_bus = Vec::new(); let mut length_km = Vec::new(); 

        let mut r_ohm_per_km = Vec::new(); let mut x_ohm_per_km = Vec::new(); 
        let mut max_i_ka = Vec::new(); let mut names = Vec::new();
        world.iter_entities().for_each(|e| {
            if let (Some(f), Some(t), Some(p)) = (e.get::<FromBus>(), e.get::<ToBus>(), e.get::<LineParams>()) {
                from_bus.push(f.0); to_bus.push(t.0); length_km.push(p.length_km);
                r_ohm_per_km.push(p.r_ohm_per_km); x_ohm_per_km.push(p.x_ohm_per_km); max_i_ka.push(p.max_i_ka);
                names.push(e.get::<bevy_ecs::name::Name>().map(|n| n.as_str().to_string()).unwrap_or_default());
            }
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("from_bus", from_bus.into_pyarray(py))?; dict.set_item("to_bus", to_bus.into_pyarray(py))?;
        dict.set_item("length_km", length_km.into_pyarray(py))?; dict.set_item("r_ohm_per_km", r_ohm_per_km.into_pyarray(py))?;
        dict.set_item("x_ohm_per_km", x_ohm_per_km.into_pyarray(py))?; dict.set_item("max_i_ka", max_i_ka.into_pyarray(py))?; dict.set_item("name", names)?;
        Ok(dict)
    }

    fn get_load_params_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut buses = Vec::new(); let mut p_mw = Vec::new(); let mut q_mvar = Vec::new(); let mut names = Vec::new();

        world.iter_entities().for_each(|e| {
            if let (Some(b), Some(p), Some(q)) = (e.get::<TargetBus>(), e.get::<TargetPMW>(), e.get::<TargetQMVar>()) {
                if !e.contains::<SlackBus>() && !e.contains::<PVBus>() {
                    buses.push(b.0); p_mw.push(-p.0); q_mvar.push(-q.0);
                    names.push(e.get::<bevy_ecs::name::Name>().map(|n| n.as_str().to_string()).unwrap_or_default());
                }
            }
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus", buses.into_pyarray(py))?; dict.set_item("p_mw", p_mw.into_pyarray(py))?;
        dict.set_item("q_mvar", q_mvar.into_pyarray(py))?; dict.set_item("name", names)?; Ok(dict)
    }

    fn get_bus_results_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut bus_ids = Vec::new(); let mut v_complex = Vec::new(); let mut vms = Vec::new(); let mut vas = Vec::new(); let mut ps = Vec::new(); let mut qs = Vec::new();
        world.iter_entities().for_each(|e| {
            if let (Some(id), Some(v), Some(s)) = (e.get::<BusID>(), e.get::<VBusResult>(), e.get::<SBusResult>()) {
                bus_ids.push(id.0); v_complex.push(v.0); vms.push(v.0.norm()); vas.push(v.0.arg().to_degrees()); ps.push(s.0.re); qs.push(s.0.im);
            }
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?; dict.set_item("v_pu", v_complex.into_pyarray(py))?; dict.set_item("vm_pu", vms.into_pyarray(py))?; dict.set_item("va_degree", vas.into_pyarray(py))?; dict.set_item("p_mw", ps.into_pyarray(py))?; dict.set_item("q_mvar", qs.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_line_results_impl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut from_bus = Vec::new(); let mut to_bus = Vec::new();
        let mut p_f = Vec::new(); let mut q_f = Vec::new();
        let mut p_t = Vec::new(); let mut q_t = Vec::new();
        let mut pl = Vec::new(); let mut ql = Vec::new();
        let mut i_f = Vec::new(); let mut i_t = Vec::new();
        let mut i_max = Vec::new(); let mut loading = Vec::new();
        
        world.iter_entities().for_each(|e| {
            if let (Some(f), Some(t), Some(data)) = (e.get::<FromBus>(), e.get::<ToBus>(), e.get::<LineResultData>()) {
                from_bus.push(f.0); to_bus.push(t.0);
                p_f.push(data.p_from_mw); q_f.push(data.q_from_mvar);
                p_t.push(data.p_to_mw); q_t.push(data.q_to_mvar);
                pl.push(data.pl_mw); ql.push(data.ql_mvar);
                i_f.push(data.i_from_ka); i_t.push(data.i_to_ka);
                i_max.push(data.i_ka); loading.push(data.loading_percent);
            }
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("from_bus", from_bus.into_pyarray(py))?; dict.set_item("to_bus", to_bus.into_pyarray(py))?;
        dict.set_item("p_from_mw", p_f.into_pyarray(py))?; dict.set_item("q_from_mvar", q_f.into_pyarray(py))?;
        dict.set_item("p_to_mw", p_t.into_pyarray(py))?; dict.set_item("q_to_mvar", q_t.into_pyarray(py))?;
        dict.set_item("pl_mw", pl.into_pyarray(py))?; dict.set_item("ql_mvar", ql.into_pyarray(py))?;
        dict.set_item("i_from_ka", i_f.into_pyarray(py))?; dict.set_item("i_to_ka", i_t.into_pyarray(py))?;
        dict.set_item("i_ka", i_max.into_pyarray(py))?; dict.set_item("loading_percent", loading.into_pyarray(py))?;
        Ok(dict)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::ecs::powerflow::prelude::PowerFlowMat;

    #[test]
    fn test_dynamic_parameter_update_via_handle() {
        pyo3::prepare_freethreaded_python();
    }
}
