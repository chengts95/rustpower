#![cfg(feature = "python")]

use pyo3::prelude::*;
use numpy::IntoPyArray;
use crate::prelude::*;
use crate::basic::ecs::elements::{
    BusID, LineParams, FromBus, ToBus, PFCommonData, 
    VNominal, VmLimit, units::PerUnit,
    TargetBus,
    generator::{TargetPMW, TargetQMVar}
};
use crate::basic::ecs::powerflow::init::{PQBus, PVBus, SlackBus};
use crate::basic::ecs::powerflow::prelude::{PowerFlowResult, PowerFlowConfig, BasePFInitPlugins};
use crate::basic::ecs::post_processing::{VBusResult, SBusResult};
use crate::io::pandapower::load_csv_zip;
use num_complex::ComplexFloat;
use pyo3::types::PyDictMethods;

use crate::basic::ecs::factory::{GridFactory, StdTypeLibrary};
use crate::basic::ecs::network::PowerFlowSolver;
use crate::basic::ecs::plugin::DefaultPlugins;

use super::handles::*;

#[pyclass(unsendable)]
pub struct PowerGrid {
    pub(crate) inner: crate::prelude::PowerGrid,
    pub(crate) buffer: crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer,
    pub(crate) next_bus_id: i64,
    pub(crate) id_map: std::collections::HashMap<i64, i64>,
}

#[pyclass(unsendable)]
pub struct GridBuilder {
    pub(crate) parent: Py<PowerGrid>,
}

#[pymethods]
impl GridBuilder {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __exit__(&mut self, py: Python<'_>, _exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) -> PyResult<()> { self.commit(py) }

    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(&mut self, py: Python<'_>, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> PyResult<(i64, BusHandle)> {
        let mut parent = self.parent.borrow_mut(py);
        let id = parent.next_bus_id;
        parent.next_bus_id += 1;
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let handle = inner.add_bus(buffer, id, vn_kv, name, vm_min, vm_max, zone).into();
        Ok((id, handle))
    }

    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(&mut self, py: Python<'_>, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<LineHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let params = if std_type.is_none() { Some(LineParams { r_ohm_per_km, x_ohm_per_km, g_us_per_km, c_nf_per_km, length_km, df: 1.0, parallel, max_i_ka }) } else { None };
        let handle = inner.add_line(buffer, from_bus, to_bus, length_km, std_type, params, name).into();
        Ok(handle)
    }

    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        buffer.apply(inner.world_mut());
        Ok(())
    }
}

#[pymethods]
impl PowerGrid {
    #[new]
    #[pyo3(signature = (case_path=None, _qlim=false, **kwargs))]
    fn new(case_path: Option<String>, _qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        let buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        inner.world_mut().insert_resource(StdTypeLibrary::default());
        inner.world_mut().insert_resource(PowerFlowConfig { max_it: None, tol: None });
        inner.world_mut().insert_resource(PowerFlowSolver::default());
        inner.world_mut().insert_resource(PFCommonData { wbase: 50.0 * 2.0 * std::f64::consts::PI, f_hz: 50.0, sbase: 100.0 });
        inner.app_mut().add_plugins((BasePFInitPlugins, DefaultPlugins));

        if let Some(args) = kwargs {
            if let Some(branch_analysis) = args.get_item("branch_analysis")? {
                if branch_analysis.extract::<bool>()? { inner.app_mut().add_plugins(crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin); }
            }
        }

        if let Some(path) = case_path {
            let net = load_csv_zip(&path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            inner.world_mut().insert_resource(PPNetwork(net));
            let mut grid = Self { inner, buffer, next_bus_id: 0, id_map: std::collections::HashMap::new() };
            grid.init_pf();
            return Ok(grid);
        }
        let mut grid = Self { inner, buffer, next_bus_id: 0, id_map: std::collections::HashMap::new() };
        grid.sync_next_bus_id();
        Ok(grid)
    }

    fn from_pp_net(&mut self, py: Python<'_>, net: Bound<'_, PyAny>) -> PyResult<()> {
        self.from_buses(py, net.getattr("bus")?)?;
        if let Ok(load) = net.getattr("load") { self.from_loads(py, load)?; }
        if let Ok(gen_df) = net.getattr("gen") { self.from_gens(py, gen_df)?; }
        if let Ok(ext) = net.getattr("ext_grid") { self.from_ext_grids(py, ext)?; }
        if let Ok(line) = net.getattr("line") { self.from_lines(py, line)?; }
        Ok(())
    }

    fn from_buses(&mut self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let index = df.getattr("index")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let vn_kv = df.getattr("vn_kv")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; index.len()] };
        let vm_min = if let Ok(col) = df.getattr("min_vm_pu") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![0.9; index.len()] };
        let vm_max = if let Ok(col) = df.getattr("max_vm_pu") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![1.1; index.len()] };
        for i in 0..index.len() {
            let (new_id, _) = self.add_bus(vn_kv[i], names[i].clone(), vm_min[i], vm_max[i], 0);
            self.id_map.insert(index[i], new_id);
        }
        Ok(())
    }

    fn from_lines(&mut self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let from_buses = df.getattr("from_bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let to_buses = df.getattr("to_bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let lengths = df.getattr("length_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let r_ohms = df.getattr("r_ohm_per_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let x_ohms = df.getattr("x_ohm_per_km")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; from_buses.len()] };
        let max_i = if let Ok(col) = df.getattr("max_i_ka") { col.call_method0("tolist")?.extract::<Vec<f64>>()? } else { vec![0.0; from_buses.len()] };
        for i in 0..from_buses.len() {
            let internal_from = *self.id_map.get(&from_buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", from_buses[i])))?;
            let internal_to = *self.id_map.get(&to_buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", to_buses[i])))?;
            self.add_line(internal_from, internal_to, lengths[i], None, r_ohms[i], x_ohms[i], 0.0, 0.0, 1, max_i[i], names[i].clone())?;
        }
        Ok(())
    }

    fn from_loads(&mut self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses = df.getattr("bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let p_mws = df.getattr("p_mw")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let q_mvars = df.getattr("q_mvar")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_load(internal_bus, p_mws[i], q_mvars[i], names[i].clone())?;
        }
        Ok(())
    }

    fn from_gens(&mut self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses = df.getattr("bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let p_mws = df.getattr("p_mw")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let vm_pus = df.getattr("vm_pu")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_gen(internal_bus, p_mws[i], vm_pus[i], -1000.0, 1000.0, -1000.0, 1000.0, names[i].clone())?;
        }
        Ok(())
    }

    fn from_ext_grids(&mut self, _py: Python<'_>, df: Bound<'_, PyAny>) -> PyResult<()> {
        let buses = df.getattr("bus")?.call_method0("tolist")?.extract::<Vec<i64>>()?;
        let vm_pus = df.getattr("vm_pu")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let va_degs = df.getattr("va_degree")?.call_method0("tolist")?.extract::<Vec<f64>>()?;
        let names = if let Ok(col) = df.getattr("name") { col.call_method0("tolist")?.extract::<Vec<Option<String>>>()? } else { vec![None; buses.len()] };
        for i in 0..buses.len() {
            let internal_bus = *self.id_map.get(&buses[i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Bus ID {} not found.", buses[i])))?;
            self.add_ext_grid(internal_bus, vm_pus[i], va_degs[i], names[i].clone())?;
        }
        Ok(())
    }

    fn internal_id(&self, external_id: i64) -> Option<i64> { self.id_map.get(&external_id).copied() }

    fn sync_next_bus_id(&mut self) {
        let world = self.inner.world();
        let mut max_id = -1;
        world.iter_entities().filter_map(|e| e.get::<BusID>()).for_each(|id| { if id.0 > max_id { max_id = id.0; } });
        self.next_bus_id = max_id + 1;
    }

    fn builder(slf: Py<Self>) -> GridBuilder { GridBuilder { parent: slf } }
    fn defer(slf: Py<Self>) -> GridBuilder { Self::builder(slf) }

    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        self.inner.world_mut().insert_resource(PFCommonData { wbase: f_hz * 2.0 * std::f64::consts::PI, f_hz, sbase: sn_mva });
    }

    fn add_std_line_type(&mut self, name: String, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, max_i_ka: f64) {
        self.inner.add_std_line_type(name, r_ohm_per_km, x_ohm_per_km, c_nf_per_km, g_us_per_km, max_i_ka);
    }

    fn add_std_trafo_type(&mut self, name: String, sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64) {
        self.inner.add_std_trafo_type(name, sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent);
    }

    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(&mut self, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> (i64, BusHandle) {
        let id = self.next_bus_id; self.next_bus_id += 1;
        let handle = self.inner.add_bus(&mut self.buffer, id, vn_kv, name, vm_min, vm_max, zone).into();
        self.buffer.apply(self.inner.world_mut()); (id, handle)
    }

    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(&mut self, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<LineHandle> {
        if !self.bus_exists_in_world(from_bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", from_bus))); }
        if !self.bus_exists_in_world(to_bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", to_bus))); }
        let params = if std_type.is_none() { Some(crate::basic::ecs::elements::LineParams { r_ohm_per_km, x_ohm_per_km, g_us_per_km, c_nf_per_km, length_km, df: 1.0, parallel, max_i_ka }) } else { None };
        let handle = self.inner.add_line(&mut self.buffer, from_bus, to_bus, length_km, std_type, params, name).into();
        self.buffer.apply(self.inner.world_mut()); Ok(handle)
    }

    fn bus_exists_in_world(&self, bus_id: i64) -> bool {
        let world = self.inner.world();
        world.iter_entities().filter_map(|e| e.get::<BusID>()).any(|id| id.0 == bus_id)
    }

    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_load(&mut self, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> PyResult<LoadHandle> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let handle = self.inner.add_load(&mut self.buffer, bus, p_mw, q_mvar, name).into();
        self.buffer.apply(self.inner.world_mut()); Ok(handle)
    }

    #[pyo3(signature = (bus, p_mw, vm_pu=1.0, p_min=-1000.0, p_max=1000.0, q_min=-1000.0, q_max=1000.0, name=None))]
    fn add_gen(&mut self, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> PyResult<GenHandle> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let handle = self.inner.add_gen(&mut self.buffer, bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name).into();
        self.buffer.apply(self.inner.world_mut()); Ok(handle)
    }

    #[pyo3(signature = (bus, vm_pu=1.0, va_degree=0.0, name=None))]
    fn add_ext_grid(&mut self, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> PyResult<ExtGridHandle> {
        if !self.bus_exists_in_world(bus) { return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Bus ID {} not found.", bus))); }
        let handle = self.inner.add_ext_grid(&mut self.buffer, bus, vm_pu, va_degree, name).into();
        self.buffer.apply(self.inner.world_mut()); Ok(handle)
    }

    fn init_pf(&mut self) {
        use bevy_ecs::schedule::Schedules; use bevy_app::Startup;
        let world = self.inner.world_mut();
        let mut schedules = world.get_resource_mut::<Schedules>().unwrap();
        if let Some(mut s) = schedules.remove(Startup) { s.run(world); }
        self.sync_next_bus_id();
    }

    fn get_bus_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
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

    fn get_line_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world();
        let mut from_bus = Vec::new(); let mut to_bus = Vec::new(); let mut length_km = Vec::new(); 
        let mut r_ohm_per_km = Vec::new(); let mut x_ohm_per_km = Vec::new(); 
        let mut max_i_ka = Vec::new(); let mut names = Vec::new();
        world.iter_entities().for_each(|e| {
            if let (Some(f), Some(t), Some(p)) = (e.get::<FromBus>(), e.get::<ToBus>(), e.get::<crate::basic::ecs::elements::LineParams>()) {
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

    fn get_load_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
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

    fn display_case_buses(&mut self, py: Python<'_>) -> PyResult<()> { let res = self.get_bus_params(py)?; let df = py.import("pandas")?.call_method1("DataFrame", (res,))?; println!("--- Bus Configuration ---\n{}", df.call_method0("__str__")?.extract::<String>()?); Ok(()) }
    fn display_case_lines(&mut self, py: Python<'_>) -> PyResult<()> { let res = self.get_line_params(py)?; let df = py.import("pandas")?.call_method1("DataFrame", (res,))?; println!("--- Line Configuration ---\n{}", df.call_method0("__str__")?.extract::<String>()?); Ok(()) }
    fn display_case_loads(&mut self, py: Python<'_>) -> PyResult<()> { let res = self.get_load_params(py)?; let df = py.import("pandas")?.call_method1("DataFrame", (res,))?; println!("--- Load Configuration ---\n{}", df.call_method0("__str__")?.extract::<String>()?); Ok(()) }

    fn solve(&mut self) { self.inner.run_pf(); self.inner.post_process(); }

    fn get_bus_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut bus_ids = Vec::new(); let mut v_complex = Vec::new(); let mut vms = Vec::new(); let mut vas = Vec::new(); let mut ps = Vec::new(); let mut qs = Vec::new();
        let mut query = world.query::<(&BusID, &VBusResult, &SBusResult)>();
        for (id, v, s) in query.iter(world) { bus_ids.push(id.0); v_complex.push(v.0); vms.push(v.0.norm()); vas.push(v.0.arg().to_degrees()); ps.push(s.0.re()); qs.push(s.0.im()); }
        let dict = pyo3::types::PyDict::new(py); dict.set_item("bus_id", bus_ids.into_pyarray(py))?; dict.set_item("v_pu", v_complex.into_pyarray(py))?; dict.set_item("vm_pu", vms.into_pyarray(py))?; dict.set_item("va_degree", vas.into_pyarray(py))?; dict.set_item("p_mw", ps.into_pyarray(py))?; dict.set_item("q_mvar", qs.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_line_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut from_bus = Vec::new(); let mut to_bus = Vec::new(); let mut p_f = Vec::new(); let mut q_f = Vec::new(); let mut p_t = Vec::new(); let mut q_t = Vec::new(); let mut pl = Vec::new(); let mut ql = Vec::new(); let mut i_f = Vec::new(); let mut i_t = Vec::new(); let mut i_max = Vec::new(); let mut loading = Vec::new();
        let mut query = world.query::<(&FromBus, &ToBus, &crate::basic::ecs::post_processing::LineResultData)>();
        for (f, t, data) in query.iter(world) { from_bus.push(f.0); to_bus.push(t.0); p_f.push(data.p_from_mw); q_f.push(data.q_from_mvar); p_t.push(data.p_to_mw); q_t.push(data.q_to_mvar); pl.push(data.pl_mw); ql.push(data.ql_mvar); i_f.push(data.i_from_ka); i_t.push(data.i_to_ka); i_max.push(data.i_ka); loading.push(data.loading_percent); }
        let dict = pyo3::types::PyDict::new(py); dict.set_item("from_bus", from_bus.into_pyarray(py))?; dict.set_item("to_bus", to_bus.into_pyarray(py))?; dict.set_item("p_from_mw", p_f.into_pyarray(py))?; dict.set_item("q_from_mvar", q_f.into_pyarray(py))?; dict.set_item("p_to_mw", p_t.into_pyarray(py))?; dict.set_item("q_to_mvar", q_t.into_pyarray(py))?; dict.set_item("pl_mw", pl.into_pyarray(py))?; dict.set_item("ql_mvar", ql.into_pyarray(py))?; dict.set_item("i_from_ka", i_f.into_pyarray(py))?; dict.set_item("i_to_ka", i_t.into_pyarray(py))?; dict.set_item("i_ka", i_max.into_pyarray(py))?; dict.set_item("loading_percent", loading.into_pyarray(py))?;
        Ok(dict)
    }

    fn display_buses(&mut self, py: Python<'_>) -> PyResult<()> { let res = self.get_bus_results(py)?; let df = py.import("pandas")?.call_method1("DataFrame", (res,))?; println!("--- Bus Results ---\n{}", df.call_method0("__str__")?.extract::<String>()?); Ok(()) }
    fn display_lines(&mut self, py: Python<'_>) -> PyResult<()> { let res = self.get_line_results(py)?; let df = py.import("pandas")?.call_method1("DataFrame", (res,))?; println!("--- Line Results ---\n{}", df.call_method0("__str__")?.extract::<String>()?); Ok(()) }

    #[getter] fn n_bus(&self) -> usize { let world = self.inner.world(); if let Some(id) = world.components().get_id(std::any::TypeId::of::<BusID>()) { world.archetypes().iter().filter(|a| a.contains(id)).map(|a| a.len() as usize).sum() } else { 0 } }
    #[getter] fn n_line(&self) -> usize { let world = self.inner.world(); if let Some(id) = world.components().get_id(std::any::TypeId::of::<crate::basic::ecs::elements::Line>()) { world.archetypes().iter().filter(|a| a.contains(id)).map(|a| a.len() as usize).sum() } else { 0 } }
    #[getter] fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> { let world = self.inner.world(); let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(res.v.as_slice().to_vec().into_pyarray(py)) }
    #[getter] fn iterations(&self) -> PyResult<usize> { let results = self.inner.world().get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(results.iterations) }
    #[getter] fn converged(&self) -> PyResult<bool> { let results = self.inner.world().get_resource::<PowerFlowResult>().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow has not been run yet"))?; Ok(results.converged) }
}
