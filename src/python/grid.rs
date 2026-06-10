use pyo3::prelude::*;
use numpy::IntoPyArray;
use crate::prelude::*;
use crate::basic::ecs::elements::*;
use crate::basic::ecs::elements::generator::*;
use crate::basic::ecs::powerflow::prelude::*;
use crate::basic::ecs::post_processing::*;
use crate::io::pandapower::load_csv_zip;
use pyo3::types::PyDictMethods;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;

use crate::basic::ecs::factory::GridFactory;
use crate::basic::ecs::network::{PowerFlowSolver, DataOps};

use super::handles::*;

#[pyclass(unsendable)]
pub struct PowerGrid {
    pub(crate) inner: crate::prelude::PowerGrid,
    pub(crate) buffer: crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer,
    pub(crate) next_bus_id: i64,
    pub(crate) id_map: std::collections::HashMap<i64, i64>,
    pub(crate) bus_to_elements: std::collections::HashMap<i64, Vec<Entity>>,
}

/// Report returned by PowerGrid.solve(). Truthy iff converged.
#[pyclass]
pub struct SolveReport {
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub runtime_ms: f64,
    /// Which rebuild level this solve triggered: "full" | "incremental"
    #[pyo3(get)]
    pub rebuild: String,
}

#[pymethods]
impl SolveReport {
    fn __bool__(&self) -> bool { self.converged }
    fn __repr__(&self) -> String {
        format!(
            "SolveReport(converged={}, iterations={}, runtime_ms={:.3}, rebuild='{}')",
            self.converged, self.iterations, self.runtime_ms, self.rebuild
        )
    }
}

/// Transactional editor (Unit of Work). All topology mutations go through here.
/// Commands are buffered (Harvard command queue, fused insert) and applied once
/// on commit; an exception inside the `with` block aborts the transaction.
#[pyclass(unsendable)]
pub struct GridEditor {
    pub(crate) parent: Py<PowerGrid>,
    /// Entities allocated during this transaction (for abort rollback).
    created: Vec<u64>,
    /// next_bus_id snapshot at transaction start (for abort rollback).
    start_next_bus_id: i64,
}

#[pymethods]
impl GridEditor {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __exit__(&mut self, py: Python<'_>, exc_type: PyObject, _exc_value: PyObject, _traceback: PyObject) -> PyResult<()> {
        if exc_type.is_none(py) { self.commit(py) } else { self.abort(py) }
    }

    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(&mut self, py: Python<'_>, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> PyResult<(i64, BusHandle)> {
        let mut parent = self.parent.borrow_mut(py);
        let id = parent.next_bus_id;
        parent.next_bus_id += 1;
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_bus(buffer, id, vn_kv, name, vm_min, vm_max, zone);
        self.created.push(entity.to_bits());
        Ok((id, BusHandle::new(entity, self.parent.clone_ref(py))))
    }

    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(&mut self, py: Python<'_>, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, r_ohm_per_km: f64, x_ohm_per_km: f64, c_nf_per_km: f64, g_us_per_km: f64, parallel: i32, max_i_ka: f64, name: Option<String>) -> PyResult<LineHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let params = if std_type.is_none() { Some(LineParams { r_ohm_per_km, x_ohm_per_km, g_us_per_km, c_nf_per_km, length_km, df: 1.0, parallel, max_i_ka }) } else { None };
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_line(buffer, from_bus, to_bus, length_km, std_type, params, name);
        self.created.push(entity.to_bits());
        Ok(LineHandle::new(entity, self.parent.clone_ref(py)))
    }

    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_load(&mut self, py: Python<'_>, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> PyResult<LoadHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, bus_to_elements, .. } = &mut *parent;
        let entity = inner.add_load(buffer, bus, p_mw, q_mvar, name);
        bus_to_elements.entry(bus).or_default().push(entity);
        self.created.push(entity.to_bits());
        Ok(LoadHandle::new(entity, self.parent.clone_ref(py)))
    }

    #[pyo3(signature = (bus, p_mw, vm_pu=1.0, p_min=-1000.0, p_max=1000.0, q_min=-1000.0, q_max=1000.0, name=None))]
    fn add_gen(&mut self, py: Python<'_>, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> PyResult<GenHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, bus_to_elements, .. } = &mut *parent;
        let entity = inner.add_gen(buffer, bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name);
        bus_to_elements.entry(bus).or_default().push(entity);
        self.created.push(entity.to_bits());
        Ok(GenHandle::new(entity, self.parent.clone_ref(py)))
    }

    #[pyo3(signature = (bus, vm_pu=1.0, va_degree=0.0, name=None))]
    fn add_ext_grid(&mut self, py: Python<'_>, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> PyResult<ExtGridHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_ext_grid(buffer, bus, vm_pu, va_degree, name);
        self.created.push(entity.to_bits());
        Ok(ExtGridHandle::new(entity, self.parent.clone_ref(py)))
    }

    #[pyo3(signature = (hv_bus, lv_bus, sn_mva=1.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=0.1, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, tap_pos=0.0, tap_neutral=0.0, tap_step_percent=1.25, name=None))]
    fn add_trafo(&mut self, py: Python<'_>, hv_bus: i64, lv_bus: i64, sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64, shift_degree: f64, tap_pos: f64, tap_neutral: f64, tap_step_percent: f64, name: Option<String>) -> PyResult<TrafoHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let params = make_trafo_device(sn_mva, vn_hv_kv, vn_lv_kv, vk_percent, vkr_percent, pfe_kw, i0_percent, shift_degree, tap_pos, tap_neutral, tap_step_percent);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_trafo(buffer, hv_bus, lv_bus, None, Some(params), name);
        self.created.push(entity.to_bits());
        Ok(TrafoHandle::new(entity, self.parent.clone_ref(py)))
    }

    #[pyo3(signature = (bus, q_mvar, p_mw=0.0, vn_kv=110.0, step=1, name=None))]
    fn add_shunt(&mut self, py: Python<'_>, bus: i64, q_mvar: f64, p_mw: f64, vn_kv: f64, step: i32, name: Option<String>) -> PyResult<ShuntHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_shunt(buffer, bus, p_mw, q_mvar, vn_kv, step, name);
        self.created.push(entity.to_bits());
        Ok(ShuntHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Remove an element by its handle. Accepts any element handle; removing a
    /// bus also removes the loads/gens attached to it.
    /// Phase 1 limitation: removal despawns immediately and is NOT rolled back
    /// by transaction abort (real inverse commands arrive with undo in Phase 3).
    fn remove(&mut self, py: Python<'_>, element: Bound<'_, PyAny>) -> PyResult<()> {
        let entity = extract_handle_entity(&element)?;
        let is_bus = element.extract::<PyRef<'_, BusHandle>>().is_ok();
        let mut parent = self.parent.borrow_mut(py);

        if is_bus {
            let bus_id = parent.inner.world().get::<BusID>(entity).map(|b| b.0);
            if let Some(bus_id) = bus_id {
                let attached = parent.bus_to_elements.remove(&bus_id).unwrap_or_default();
                let world = parent.inner.world_mut();
                for e in attached {
                    if world.get_entity(e).is_ok() { world.entity_mut(e).despawn(); }
                }
            }
        }
        let world = parent.inner.world_mut();
        if world.get_entity(entity).is_ok() { world.entity_mut(entity).despawn(); }
        Ok(())
    }

    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut parent = self.parent.borrow_mut(py);
        {
            let PowerGrid { inner, buffer, .. } = &mut *parent;
            buffer.apply(inner.world_mut());
        }
        // Topology changed: post the rebuild event; the next solve consumes
        // it and runs the PFInit schedule.
        let _ = parent
            .inner
            .world_mut()
            .write_message(crate::basic::ecs::powerflow::structure_update::FullRebuildEvent);
        parent.sync_bus_to_elements();
        self.created.clear();
        Ok(())
    }

    /// Discard all buffered commands and despawn entities allocated in this
    /// transaction. The world is left as it was at transaction start.
    fn abort(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut parent = self.parent.borrow_mut(py);
        parent.buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        {
            let world = parent.inner.world_mut();
            for bits in self.created.drain(..) {
                let e = Entity::from_bits(bits);
                if world.get_entity(e).is_ok() { world.entity_mut(e).despawn(); }
            }
        }
        parent.next_bus_id = self.start_next_bus_id;
        parent.sync_bus_to_elements();
        Ok(())
    }
}

fn extract_handle_entity(element: &Bound<'_, PyAny>) -> PyResult<Entity> {
    macro_rules! try_handle {
        ($($t:ty),+) => {
            $(if let Ok(h) = element.extract::<PyRef<'_, $t>>() { return Ok(h.entity()); })+
        };
    }
    try_handle!(BusHandle, LineHandle, TrafoHandle, LoadHandle, GenHandle, ExtGridHandle, ShuntHandle, SGenHandle, SwitchHandle);
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("expected an element handle"))
}

fn make_trafo_device(sn_mva: f64, vn_hv_kv: f64, vn_lv_kv: f64, vk_percent: f64, vkr_percent: f64, pfe_kw: f64, i0_percent: f64, shift_degree: f64, tap_pos: f64, tap_neutral: f64, tap_step_percent: f64) -> crate::basic::ecs::elements::TransformerDevice {
    crate::basic::ecs::elements::TransformerDevice {
        df: 1.0, i0_percent, pfe_kw, vk_percent, vkr_percent, shift_degree, sn_mva, vn_hv_kv, vn_lv_kv, max_loading_percent: None, parallel: 1,
        tap: Some(crate::basic::ecs::elements::TapChanger {
            side: Some("hv".to_string()), neutral: Some(tap_neutral), max: Some(tap_pos + 10.0), min: Some(tap_pos - 10.0), pos: Some(tap_pos), step_degree: None, step_percent: Some(tap_step_percent), is_phase_shifter: false,
        }),
    }
}

#[pymethods]
impl PowerGrid {
    #[new]
    #[pyo3(signature = (case_path=None, _qlim=false, **kwargs))]
    fn new(case_path: Option<String>, _qlim: bool, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        let buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        inner.world_mut().insert_resource(crate::basic::ecs::factory::StdTypeLibrary::default());
        // Default system base; overwritten by case ingestion or set_base().
        inner.world_mut().insert_resource(PFCommonData {
            sbase: 100.0,
            f_hz: 50.0,
            wbase: 2.0 * std::f64::consts::PI * 50.0,
        });

        // BasePFPlugin registers ecs_run_pf in the Solve stage (and inserts
        // PowerFlowConfig / PowerFlowSolver); StructureUpdatePlugin keeps the
        // solver matrices in sync with component changes between solves;
        // VBusUpdatePlugin writes solved voltages back for warm starts.
        inner.app_mut().add_plugins((
            crate::basic::ecs::plugin::BasePFPlugin,
            crate::basic::ecs::powerflow::structure_update::StructureUpdatePlugin,
            crate::basic::ecs::powerflow::result_extract::VBusUpdatePlugin,
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
        } else {
            grid.sync_next_bus_id();
        }
        Ok(grid)
    }

    fn from_pp_net(slf: Py<Self>, py: Python<'_>, net: Bound<'_, PyAny>) -> PyResult<()> {
        let mut rust_net = crate::io::pandapower::Network::default();
        rust_net.from_pp_net(net)?;
        let mut grid_py = slf.borrow_mut(py);
        // Loading a network replaces the grid: clear any previously spawned
        // entities so buses/branches are not duplicated. Existing handles
        // become invalid after this call.
        grid_py.clear_grid_entities();
        grid_py.inner.world_mut().insert_resource(PPNetwork(rust_net));
        grid_py.init_pf();
        Ok(())
    }

    fn load_network(&mut self, net: crate::io::pandapower::Network) {
        self.clear_grid_entities();
        self.inner.world_mut().insert_resource(PPNetwork(net));
        self.init_pf();
    }

    /// Synchronous full rebuild: runs the unified PFInit schedule (the same
    /// pipeline that FullRebuildEvent triggers inside solve()). Kept public
    /// for explicit use; normal workflows never need to call it.
    fn init_pf(&mut self) {
        let _ = self
            .inner
            .world_mut()
            .try_run_schedule(crate::basic::ecs::powerflow::pf_init::PFInit);
        self.sync_next_bus_id();
        self.sync_bus_to_elements();
    }

    /// Run the power flow. Fully event-driven: pending FullRebuildEvents
    /// (editor commits, in_service toggles) make structure_update run the
    /// PFInit schedule inside this same update; parameter changes flow
    /// through the ParamDiff bus. Raises only on validation errors (empty
    /// grid, no slack); divergence is a legal result reported through the
    /// (falsy) SolveReport.
    fn solve(&mut self) -> PyResult<SolveReport> {
        use crate::basic::ecs::powerflow::structure_update::{FullRebuildEvent, LastStructureAction};
        let t0 = std::time::Instant::now();

        // First-ever build: nothing has posted a rebuild event yet.
        if !self.inner.world().contains_resource::<PowerFlowMat>() {
            let _ = self.inner.world_mut().write_message(FullRebuildEvent);
        }

        self.inner.run_pf();

        let full_rebuild = self
            .inner
            .world()
            .get_resource::<LastStructureAction>()
            .map(|a| a.full_rebuild)
            .unwrap_or(false);
        if full_rebuild {
            self.sync_next_bus_id();
            self.sync_bus_to_elements();
        }

        // Post-rebuild validation with readable errors (the solver itself
        // reports a degenerate problem as non-convergence, never a panic).
        {
            let world = self.inner.world_mut();
            let n_bus = world.query::<&BusID>().iter(world).count();
            if n_bus == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Cannot solve an empty grid: add buses and an ext_grid first",
                ));
            }
            let n_slack = world.query_filtered::<Entity, With<SlackBus>>().iter(world).count();
            if n_slack == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "No slack bus: add an ext_grid to provide the voltage/angle reference",
                ));
            }
        }

        let (converged, iterations) = self
            .inner
            .world()
            .get_resource::<PowerFlowResult>()
            .map(|r| (r.converged, r.iterations))
            .unwrap_or((false, 0));
        if converged {
            self.inner.post_process();
        }
        Ok(SolveReport {
            converged,
            iterations,
            runtime_ms: t0.elapsed().as_secs_f64() * 1e3,
            rebuild: if full_rebuild { "full" } else { "incremental" }.to_string(),
        })
    }

    fn reset_state(&mut self) { self.reset_state_impl(); }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.world().get_resource::<PowerFlowResult>().map(|r| r.converged).unwrap_or(false)
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.world().get_resource::<PowerFlowResult>().map(|r| r.iterations).unwrap_or(0)
    }

    /// Complex bus voltages (p.u.) of the last solve, in original bus order.
    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.inner.world();
        let res = world.get_resource::<PowerFlowResult>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No power flow result: call solve() first"))?;
        let mat = world.get_resource::<PowerFlowMat>()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Power flow not initialized"))?;
        let mut out = vec![num_complex::Complex64::new(0.0, 0.0); res.v.len()];
        for (i, &val) in res.v.iter().enumerate() {
            out[mat.from_perm[i]] = val;
        }
        Ok(out.into_pyarray(py))
    }

    #[getter]
    fn n_bus(&mut self) -> usize {
        let world = self.inner.world_mut();
        world.query::<&BusID>().iter(world).count()
    }

    /// Open a transactional editor. All topology mutations (add_*/remove) go
    /// through it; `with grid.edit() as e:` commits on clean exit and aborts
    /// on exception.
    fn edit(slf: Py<Self>, py: Python<'_>) -> GridEditor {
        let start_next_bus_id = slf.borrow(py).next_bus_id;
        GridEditor { parent: slf.clone_ref(py), created: Vec::new(), start_next_bus_id }
    }

    /// Build a PowerGrid directly from a live pandapower net object.
    /// The pandapower data model is discarded after ingestion.
    #[classmethod]
    fn from_pandapower(_cls: &Bound<'_, pyo3::types::PyType>, py: Python<'_>, net: Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let grid = Self::new(None, false, None)?;
        let slf = Py::new(py, grid)?;
        Self::from_pp_net(slf.clone_ref(py), py, net)?;
        Ok(slf)
    }

    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        GridFactory::set_base(&mut self.inner, f_hz, sn_mva);
    }

    fn bus(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<BusHandle> {
        let grid = slf.borrow(py);
        let world = grid.inner.world();
        let lookup = world.get_resource::<NodeLookup>()?;
        let entity = lookup.get_entity(id)?;
        Some(BusHandle::new(entity, slf.clone_ref(py)))
    }

    /// First load matching the filters, or None. Query-backed: no shadow index.
    #[pyo3(signature = (bus=None, name=None))]
    fn load(slf: Py<Self>, py: Python<'_>, bus: Option<i64>, name: Option<String>) -> Option<LoadHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world.query_filtered::<(Entity, &TargetBus, Option<&bevy_ecs::name::Name>), With<LoadCfg>>();
        for (e, tb, n) in q.iter(world) {
            if let Some(b) = bus { if tb.0 != b { continue; } }
            if let Some(ref nm) = name { if n.map(|x| x.as_str()) != Some(nm.as_str()) { continue; } }
            return Some(LoadHandle::new(e, slf.clone_ref(py)));
        }
        None
    }

    /// All loads (optionally restricted to one bus).
    #[pyo3(signature = (bus=None))]
    fn loads(slf: Py<Self>, py: Python<'_>, bus: Option<i64>) -> Vec<LoadHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world.query_filtered::<(Entity, &TargetBus), With<LoadCfg>>();
        let mut out = Vec::new();
        for (e, tb) in q.iter(world) {
            if let Some(b) = bus { if tb.0 != b { continue; } }
            out.push(LoadHandle::new(e, slf.clone_ref(py)));
        }
        out
    }

    /// First PV generator matching the filters (slack excluded), or None.
    #[pyo3(signature = (bus=None, name=None))]
    fn r#gen(slf: Py<Self>, py: Python<'_>, bus: Option<i64>, name: Option<String>) -> Option<GenHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world.query_filtered::<(Entity, &TargetBus, Option<&bevy_ecs::name::Name>, Has<Slack>), With<GeneratorCfg>>();
        for (e, tb, n, is_slack) in q.iter(world) {
            if is_slack { continue; }
            if let Some(b) = bus { if tb.0 != b { continue; } }
            if let Some(ref nm) = name { if n.map(|x| x.as_str()) != Some(nm.as_str()) { continue; } }
            return Some(GenHandle::new(e, slf.clone_ref(py)));
        }
        None
    }

    /// Line between two buses (either direction), or None.
    fn line(slf: Py<Self>, py: Python<'_>, from_bus: i64, to_bus: i64) -> Option<LineHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world.query::<(Entity, &FromBus, &ToBus, &LineParams)>();
        for (e, f, t, _) in q.iter(world) {
            if (f.0 == from_bus && t.0 == to_bus) || (f.0 == to_bus && t.0 == from_bus) {
                return Some(LineHandle::new(e, slf.clone_ref(py)));
            }
        }
        None
    }

    /// Element counts overview.
    fn describe<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let n_bus = world.query::<&BusID>().iter(world).count();
        let n_line = world.query::<&LineParams>().iter(world).count();
        let n_trafo = world.query::<&crate::basic::ecs::elements::TransformerDevice>().iter(world).count();
        let n_load = world.query::<&LoadCfg>().iter(world).count();
        let mut n_gen = 0usize;
        let mut n_ext = 0usize;
        let mut q = world.query_filtered::<Has<Slack>, With<GeneratorCfg>>();
        for is_slack in q.iter(world) {
            if is_slack { n_ext += 1; } else { n_gen += 1; }
        }
        let n_shunt = world.query::<&ShuntDevice>().iter(world).count();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("element", vec!["bus", "line", "trafo", "load", "gen", "ext_grid", "shunt"])?;
        dict.set_item("count", vec![n_bus, n_line, n_trafo, n_load, n_gen, n_ext, n_shunt])?;
        py.import("pandas")?.call_method1("DataFrame", (dict,))
    }

    fn display_case_loads<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let mut buses = Vec::new(); let mut ps = Vec::new(); let mut qs = Vec::new(); let mut names = Vec::new();
        let mut query = world.query_filtered::<(&TargetBus, &TargetPMW, &TargetQMVar, Option<&bevy_ecs::name::Name>), With<LoadCfg>>();
        query.iter(world).for_each(|(bus, p, q, name)| {
            buses.push(bus.0);
            // Targets store injections (consumption is negative); display as consumption
            ps.push(-p.0); qs.push(-q.0);
            names.push(name.map(|n| n.as_str().to_string()).unwrap_or_default());
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus", buses.into_pyarray(py))?; dict.set_item("p_mw", ps.into_pyarray(py))?;
        dict.set_item("q_mvar", qs.into_pyarray(py))?; dict.set_item("name", names)?;
        py.import("pandas")?.call_method1("DataFrame", (dict,))
    }

    #[getter] fn n_line(&mut self) -> usize { let world = self.inner.world_mut(); world.query::<&crate::basic::ecs::elements::Line>().iter(world).count() }

    fn get_bus_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let res = self.get_bus_results_impl(py)?;
        py.import("pandas")?.call_method1("DataFrame", (res,))
    }

    fn get_line_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let res = self.get_line_results_impl(py)?;
        py.import("pandas")?.call_method1("DataFrame", (res,))
    }

    fn get_bus_params<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let res = self.get_bus_params_impl(py)?;
        py.import("pandas")?.call_method1("DataFrame", (res,))
    }

    /// Bus results of the last solve as a DataFrame (pandapower's res_bus).
    #[getter]
    fn res_bus<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { self.get_bus_results(py) }

    /// Line results of the last solve as a DataFrame (pandapower's res_line).
    #[getter]
    fn res_line<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { self.get_line_results(py) }

    fn display_case_buses<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { self.get_bus_params(py) }
}

impl PowerGrid {
    fn bus_exists_in_world(&mut self, bus_id: i64) -> bool {
        let world = self.inner.world_mut();
        if let Some(lookup) = world.get_resource::<NodeLookup>() { lookup.get_entity(bus_id).is_some() } else { bus_id < self.next_bus_id }
    }

    /// Despawn all grid-domain entities (buses, branches, devices) ahead of a
    /// network reload. Deliberately NOT World::clear_entities(): in recent
    /// Bevy, schedules/systems are entity-backed and a blanket clear would
    /// destroy the Main schedule along with the grid.
    pub(crate) fn clear_grid_entities(&mut self) {
        let world = self.inner.world_mut();
        let mut to_despawn: Vec<Entity> = Vec::new();
        macro_rules! collect {
            ($t:ty) => {
                let mut q = world.query_filtered::<Entity, With<$t>>();
                to_despawn.extend(q.iter(world));
            };
        }
        collect!(BusID);
        collect!(TargetBus);
        collect!(LineParams);
        collect!(crate::basic::ecs::elements::TransformerDevice);
        collect!(EShunt);
        collect!(Switch);
        to_despawn.sort_unstable();
        to_despawn.dedup();
        for e in to_despawn {
            // Children (admittance branches) despawn with their parents.
            if world.get_entity(e).is_ok() {
                world.entity_mut(e).despawn();
            }
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
            if let Some(bus) = e.get::<TargetBus>() { self.bus_to_elements.entry(bus.0).or_default().push(e.id()); }
        });
    }

    fn reset_state_impl(&mut self) {
        let world = self.inner.world_mut();
        let mut query = world.query_filtered::<&mut SBusInjPu, With<BusID>>();
        for mut s in query.iter_mut(world) { s.0 = num_complex::Complex64::new(0.0, 0.0); }
        world.remove_resource::<PowerFlowResult>();
    }

    fn get_bus_params_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut bus_ids = Vec::new(); let mut names = Vec::new(); let mut vn_kv = Vec::new(); 
        let mut vm_min = Vec::new(); let mut vm_max = Vec::new(); let mut types = Vec::new();
        let mut query = world.query::<(&BusID, Option<&bevy_ecs::name::Name>, Option<&VNominal>, Option<&VmLimit<PerUnit>>, Has<SlackBus>, Has<PVBus>)>();
        query.iter(world).for_each(|(id, name, vnom, vlim, is_slack, is_pv)| {
            bus_ids.push(id.0); names.push(name.map(|n| n.as_str().to_string()).unwrap_or_default());
            vn_kv.push(vnom.map(|v| v.0.0).unwrap_or(0.0));
            let (min, max) = vlim.map(|l| (l.min(), l.max())).unwrap_or((0.9, 1.1));
            vm_min.push(min); vm_max.push(max);
            types.push(if is_slack { "Slack" } else if is_pv { "PV" } else { "PQ" });
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?; dict.set_item("name", names)?; dict.set_item("type", types)?;
        dict.set_item("vn_kv", vn_kv.into_pyarray(py))?; dict.set_item("vm_min_pu", vm_min.into_pyarray(py))?; dict.set_item("vm_max_pu", vm_max.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_bus_results_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut bus_ids = Vec::new(); let mut v_complex = Vec::new(); let mut vms = Vec::new(); let mut vas = Vec::new(); let mut ps = Vec::new(); let mut qs = Vec::new();
        let mut query = world.query::<(&BusID, &VBusResult, &SBusResult)>();
        query.iter(world).for_each(|(id, v, s)| {
            bus_ids.push(id.0); v_complex.push(v.0); vms.push(v.0.norm()); vas.push(v.0.arg().to_degrees()); ps.push(s.0.re); qs.push(s.0.im);
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?; dict.set_item("v_pu", v_complex.into_pyarray(py))?; dict.set_item("vm_pu", vms.into_pyarray(py))?; dict.set_item("va_degree", vas.into_pyarray(py))?; dict.set_item("p_mw", ps.into_pyarray(py))?; dict.set_item("q_mvar", qs.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_line_results_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut from_bus = Vec::new(); let mut to_bus = Vec::new();
        let mut p_f = Vec::new(); let mut q_f = Vec::new(); let mut i_f = Vec::new();
        let mut query = world.query::<(&FromBus, &ToBus, &LineResultData)>();
        query.iter(world).for_each(|(f, t, d)| {
            from_bus.push(f.0); to_bus.push(t.0);
            p_f.push(d.p_from_mw); q_f.push(d.q_from_mvar); i_f.push(d.i_from_ka);
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("from_bus", from_bus.into_pyarray(py))?; dict.set_item("to_bus", to_bus.into_pyarray(py))?;
        dict.set_item("p_mw", p_f.into_pyarray(py))?; dict.set_item("q_mvar", q_f.into_pyarray(py))?; dict.set_item("i_ka", i_f.into_pyarray(py))?;
        Ok(dict)
    }
}
