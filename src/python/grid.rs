use crate::basic::ecs::elements::*;
use crate::basic::ecs::post_processing::*;
use crate::basic::ecs::powerflow::prelude::*;
use crate::io::pandapower::load_csv_zip;
use crate::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDictMethods;

use crate::basic::ecs::factory::GridFactory;
use crate::basic::ecs::network::DataOps;

use super::handles::*;

/// Core power grid object.
///
/// Supports three primary workflows:
/// A. Batch:      PowerGrid("case.zip").solve(); grid.res_bus
/// B. Parameter:  load.p_mw = p; grid.solve()        # incremental, warm start
/// C. Topology:   with grid.edit() as e: ...; grid.solve()  # auto rebuild
#[pyclass(unsendable)]
pub struct PowerGrid {
    pub(crate) inner: crate::prelude::PowerGrid,
    pub(crate) buffer: crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer,
    pub(crate) next_bus_id: i64,
    /// Lazy post-processing: set to `true` after a converged solve(),
    /// cleared when `res_bus` / `res_line` (or any result getter) is accessed.
    post_process_dirty: bool,
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
    fn __bool__(&self) -> bool {
        self.converged
    }
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
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<()> {
        if exc_type.is_none(py) {
            self.commit(py)
        } else {
            self.abort(py)
        }
    }

    /// Add a bus to the grid transaction.
    ///
    /// vn_kv: Nominal voltage in kV.
    /// name: Optional name of the bus.
    /// vm_min: Minimum voltage limit in p.u. (default: 0.9).
    /// vm_max: Maximum voltage limit in p.u. (default: 1.1).
    /// zone: Optional zone identifier.
    #[pyo3(signature = (vn_kv, name=None, vm_min=0.9, vm_max=1.1, zone=0))]
    fn add_bus(
        &mut self,
        py: Python<'_>,
        vn_kv: f64,
        name: Option<String>,
        vm_min: f64,
        vm_max: f64,
        zone: i64,
    ) -> PyResult<(i64, BusHandle)> {
        let mut parent = self.parent.borrow_mut(py);
        let id = parent.next_bus_id;
        parent.next_bus_id += 1;
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_bus(buffer, id, vn_kv, name, vm_min, vm_max, zone);
        self.created.push(entity.to_bits());
        Ok((id, BusHandle::new(entity, self.parent.clone_ref(py))))
    }

    /// Add a line between two buses to the grid transaction.
    ///
    /// from_bus: The source bus ID.
    /// to_bus: The destination bus ID.
    /// length_km: Line length in km.
    /// std_type: Optional standard type name for looking up parameters.
    /// r_ohm_per_km: Resistance in Ohm/km (default: 0.1).
    /// x_ohm_per_km: Reactance in Ohm/km (default: 0.1).
    /// c_nf_per_km: Capacitance in nF/km (default: 0.0).
    /// g_us_per_km: Conductance in uS/km (default: 0.0).
    /// parallel: Number of parallel lines (default: 1).
    /// max_i_ka: Maximum current capacity in kA (default: 0.0).
    /// name: Optional name of the line.
    #[pyo3(signature = (from_bus, to_bus, length_km, std_type=None, r_ohm_per_km=0.1, x_ohm_per_km=0.1, c_nf_per_km=0.0, g_us_per_km=0.0, parallel=1, max_i_ka=0.0, name=None))]
    fn add_line(
        &mut self,
        py: Python<'_>,
        from_bus: i64,
        to_bus: i64,
        length_km: f64,
        std_type: Option<String>,
        r_ohm_per_km: f64,
        x_ohm_per_km: f64,
        c_nf_per_km: f64,
        g_us_per_km: f64,
        parallel: i32,
        max_i_ka: f64,
        name: Option<String>,
    ) -> PyResult<LineHandle> {
        let mut parent = self.parent.borrow_mut(py);
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
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_line(buffer, from_bus, to_bus, length_km, std_type, params, name);
        self.created.push(entity.to_bits());
        Ok(LineHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Add a load to a bus in the grid transaction.
    ///
    /// bus: Bus ID where the load is attached.
    /// p_mw: Active power consumption (MW, positive for consumption).
    /// q_mvar: Reactive power consumption (MVar, positive for consumption).
    /// name: Optional name of the load.
    #[pyo3(signature = (bus, p_mw, q_mvar, name=None))]
    fn add_load(
        &mut self,
        py: Python<'_>,
        bus: i64,
        p_mw: f64,
        q_mvar: f64,
        name: Option<String>,
    ) -> PyResult<LoadHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_load(buffer, bus, p_mw, q_mvar, name);
        self.created.push(entity.to_bits());
        Ok(LoadHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Add a PV generator to a bus in the grid transaction.
    ///
    /// bus: Bus ID where the generator is attached.
    /// p_mw: Active power production (MW).
    /// vm_pu: Target voltage magnitude in p.u. (default: 1.0).
    /// p_min/p_max: Active power limits in MW.
    /// q_min/q_max: Reactive power limits in MVar.
    /// name: Optional name of the generator.
    #[pyo3(signature = (bus, p_mw, vm_pu=1.0, p_min=-1000.0, p_max=1000.0, q_min=-1000.0, q_max=1000.0, name=None))]
    fn add_gen(
        &mut self,
        py: Python<'_>,
        bus: i64,
        p_mw: f64,
        vm_pu: f64,
        p_min: f64,
        p_max: f64,
        q_min: f64,
        q_max: f64,
        name: Option<String>,
    ) -> PyResult<GenHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_gen(buffer, bus, p_mw, vm_pu, p_min, p_max, q_min, q_max, name);
        self.created.push(entity.to_bits());
        Ok(GenHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Add an external grid connection (Slack bus reference) to a bus in the grid transaction.
    ///
    /// bus: Bus ID to connect to.
    /// vm_pu: Target voltage magnitude in p.u. (default: 1.0).
    /// va_degree: Reference voltage angle in degrees (default: 0.0).
    /// name: Optional name of the external grid.
    #[pyo3(signature = (bus, vm_pu=1.0, va_degree=0.0, name=None))]
    fn add_ext_grid(
        &mut self,
        py: Python<'_>,
        bus: i64,
        vm_pu: f64,
        va_degree: f64,
        name: Option<String>,
    ) -> PyResult<ExtGridHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_ext_grid(buffer, bus, vm_pu, va_degree, name);
        self.created.push(entity.to_bits());
        Ok(ExtGridHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Add a transformer between a high voltage bus and a low voltage bus to the grid transaction.
    ///
    /// hv_bus: High voltage bus ID.
    /// lv_bus: Low voltage bus ID.
    /// sn_mva: Nominal apparent power in MVA.
    /// vn_hv_kv: Rated voltage of the HV side in kV.
    /// vn_lv_kv: Rated voltage of the LV side in kV.
    /// vk_percent: Short-circuit voltage in percent.
    /// vkr_percent: Real part of short-circuit voltage in percent.
    /// pfe_kw: Iron losses in kW.
    /// i0_percent: Open-loop current in percent.
    /// shift_degree: Phase shift angle in degrees.
    /// tap_pos: Current tap position.
    /// tap_neutral: Neutral tap position.
    /// tap_step_percent: Tap step size in percent.
    /// name: Optional name of the transformer.
    #[pyo3(signature = (hv_bus, lv_bus, sn_mva=1.0, vn_hv_kv=110.0, vn_lv_kv=10.0, vk_percent=10.0, vkr_percent=0.1, pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0, tap_pos=0.0, tap_neutral=0.0, tap_step_percent=1.25, name=None))]
    fn add_trafo(
        &mut self,
        py: Python<'_>,
        hv_bus: i64,
        lv_bus: i64,
        sn_mva: f64,
        vn_hv_kv: f64,
        vn_lv_kv: f64,
        vk_percent: f64,
        vkr_percent: f64,
        pfe_kw: f64,
        i0_percent: f64,
        shift_degree: f64,
        tap_pos: f64,
        tap_neutral: f64,
        tap_step_percent: f64,
        name: Option<String>,
    ) -> PyResult<TrafoHandle> {
        let mut parent = self.parent.borrow_mut(py);
        let params = make_trafo_device(
            sn_mva,
            vn_hv_kv,
            vn_lv_kv,
            vk_percent,
            vkr_percent,
            pfe_kw,
            i0_percent,
            shift_degree,
            tap_pos,
            tap_neutral,
            tap_step_percent,
        );
        let PowerGrid { inner, buffer, .. } = &mut *parent;
        let entity = inner.add_trafo(buffer, hv_bus, lv_bus, None, Some(params), name);
        self.created.push(entity.to_bits());
        Ok(TrafoHandle::new(entity, self.parent.clone_ref(py)))
    }

    /// Add a shunt element to a bus in the grid transaction.
    ///
    /// bus: Bus ID where the shunt is attached.
    /// q_mvar: Reactive power consumption of the shunt in MVar at nominal voltage.
    /// p_mw: Active power consumption of the shunt in MW at nominal voltage (default: 0.0).
    /// vn_kv: Rated voltage of the shunt in kV.
    /// step: Number of active steps (default: 1).
    /// name: Optional name of the shunt.
    #[pyo3(signature = (bus, q_mvar, p_mw=0.0, vn_kv=110.0, step=1, name=None))]
    fn add_shunt(
        &mut self,
        py: Python<'_>,
        bus: i64,
        q_mvar: f64,
        p_mw: f64,
        vn_kv: f64,
        step: i32,
        name: Option<String>,
    ) -> PyResult<ShuntHandle> {
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
                let world = parent.inner.world_mut();
                let attached: Vec<Entity> = {
                    let mut q = world.query::<(Entity, &TargetBus)>();
                    q.iter(world)
                        .filter_map(|(e, tb)| if tb.0 == bus_id { Some(e) } else { None })
                        .collect()
                };
                for e in attached {
                    if world.get_entity(e).is_ok() {
                        world.entity_mut(e).despawn();
                    }
                }
            }
        }
        let world = parent.inner.world_mut();
        if world.get_entity(entity).is_ok() {
            world.entity_mut(entity).despawn();
        }
        Ok(())
    }

    /// Commit all buffered topology modifications to the grid.
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
                if world.get_entity(e).is_ok() {
                    world.entity_mut(e).despawn();
                }
            }
        }
        parent.next_bus_id = self.start_next_bus_id;
        Ok(())
    }
}

fn extract_handle_entity(element: &Bound<'_, PyAny>) -> PyResult<Entity> {
    macro_rules! try_handle {
        ($($t:ty),+) => {
            $(if let Ok(h) = element.extract::<PyRef<'_, $t>>() { return Ok(h.entity()); })+
        };
    }
    try_handle!(
        BusHandle,
        LineHandle,
        TrafoHandle,
        LoadHandle,
        GenHandle,
        ExtGridHandle,
        ShuntHandle,
        SGenHandle,
        SwitchHandle
    );
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "expected an element handle",
    ))
}

fn make_trafo_device(
    sn_mva: f64,
    vn_hv_kv: f64,
    vn_lv_kv: f64,
    vk_percent: f64,
    vkr_percent: f64,
    pfe_kw: f64,
    i0_percent: f64,
    shift_degree: f64,
    tap_pos: f64,
    tap_neutral: f64,
    tap_step_percent: f64,
) -> crate::basic::ecs::elements::TransformerDevice {
    crate::basic::ecs::elements::TransformerDevice {
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
        parallel: 1,
        tap: Some(crate::basic::ecs::elements::TapChanger {
            side: Some("hv".to_string()),
            neutral: Some(tap_neutral),
            max: Some(tap_pos + 10.0),
            min: Some(tap_pos - 10.0),
            pos: Some(tap_pos),
            step_degree: None,
            step_percent: Some(tap_step_percent),
            is_phase_shifter: false,
        }),
    }
}

#[pymethods]
impl PowerGrid {
    /// Create a new PowerGrid instance.
    ///
    /// case_path: Optional path to a pandapower ZIP case file to load.
    /// qlim: If True, enforces generator reactive power limits by dynamically demoting PV buses to PQ when they saturate.
    /// kwargs: Additional configuration flags (e.g. branch_analysis=True).
    #[new]
    #[pyo3(signature = (case_path=None, qlim=false, **kwargs))]
    fn new(
        case_path: Option<String>,
        qlim: bool,
        kwargs: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let mut inner = crate::prelude::PowerGrid::default();
        let buffer = crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer::new();
        inner
            .world_mut()
            .insert_resource(crate::basic::ecs::factory::StdTypeLibrary::default());
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

        #[cfg(feature = "archive")]
        inner
            .app_mut()
            .add_plugins(crate::io::archive::aurora_format::ArchivePlugin);

        // Generator reactive limit enforcement: PV->PQ demotion via the
        // NodeTypeChangeEvent path (re-derives matrices from current tags,
        // never relabels) inside an outer iteration loop.
        if qlim {
            inner
                .app_mut()
                .add_plugins(crate::basic::ecs::powerflow::qlim::QLimPlugin);
        }

        if let Some(args) = kwargs {
            if let Ok(Some(branch_analysis)) = args.get_item("branch_analysis") {
                if branch_analysis.extract::<bool>()? {
                    inner.app_mut().add_plugins(
                        crate::basic::ecs::powerflow::branch_data::BranchAnalysisPlugin,
                    );
                }
            }
        }

        let mut grid = Self {
            inner,
            buffer,
            next_bus_id: 0,
            post_process_dirty: false,
        };

        if let Some(path) = case_path {
            let net = load_csv_zip(&path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            grid.inner.world_mut().insert_resource(PPNetwork(net));
            grid.init_pf();
        } else {
            grid.sync_next_bus_id();
        }
        Ok(grid)
    }

    /// Load a network from a live Python `pandapowerNet` object (from the Python `pandapower` library).
    fn from_pp_net(slf: Py<Self>, py: Python<'_>, net: Bound<'_, PyAny>) -> PyResult<()> {
        let mut rust_net = crate::io::pandapower::Network::default();
        rust_net.from_pp_net(net)?;
        let mut grid_py = slf.borrow_mut(py);
        // Loading a network replaces the grid: clear any previously spawned
        // entities so buses/branches are not duplicated. Existing handles
        // become invalid after this call.
        grid_py.clear_grid_entities();
        grid_py
            .inner
            .world_mut()
            .insert_resource(PPNetwork(rust_net));
        grid_py.init_pf();
        Ok(())
    }

    /// Load a `rustpower.Network` object (the Rust-native parsed grid representation) into the grid.
    ///
    /// This clears all existing entities, re-ingests the network components,
    /// and rebuilds the grid. Existing element handles become invalid.
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
        // This synchronous rebuild supersedes any pending rebuild request;
        // leaving the event in the bus would trigger a redundant second
        // rebuild inside the next update (wiping e.g. a user-set v_init).
        if let Some(mut msgs) = self
            .inner
            .world_mut()
            .get_resource_mut::<bevy_ecs::message::Messages<
                crate::basic::ecs::powerflow::structure_update::FullRebuildEvent,
            >>()
        {
            msgs.clear();
        }
        self.sync_next_bus_id();
    }

    /// Run the power flow. Fully event-driven: pending FullRebuildEvents
    /// (editor commits, in_service toggles) make structure_update run the
    /// PFInit schedule inside this same update; parameter changes flow
    /// through the ParamDiff bus. Raises only on validation errors (empty
    /// grid, no slack); divergence is a legal result reported through the
    /// (falsy) SolveReport.
    ///
    /// v_init: optional complex voltage start vector (p.u.), indexed by bus
    /// id. Pure warm start — PV/slack setpoints are re-pinned afterwards, so
    /// it never alters the physics. The bus-id → solver-ordering permutation
    /// is applied internally.
    #[pyo3(signature = (v_init=None, max_iter=None, tol=None))]
    fn solve(
        &mut self,
        py: Python<'_>,
        v_init: Option<Bound<'_, numpy::PyArray1<num_complex::Complex64>>>,
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> PyResult<SolveReport> {
        use crate::basic::ecs::powerflow::structure_update::{
            FullRebuildEvent, LastStructureAction,
        };
        let t0 = std::time::Instant::now();

        let mut sync_full = false;
        if let Some(v) = v_init {
            // A rebuild pending inside run_pf() would zero VBusPu AFTER we
            // write the start vector; perform it synchronously first
            // (init_pf also clears the now-superseded rebuild event).
            let rebuild_pending = !self.inner.world().contains_resource::<PowerFlowMat>()
                || self
                    .inner
                    .world()
                    .get_resource::<bevy_ecs::message::Messages<FullRebuildEvent>>()
                    .map(|m| !m.is_empty())
                    .unwrap_or(false);
            if rebuild_pending {
                self.init_pf();
                sync_full = true;
            }
            self.set_v(py, v)?;
        } else if !self.inner.world().contains_resource::<PowerFlowMat>() {
            // First-ever build: nothing has posted a rebuild event yet.
            let _ = self.inner.world_mut().write_message(FullRebuildEvent);
        }

        self.inner.world_mut().insert_resource(
            crate::basic::ecs::powerflow::systems::PowerFlowConfig {
                max_it: max_iter,
                tol: tol,
            },
        );

        self.inner.run_pf();

        let full_rebuild = sync_full
            || self
                .inner
                .world()
                .get_resource::<LastStructureAction>()
                .map(|a| a.full_rebuild)
                .unwrap_or(false);
        if full_rebuild {
            self.sync_next_bus_id();
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
            let n_slack = world
                .query_filtered::<Entity, With<SlackBus>>()
                .iter(world)
                .count();
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
        // Mark dirty for lazy post-processing: actual computation
        // deferred until res_bus / res_line is accessed.
        self.post_process_dirty = converged;
        Ok(SolveReport {
            converged,
            iterations,
            runtime_ms: t0.elapsed().as_secs_f64() * 1e3,
            rebuild: if full_rebuild { "full" } else { "incremental" }.to_string(),
        })
    }

    /// Reset the power flow solver state, clearing result vectors and resetting bus injections.
    fn reset_state(&mut self) {
        self.reset_state_impl();
    }

    /// Enable or disable the Iwamoto optimal multiplier solver dynamically at runtime.
    #[pyo3(signature = (enable=true))]
    fn enable_iwamoto(&mut self, enable: bool) {
        if enable {
            let app = self.inner.app_mut();
            if !app.is_plugin_added::<crate::basic::ecs::plugin::IwamotoPlugin>() {
                app.add_plugins(crate::basic::ecs::plugin::IwamotoPlugin);
            }
            self.inner
                .world_mut()
                .insert_resource(crate::basic::ecs::plugin::CustomSolverActive);
            self.inner
                .world_mut()
                .remove_resource::<crate::basic::ecs::dcpf::DcpfSolverActive>();
        } else {
            self.inner
                .world_mut()
                .remove_resource::<crate::basic::ecs::plugin::CustomSolverActive>();
        }
    }

    /// Enable or disable the DCPF-initialized Newton-Raphson solver dynamically at runtime.
    #[pyo3(signature = (enable=true))]
    fn enable_dcpf_init(&mut self, enable: bool) -> PyResult<()> {
        if enable {
            let app = self.inner.app_mut();
            if !app.is_plugin_added::<crate::basic::ecs::dcpf::DcpfNewtonPfPlugin>() {
                app.add_plugins(crate::basic::ecs::dcpf::DcpfNewtonPfPlugin);
            }
            let world = self.inner.world_mut();
            if !world.contains_resource::<crate::basic::dcpf::DcpfModel>() {
                let model = world
                    .run_system_once(crate::basic::ecs::dcpf::build_dcpf_model)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                world.insert_resource(model);
            }
            world.insert_resource(crate::basic::ecs::dcpf::DcpfSolverActive);
            world.remove_resource::<crate::basic::ecs::plugin::CustomSolverActive>();
        } else {
            self.inner
                .world_mut()
                .remove_resource::<crate::basic::ecs::dcpf::DcpfSolverActive>();
        }
        Ok(())
    }

    /// Compute the DCPF initial voltage vector (unpermuted, in original bus order) as a numpy array.
    fn compute_dcpf_v<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.inner.world_mut();
        if !world.contains_resource::<crate::basic::dcpf::DcpfModel>() {
            let model = world
                .run_system_once(crate::basic::ecs::dcpf::build_dcpf_model)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            world.insert_resource(model);
        }

        let mut dcpf_model = world
            .remove_resource::<crate::basic::dcpf::DcpfModel>()
            .unwrap();
        let mat = world
            .get_resource::<crate::basic::ecs::powerflow::systems::PowerFlowMat>()
            .unwrap();
        let n_active = dcpf_model.n_active;
        let mut theta_ws = vec![0.0; n_active];
        let mut solver = crate::basic::solver::DefaultSolver::default();

        let v_perm_res = crate::basic::dcpf::dcpf_initial_v(
            &mut dcpf_model,
            &mat.s_bus,
            &mat.v_bus_init,
            &mut theta_ws,
            &mut solver,
        );
        let n = mat.v_bus_init.len();
        let from_perm = mat.from_perm.clone();
        world.insert_resource(dcpf_model);

        match v_perm_res {
            Ok(v_perm) => {
                let mut v_orig = vec![num_complex::Complex64::new(0.0, 0.0); n];
                for (new_idx, &orig_idx) in from_perm.iter().enumerate() {
                    v_orig[orig_idx] = v_perm[new_idx];
                }
                Ok(numpy::PyArray1::from_vec(py, v_orig))
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "DCPF solve failed: {e}"
            ))),
        }
    }

    /// Solves DCPF and applies the resulting phase angles to the internal initial voltage vector.
    fn apply_dcpf_init(&mut self) -> PyResult<()> {
        let world = self.inner.world_mut();
        if !world.contains_resource::<crate::basic::dcpf::DcpfModel>() {
            let model = world
                .run_system_once(crate::basic::ecs::dcpf::build_dcpf_model)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            world.insert_resource(model);
        }

        let mut dcpf_model = world
            .remove_resource::<crate::basic::dcpf::DcpfModel>()
            .unwrap();
        let mut mat = world
            .get_resource_mut::<crate::basic::ecs::powerflow::systems::PowerFlowMat>()
            .unwrap();
        let n_active = dcpf_model.n_active;
        let mut theta_ws = vec![0.0; n_active];
        let mut solver = crate::basic::solver::DefaultSolver::default();

        let v_perm_res = crate::basic::dcpf::dcpf_initial_v(
            &mut dcpf_model,
            &mat.s_bus,
            &mat.v_bus_init,
            &mut theta_ws,
            &mut solver,
        );

        match v_perm_res {
            Ok(v_perm) => {
                mat.v_bus_init = v_perm;
                drop(mat);
                world.insert_resource(dcpf_model);
                Ok(())
            }
            Err(e) => {
                drop(mat);
                world.insert_resource(dcpf_model);
                Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "DCPF solve failed: {e}"
                )))
            }
        }
    }

    /// Enable or disable Jacobian caching for the power grid solver.
    /// When enabled, it reuses the symbolic factorization across multiple solves.
    #[pyo3(signature = (enable=true))]
    fn enable_cache(&mut self, enable: bool) {
        if enable {
            if !self
                .inner
                .world()
                .contains_resource::<crate::basic::newtonpf::NewtonCache>()
            {
                self.inner
                    .world_mut()
                    .insert_resource(crate::basic::newtonpf::NewtonCache::default());
            }
        } else {
            self.inner
                .world_mut()
                .remove_resource::<crate::basic::newtonpf::NewtonCache>();
        }
    }

    /// Whether the last power flow solve converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner
            .world()
            .get_resource::<PowerFlowResult>()
            .map(|r| r.converged)
            .unwrap_or(false)
    }

    /// The number of iterations taken by the last power flow solve.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner
            .world()
            .get_resource::<PowerFlowResult>()
            .map(|r| r.iterations)
            .unwrap_or(0)
    }

    /// Complex bus voltages (p.u.) of the last solve, in original bus order.
    #[getter]
    fn v<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<num_complex::Complex64>>> {
        let world = self.inner.world();
        let res = world.get_resource::<PowerFlowResult>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No power flow result: call solve() first",
            )
        })?;
        Ok(res.v.as_slice().to_vec().into_pyarray(py))
    }

    /// Return the original (unpermuted) Y-bus as a scipy.sparse.csc_matrix.
    #[getter]
    fn y_bus<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world();
        if let Some(orig) = world.get_resource::<crate::basic::ecs::powerflow::systems::OriginalYBus>() {
            let indptr = orig.0.col_offsets().to_vec().into_pyarray(py);
            let indices = orig.0.row_indices().to_vec().into_pyarray(py);
            let data = orig.0.values().to_vec().into_pyarray(py);
            let sp = py.import("scipy.sparse")?;
            let shape = (orig.0.nrows(), orig.0.ncols());
            return sp.call_method1("csc_matrix", ((data, indices, indptr), shape));
        }
        if let Some(mat) = world.get_resource::<PowerFlowMat>() {
            let orig = crate::basic::sparse::utils::permute_csc_to_csc_local_sort(
                &mat.y_bus,
                &mat.to_perm,
                &mat.from_perm,
            );
            let indptr = orig.col_offsets().to_vec().into_pyarray(py);
            let indices = orig.row_indices().to_vec().into_pyarray(py);
            let data = orig.values().to_vec().into_pyarray(py);
            let sp = py.import("scipy.sparse")?;
            let shape = (orig.nrows(), orig.ncols());
            return sp.call_method1("csc_matrix", ((data, indices, indptr), shape));
        }
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Ybus not initialized: solve() or init_pf() must be called first",
        ))
    }

    /// Set the voltage start vector (p.u. complex), indexed by bus id.
    /// Writes VBusPu on each bus, re-pins PV/slack setpoints (v_inj) so this
    /// is a pure warm start, and fires VoltageChangeEvent; the sync into the
    /// solver vector applies the bus-id → solver-ordering permutation
    /// internally (reorder_index in vbus_pu_update).
    #[setter]
    fn set_v(
        &mut self,
        _py: Python<'_>,
        v: Bound<'_, numpy::PyArray1<num_complex::Complex64>>,
    ) -> PyResult<()> {
        if !self.inner.world().contains_resource::<PowerFlowMat>() {
            self.init_pf();
        }
        let v_ro = v.readonly();
        let v_arr = v_ro.as_slice()?;
        let world = self.inner.world_mut();
        let n = world.resource::<PowerFlowMat>().v_bus_init.len();
        if v_arr.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "v has length {}, expected {} (one entry per bus id)",
                v_arr.len(),
                n
            )));
        }
        let mut bus_q = world.query::<(&BusID, &mut VBusPu)>();
        for (id, mut bus_v) in bus_q.iter_mut(world) {
            let idx = id.0 as usize;
            if idx < v_arr.len() {
                bus_v.0 = v_arr[idx];
            }
        }
        // Re-pin PV/slack magnitude and slack angle targets: v overrides the
        // start point, never the setpoints.
        let _ = world.run_system_once(crate::basic::ecs::powerflow::init::v_inj);
        let _ =
            world.write_message(crate::basic::ecs::powerflow::structure_update::VoltageChangeEvent);
        Ok(())
    }

    /// Get the number of buses in the grid.
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
        GridEditor {
            parent: slf.clone_ref(py),
            created: Vec::new(),
            start_next_bus_id,
        }
    }

    /// Build a PowerGrid directly from a live Python `pandapowerNet` object (from the Python `pandapower` library).
    ///
    /// The Python pandapower data model is discarded after ingestion.
    #[classmethod]
    #[pyo3(signature = (net, **kwargs))]
    fn from_pandapower(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        net: Bound<'_, PyAny>,
        kwargs: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        let grid = Self::new(None, false, kwargs)?;
        let slf = Py::new(py, grid)?;
        Self::from_pp_net(slf.clone_ref(py), py, net)?;
        Ok(slf)
    }

    /// Set the base frequency (Hz) and system base MVA.
    #[pyo3(signature = (f_hz=50.0, sn_mva=100.0))]
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        GridFactory::set_base(&mut self.inner, f_hz, sn_mva);
    }

    /// Retrieve a BusHandle by its ID. Returns None if not found.
    fn bus(slf: Py<Self>, py: Python<'_>, id: i64) -> Option<BusHandle> {
        let grid = slf.borrow(py);
        let world = grid.inner.world();
        let lookup = world.get_resource::<NodeLookup>()?;
        let entity = lookup.get_entity(id)?;
        Some(BusHandle::new(entity, slf.clone_ref(py)))
    }

    /// First load matching the filters, or None. Query-backed: no shadow index.
    #[pyo3(signature = (bus=None, name=None))]
    fn load(
        slf: Py<Self>,
        py: Python<'_>,
        bus: Option<i64>,
        name: Option<String>,
    ) -> Option<LoadHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world
            .query_filtered::<(Entity, &TargetBus, Option<&bevy_ecs::name::Name>), With<LoadCfg>>();
        for (e, tb, n) in q.iter(world) {
            if let Some(b) = bus {
                if tb.0 != b {
                    continue;
                }
            }
            if let Some(ref nm) = name {
                if n.map(|x| x.as_str()) != Some(nm.as_str()) {
                    continue;
                }
            }
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
            if let Some(b) = bus {
                if tb.0 != b {
                    continue;
                }
            }
            out.push(LoadHandle::new(e, slf.clone_ref(py)));
        }
        out
    }

    /// First PV generator matching the filters (slack excluded), or None.
    #[pyo3(signature = (bus=None, name=None))]
    fn r#gen(
        slf: Py<Self>,
        py: Python<'_>,
        bus: Option<i64>,
        name: Option<String>,
    ) -> Option<GenHandle> {
        let mut grid = slf.borrow_mut(py);
        let world = grid.inner.world_mut();
        let mut q = world.query_filtered::<(
            Entity,
            &TargetBus,
            Option<&bevy_ecs::name::Name>,
            Has<Slack>,
        ), With<GeneratorCfg>>();
        for (e, tb, n, is_slack) in q.iter(world) {
            if is_slack {
                continue;
            }
            if let Some(b) = bus {
                if tb.0 != b {
                    continue;
                }
            }
            if let Some(ref nm) = name {
                if n.map(|x| x.as_str()) != Some(nm.as_str()) {
                    continue;
                }
            }
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
        let n_trafo = world
            .query::<&crate::basic::ecs::elements::TransformerDevice>()
            .iter(world)
            .count();
        let n_load = world.query::<&LoadCfg>().iter(world).count();
        let mut n_gen = 0usize;
        let mut n_ext = 0usize;
        let mut q = world.query_filtered::<Has<Slack>, With<GeneratorCfg>>();
        for is_slack in q.iter(world) {
            if is_slack {
                n_ext += 1;
            } else {
                n_gen += 1;
            }
        }
        let n_shunt = world.query::<&ShuntDevice>().iter(world).count();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(
            "element",
            vec!["bus", "line", "trafo", "load", "gen", "ext_grid", "shunt"],
        )?;
        dict.set_item(
            "count",
            vec![n_bus, n_line, n_trafo, n_load, n_gen, n_ext, n_shunt],
        )?;
        py.import("pandas")?.call_method1("DataFrame", (dict,))
    }

    /// Return a pandas DataFrame showing all load parameters in the case.
    fn display_case_loads<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let mut buses = Vec::new();
        let mut ps = Vec::new();
        let mut qs = Vec::new();
        let mut names = Vec::new();
        let mut query = world.query_filtered::<(
            &TargetBus,
            &TargetPMW,
            &TargetQMVar,
            Option<&bevy_ecs::name::Name>,
        ), With<LoadCfg>>();
        query.iter(world).for_each(|(bus, p, q, name)| {
            buses.push(bus.0);
            // Targets store injections (consumption is negative); display as consumption
            ps.push(-p.0);
            qs.push(-q.0);
            names.push(name.map(|n| n.as_str().to_string()).unwrap_or_default());
        });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus", buses.into_pyarray(py))?;
        dict.set_item("p_mw", ps.into_pyarray(py))?;
        dict.set_item("q_mvar", qs.into_pyarray(py))?;
        dict.set_item("name", names)?;
        py.import("pandas")?.call_method1("DataFrame", (dict,))
    }

    /// Get the number of lines in the grid.
    #[getter]
    fn n_line(&mut self) -> usize {
        let world = self.inner.world_mut();
        world
            .query::<&crate::basic::ecs::elements::Line>()
            .iter(world)
            .count()
    }

    /// Return the raw bus results as a DataFrame.
    fn get_bus_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.ensure_post_processed();
        self.get_bus_results_impl(py)
    }

    /// Return the raw line results as a DataFrame.
    fn get_line_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.ensure_post_processed();
        self.get_line_results_impl(py)
    }

    /// Return the raw transformer results as a DataFrame.
    fn get_trafo_results<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.ensure_post_processed();
        self.get_trafo_results_impl(py)
    }

    /// Return the raw bus parameters as a dictionary.
    fn get_bus_params<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let res = self.get_bus_params_impl(py)?;
        py.import("pandas")?.call_method1("DataFrame", (res,))
    }

    /// Bus results of the last solve as a DataFrame (pandapower's res_bus).
    /// Lazy: triggers post-processing on first access after solve().
    #[getter]
    fn res_bus<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_bus_results(py)
    }

    /// Line results of the last solve as a DataFrame (pandapower's res_line).
    /// Lazy: triggers post-processing on first access after solve().
    #[getter]
    fn res_line<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_line_results(py)
    }

    /// Transformer results of the last solve as a DataFrame (pandapower's res_trafo).
    /// Lazy: triggers post-processing on first access after solve().
    #[getter]
    fn res_trafo<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_trafo_results(py)
    }

    /// Return a pandas DataFrame showing all bus parameters in the case.
    fn display_case_buses<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.get_bus_params(py)
    }

    /// Explicitly run post-processing (extract bus + line results from the
    /// solver output).  Usually not needed — `res_bus` / `res_line` trigger
    /// this lazily — but exposed for scripts that need the ECS components
    /// populated before an archive snapshot.
    fn post_process(&mut self) {
        self.ensure_post_processed();
    }

    /// Serialize the case-file ECS state (grid topology, parameters) to an
    /// in-memory ZIP archive of Parquet files.  Returns raw `bytes` that can
    /// be opened with `zipfile.ZipFile(io.BytesIO(data))` or fed to DuckDB
    /// `read_parquet`.
    #[cfg(feature = "arrow")]
    fn get_parquet_case<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        use bevy_archive::binary_archive::WorldArrowSnapshot;

        let world = self.inner.world();
        let reg = world.get_resource::<ArchiveSnapshotRes>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ArchiveSnapshotRes not available — archive feature may not be initialized",
            )
        })?;
        let snapshot =
            WorldArrowSnapshot::from_world_reg(world, &reg.0.case_file_reg).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("snapshot error: {}", e))
            })?;
        let zip_bytes = snapshot.to_zip(None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("zip error: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &zip_bytes))
    }

    /// Serialize the output ECS state (bus voltages, line results) to an
    /// in-memory ZIP archive of Parquet files.  Triggers lazy post-processing
    /// if needed.
    #[cfg(feature = "arrow")]
    fn get_parquet_results<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        use bevy_archive::binary_archive::WorldArrowSnapshot;

        self.ensure_post_processed();

        let world = self.inner.world();
        let reg = world.get_resource::<ArchiveSnapshotRes>().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ArchiveSnapshotRes not available — archive feature may not be initialized",
            )
        })?;
        let snapshot =
            WorldArrowSnapshot::from_world_reg(world, &reg.0.output_reg).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("snapshot error: {}", e))
            })?;
        let zip_bytes = snapshot.to_zip(None).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("zip error: {}", e))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &zip_bytes))
    }

    /// Load a case-file ECS state from a ZIP archive of Parquet files (restores
    /// topology and parameters). Clears any existing entities first.
    #[cfg(feature = "arrow")]
    fn load_parquet_case<'py>(
        &mut self,
        _py: Python<'py>,
        zip_bytes: Bound<'py, pyo3::types::PyBytes>,
    ) -> PyResult<()> {
        use crate::io::archive::aurora_format::ArchiveSnapshotRes;
        use bevy_archive::binary_archive::WorldArrowSnapshot;

        self.clear_grid_entities();

        let zip_data = zip_bytes.as_bytes();
        let snapshot = WorldArrowSnapshot::from_zip(zip_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("from_zip error: {}", e))
        })?;

        let world = self.inner.world_mut();
        let registry = {
            let reg = world.get_resource::<ArchiveSnapshotRes>().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "ArchiveSnapshotRes not available — archive feature may not be initialized",
                )
            })?;
            reg.0.case_file_reg.clone()
        };

        snapshot.to_world_reg(world, &registry).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("to_world error: {}", e))
        })?;

        self.sync_next_bus_id();
        self.init_pf();

        Ok(())
    }
}

impl PowerGrid {
    /// Run ECS post-processing (bus + line result extraction) only if the
    /// dirty flag is set.  Called lazily from `res_bus`, `res_line`, and any
    /// other result getter.
    pub(crate) fn ensure_post_processed(&mut self) {
        if self.post_process_dirty {
            self.inner.post_process();
            self.post_process_dirty = false;
        }
    }

    fn bus_exists_in_world(&mut self, bus_id: i64) -> bool {
        let world = self.inner.world_mut();
        if let Some(lookup) = world.get_resource::<NodeLookup>() {
            lookup.get_entity(bus_id).is_some()
        } else {
            bus_id < self.next_bus_id
        }
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
        world
            .iter_entities()
            .filter_map(|e| e.get::<BusID>())
            .for_each(|id| {
                if id.0 > max_id {
                    max_id = id.0;
                }
            });
        self.next_bus_id = max_id + 1;
    }

    fn reset_state_impl(&mut self) {
        let world = self.inner.world_mut();
        let mut query = world.query_filtered::<&mut SBusInjPu, With<BusID>>();
        for mut s in query.iter_mut(world) {
            s.0 = num_complex::Complex64::new(0.0, 0.0);
        }
        world.remove_resource::<PowerFlowResult>();
        self.post_process_dirty = false;
    }

    fn get_bus_params_impl<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let world = self.inner.world_mut();
        let mut bus_ids = Vec::new();
        let mut names = Vec::new();
        let mut vn_kv = Vec::new();
        let mut vm_min = Vec::new();
        let mut vm_max = Vec::new();
        let mut types = Vec::new();
        let mut query = world.query::<(
            &BusID,
            Option<&bevy_ecs::name::Name>,
            Option<&VNominal>,
            Option<&VmLimit<PerUnit>>,
            Has<SlackBus>,
            Has<PVBus>,
        )>();
        query
            .iter(world)
            .for_each(|(id, name, vnom, vlim, is_slack, is_pv)| {
                bus_ids.push(id.0);
                names.push(name.map(|n| n.as_str().to_string()).unwrap_or_default());
                vn_kv.push(vnom.map(|v| v.0.0).unwrap_or(0.0));
                let (min, max) = vlim.map(|l| (l.min(), l.max())).unwrap_or((0.9, 1.1));
                vm_min.push(min);
                vm_max.push(max);
                types.push(if is_slack {
                    "Slack"
                } else if is_pv {
                    "PV"
                } else {
                    "PQ"
                });
            });
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?;
        dict.set_item("name", names)?;
        dict.set_item("type", types)?;
        dict.set_item("vn_kv", vn_kv.into_pyarray(py))?;
        dict.set_item("vm_min_pu", vm_min.into_pyarray(py))?;
        dict.set_item("vm_max_pu", vm_max.into_pyarray(py))?;
        Ok(dict)
    }

    fn get_bus_results_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let mut query = world.query::<(&BusID, &VBusResult, &SBusResult)>();
        let count = query.iter(world).count();

        let mut bus_ids = Vec::with_capacity(count);
        let mut v_complex = Vec::with_capacity(count);
        let mut vms = Vec::with_capacity(count);
        let mut vas = Vec::with_capacity(count);
        let mut ps = Vec::with_capacity(count);
        let mut qs = Vec::with_capacity(count);

        query.iter(world).for_each(|(id, v, s)| {
            bus_ids.push(id.0);
            v_complex.push(v.0);
            vms.push(v.0.norm());
            vas.push(v.0.arg().to_degrees());
            ps.push(s.0.re);
            qs.push(s.0.im);
        });

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("bus_id", bus_ids.into_pyarray(py))?;
        dict.set_item("v_pu", v_complex.into_pyarray(py))?;
        dict.set_item("vm_pu", vms.into_pyarray(py))?;
        dict.set_item("va_degree", vas.into_pyarray(py))?;
        dict.set_item("p_mw", ps.into_pyarray(py))?;
        dict.set_item("q_mvar", qs.into_pyarray(py))?;
        py.import("pandas")?.call_method1("DataFrame", (dict,))
    }

    fn get_line_results_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let mut query = world.query::<(&FromBus, &ToBus, &LineResultData)>();
        let count = query.iter(world).count();

        let mut from_bus = Vec::with_capacity(count);
        let mut to_bus = Vec::with_capacity(count);
        let mut data = Vec::with_capacity(count * 14);

        query.iter(world).for_each(|(f, t, d)| {
            from_bus.push(f.0);
            to_bus.push(t.0);
            data.extend_from_slice(d.as_slice());
        });

        let pandas = py.import("pandas")?;
        let py_data_2d = data
            .into_pyarray(py)
            .call_method1("reshape", ((count, 14),))?;
        const LINE_COLS: &[&str] = &[
            "p_from_mw",
            "q_from_mvar",
            "p_to_mw",
            "q_to_mvar",
            "pl_mw",
            "ql_mvar",
            "i_from_ka",
            "i_to_ka",
            "i_ka",
            "vm_from_pu",
            "va_from_degree",
            "vm_to_pu",
            "va_to_degree",
            "loading_percent",
        ];
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("columns", LINE_COLS)?;
        let df = pandas.call_method("DataFrame", (py_data_2d,), Some(&kwargs))?;
        df.call_method1("insert", (0, "to_bus", to_bus.into_pyarray(py)))?;
        df.call_method1("insert", (0, "from_bus", from_bus.into_pyarray(py)))?;
        Ok(df)
    }

    fn get_trafo_results_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let world = self.inner.world_mut();
        let mut query = world.query::<(&FromBus, &ToBus, &TrafoResultData)>();
        let count = query.iter(world).count();

        let mut hv_bus = Vec::with_capacity(count);
        let mut lv_bus = Vec::with_capacity(count);
        let mut data = Vec::with_capacity(count * 13);

        query.iter(world).for_each(|(f, t, d)| {
            hv_bus.push(f.0);
            lv_bus.push(t.0);
            data.extend_from_slice(d.as_slice());
        });

        let pandas = py.import("pandas")?;
        let py_data_2d = data
            .into_pyarray(py)
            .call_method1("reshape", ((count, 13),))?;
        const TRAFO_COLS: &[&str] = &[
            "p_hv_mw",
            "q_hv_mvar",
            "p_lv_mw",
            "q_lv_mvar",
            "pl_mw",
            "ql_mvar",
            "i_hv_ka",
            "i_lv_ka",
            "vm_hv_pu",
            "va_hv_degree",
            "vm_lv_pu",
            "va_lv_degree",
            "loading_percent",
        ];
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("columns", TRAFO_COLS)?;
        let df = pandas.call_method("DataFrame", (py_data_2d,), Some(&kwargs))?;
        df.call_method1("insert", (0, "lv_bus", lv_bus.into_pyarray(py)))?;
        df.call_method1("insert", (0, "hv_bus", hv_bus.into_pyarray(py)))?;
        Ok(df)
    }
}
