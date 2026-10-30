//! Python ownership and column conversion around the native N-1 study.
use super::{
    grid::PowerGrid,
    handles::{LineHandle, TrafoHandle},
};
use crate::basic::ecs::network::DataOps;
use crate::n1::*;
use bevy_ecs::prelude::*;
use numpy::IntoPyArray;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::time::Instant;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
fn initialization(value: &str) -> PyResult<N1Initialization> {
    match value {
        "base-ac" => Ok(N1Initialization::BaseAc),
        "dc" => Ok(N1Initialization::DcPerCase),
        _ => Err(value_error("initialization must be 'base-ac' or 'dc'")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Auto,
    Cpu,
    Cuda,
    Hybrid,
}
impl Backend {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(value_error(
                "backend must be 'auto', 'cpu', 'cuda' or 'hybrid'",
            )),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Hybrid => "hybrid",
        }
    }
}

/// A reusable selection tied to its source PowerGrid.
#[pyclass(name = "N1Faults", unsendable)]
pub struct Faults {
    owner: Py<PowerGrid>,
    entities: Vec<Entity>,
}
#[pymethods]
impl Faults {
    fn __len__(&self) -> usize {
        self.entities.len()
    }
    #[getter]
    fn entity_ids<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<u64>> {
        self.entities
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
}

fn resolve_faults(owner: &Py<PowerGrid>, faults: &Bound<'_, PyAny>) -> PyResult<N1FaultSet> {
    if let Ok(faults) = faults.extract::<PyRef<'_, Faults>>() {
        if faults.owner.as_ptr() != owner.as_ptr() {
            return Err(value_error("faults belong to another grid"));
        }
        return Ok(N1FaultSet::out_of_service(faults.entities.iter().copied()));
    }
    let mut entities = Vec::new();
    for item in faults.try_iter()? {
        let item = item?;
        let (entity, source) = if let Ok(h) = item.extract::<PyRef<'_, LineHandle>>() {
            (h.entity(), h.grid.as_ptr())
        } else if let Ok(h) = item.extract::<PyRef<'_, TrafoHandle>>() {
            (h.entity(), h.grid.as_ptr())
        } else {
            return Err(PyTypeError::new_err(
                "faults require line or transformer handles",
            ));
        };
        if source != owner.as_ptr() {
            return Err(value_error("fault handle belongs to another grid"));
        }
        entities.push(entity);
    }
    Ok(N1FaultSet::out_of_service(entities))
}

#[pyclass(name = "N1Study", unsendable)]
pub struct Study {
    owner: Py<PowerGrid>,
    prepared: PreparedN1,
    faults: N1FaultSet,
    preparation_seconds: f64,
}

// The prepared snapshot stays borrowed until all native scoped workers join.
struct ReleasedStudy<'a>(&'a PreparedN1);
impl ReleasedStudy<'_> {
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn run(
        self,
        faults: &N1FaultSet,
        options: N1RunOptions,
        detailed: bool,
        backend: Backend,
        cuda_batch_size: usize,
        cuda_streams: usize,
        cuda_queue_batches: usize,
    ) -> Result<(N1StudyReport, Vec<N1Event>, &'static str), String> {
        let study = self.0.study().map_err(|e| e.to_string())?;
        let base_ac = matches!(
            options.initialization.unwrap_or(self.0.initialization()),
            N1Initialization::BaseAc
        );
        let selected = match backend {
            Backend::Auto => {
                #[cfg(feature = "experimental_n1_cuda")]
                {
                    if base_ac && crate::n1::cuda::cuda_status().is_ok() {
                        Backend::Cuda
                    } else {
                        Backend::Cpu
                    }
                }
                #[cfg(not(feature = "experimental_n1_cuda"))]
                {
                    Backend::Cpu
                }
            }
            explicit => explicit,
        };
        let mut events = Vec::new();
        let collect = |batch: N1EventBatch| events.extend(batch.events);
        let reporting = if detailed {
            N1Reporting::Detailed
        } else {
            N1Reporting::StatusOnly
        };
        let report = match selected {
            Backend::Cpu => if detailed {
                study.details(faults, options, collect)
            } else {
                study.run(faults, options, collect)
            }
            .map_err(|e| e.to_string())?,
            Backend::Cuda => {
                #[cfg(feature = "experimental_n1_cuda")]
                {
                    crate::n1::cuda::cuda_status()?;
                    crate::n1::cuda::run_n1_study_cuda(
                        &study,
                        faults,
                        crate::n1::cuda::CudaN1Config {
                            batch_size: cuda_batch_size,
                            streams: cuda_streams,
                            max_iter: options.config.max_iter,
                            tolerance: options.config.tolerance,
                            reporting,
                        },
                        collect,
                    )
                    .map_err(|e| e.to_string())?
                }
                #[cfg(not(feature = "experimental_n1_cuda"))]
                {
                    return Err("CUDA N-1 is not included in this build".into());
                }
            }
            Backend::Hybrid => {
                #[cfg(feature = "experimental_n1_cuda")]
                {
                    crate::n1::cuda::cuda_status()?;
                    crate::n1::cuda::run_n1_study_hybrid(
                        &study,
                        faults,
                        crate::n1::cuda::HybridN1Config {
                            cpu_workers: options.config.workers,
                            cpu_batch_size: options.config.batch_size,
                            cuda_batch_size,
                            cuda_streams,
                            cuda_queue_batches,
                            max_iter: options.config.max_iter,
                            tolerance: options.config.tolerance,
                            reporting,
                        },
                        collect,
                    )
                    .map_err(|e| e.to_string())?
                }
                #[cfg(not(feature = "experimental_n1_cuda"))]
                {
                    return Err("hybrid N-1 requires a CUDA-enabled build".into());
                }
            }
            Backend::Auto => unreachable!("auto backend was resolved before execution"),
        };
        Ok((report, events, selected.name()))
    }
}
impl Study {
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        py: Python<'_>,
        faults: Option<Bound<'_, PyAny>>,
        workers: usize,
        batch_size: usize,
        mode: Option<&str>,
        max_iter: usize,
        tolerance: f64,
        detailed: bool,
        backend: &str,
        cuda_batch_size: usize,
        cuda_streams: usize,
        cuda_queue_batches: usize,
    ) -> PyResult<Report> {
        let backend = Backend::parse(backend)?;
        let faults = match faults {
            Some(faults) => resolve_faults(&self.owner, &faults)?,
            None => self.faults.clone(),
        };
        let options = N1RunOptions {
            config: N1Config {
                workers,
                batch_size,
                max_iter,
                tolerance,
                ..Default::default()
            },
            initialization: mode.map(initialization).transpose()?,
            ..Default::default()
        };
        let started = Instant::now();
        let released = ReleasedStudy(&self.prepared);
        let (report, events, backend) = py
            .allow_threads(move || {
                released.run(
                    &faults,
                    options,
                    detailed,
                    backend,
                    cuda_batch_size,
                    cuda_streams,
                    cuda_queue_batches,
                )
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Report {
            owner: self.owner.clone_ref(py),
            report,
            events,
            backend,
            preparation_seconds: self.preparation_seconds,
            execution_seconds: started.elapsed().as_secs_f64(),
        })
    }
}
#[pymethods]
impl Study {
    #[pyo3(signature = (faults=None, *, workers=1, batch_size=8, initialization=None, max_iter=30, tolerance=1e-10, backend="auto", cuda_batch_size=64, cuda_streams=1, cuda_queue_batches=2))]
    fn run(
        &self,
        py: Python<'_>,
        faults: Option<Bound<'_, PyAny>>,
        workers: usize,
        batch_size: usize,
        initialization: Option<&str>,
        max_iter: usize,
        tolerance: f64,
        backend: &str,
        cuda_batch_size: usize,
        cuda_streams: usize,
        cuda_queue_batches: usize,
    ) -> PyResult<Report> {
        self.execute(
            py,
            faults,
            workers,
            batch_size,
            initialization,
            max_iter,
            tolerance,
            false,
            backend,
            cuda_batch_size,
            cuda_streams,
            cuda_queue_batches,
        )
    }

    #[pyo3(signature = (faults=None, *, workers=1, batch_size=8, initialization=None, max_iter=30, tolerance=1e-10, backend="auto", cuda_batch_size=64, cuda_streams=1, cuda_queue_batches=2))]
    fn details(
        &self,
        py: Python<'_>,
        faults: Option<Bound<'_, PyAny>>,
        workers: usize,
        batch_size: usize,
        initialization: Option<&str>,
        max_iter: usize,
        tolerance: f64,
        backend: &str,
        cuda_batch_size: usize,
        cuda_streams: usize,
        cuda_queue_batches: usize,
    ) -> PyResult<Report> {
        self.execute(
            py,
            faults,
            workers,
            batch_size,
            initialization,
            max_iter,
            tolerance,
            true,
            backend,
            cuda_batch_size,
            cuda_streams,
            cuda_queue_batches,
        )
    }

    #[getter]
    fn faults(&self, py: Python<'_>) -> Faults {
        Faults {
            owner: self.owner.clone_ref(py),
            entities: self.faults.entities().to_vec(),
        }
    }
    #[getter]
    fn preparation_seconds(&self) -> f64 {
        self.preparation_seconds
    }
}

#[pyclass(name = "N1Report", unsendable)]
pub struct Report {
    owner: Py<PowerGrid>,
    report: N1StudyReport,
    events: Vec<N1Event>,
    backend: &'static str,
    preparation_seconds: f64,
    execution_seconds: f64,
}
#[pymethods]
impl Report {
    #[getter]
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let s = &self.report.solve;
        for (name, n) in [
            ("processed", self.report.processed()),
            ("passed", self.report.passed()),
            ("violated", self.report.violated()),
            ("numerical_failed", self.report.numerical_failed()),
            ("islanded", self.report.islanded()),
            ("numerically_solved", s.processed),
            ("total_iterations", s.total_iterations),
            ("attempted_iterations", s.attempted_iterations),
            ("workers_started", s.workers_started),
            ("emitted_cases", s.emitted_cases),
            ("factor_calls", s.linear.factor_calls),
            ("refactor_calls", s.linear.refactor_calls),
        ] {
            d.set_item(name, n)?;
        }
        d.set_item("backend", self.backend)?;
        d.set_item("preparation_seconds", self.preparation_seconds)?;
        d.set_item("execution_seconds", self.execution_seconds)?;
        d.set_item("ac_seconds", s.ac_time.as_secs_f64())?;
        d.set_item("dc_seconds", s.dc_time.as_secs_f64())?;
        Ok(d)
    }
    #[getter]
    fn numerical_failures(&self, py: Python<'_>) -> Faults {
        Faults {
            owner: self.owner.clone_ref(py),
            entities: self
                .events
                .iter()
                .filter(|e| e.outcome == N1Outcome::NumericalFailure)
                .map(|e| e.outage_entity)
                .collect(),
        }
    }
    /// Constraint or numerical rejections; islands have a separate selection.
    #[getter]
    fn rejected_faults(&self, py: Python<'_>) -> Faults {
        Faults {
            owner: self.owner.clone_ref(py),
            entities: self
                .events
                .iter()
                .filter(|e| e.outcome != N1Outcome::Passed)
                .map(|e| e.outage_entity)
                .collect(),
        }
    }
    #[getter]
    fn island_faults(&self, py: Python<'_>) -> Faults {
        Faults {
            owner: self.owner.clone_ref(py),
            entities: self
                .report
                .islands
                .iter()
                .map(|e| e.outage_entity)
                .collect(),
        }
    }
    /// Numeric columns are independent NumPy buffers, constructed on demand.
    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        macro_rules! column {
            ($name:literal, $value:expr) => {
                d.set_item(
                    $name,
                    self.events
                        .iter()
                        .map($value)
                        .collect::<Vec<_>>()
                        .into_pyarray(py),
                )?;
            };
        }
        column!("case_id", |e: &N1Event| e.case_id as u64);
        column!("outage_index", |e: &N1Event| e.outage_index as u64);
        column!("outage_id", |e: &N1Event| e.outage.id);
        column!("outage_entity", |e: &N1Event| e.outage_entity.to_bits());
        column!("iterations", |e: &N1Event| e.iterations as u64);
        column!("residual", |e: &N1Event| e.residual);
        column!("hit_count", |e: &N1Event| e.hits.len() as u64);
        d.set_item(
            "outage_kind",
            self.events
                .iter()
                .map(|e| format!("{:?}", e.outage.kind))
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "outcome",
            self.events
                .iter()
                .map(|e| format!("{:?}", e.outcome))
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "stage",
            self.events
                .iter()
                .map(|e| format!("{:?}", e.stage))
                .collect::<Vec<_>>(),
        )?;
        d.set_item(
            "error",
            self.events
                .iter()
                .map(|e| e.error.as_deref())
                .collect::<Vec<_>>(),
        )?;
        Ok(d)
    }
    /// Flattened hits link to events by event_row and to branch_results by result_index.
    #[getter]
    fn hits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let (mut rows, mut rules, mut sources, mut kinds, mut ids, mut values, mut result_indices) = (
            Vec::<u64>::new(),
            Vec::<u64>::new(),
            Vec::<u64>::new(),
            Vec::<String>::new(),
            Vec::<i64>::new(),
            Vec::<f64>::new(),
            Vec::<i64>::new(),
        );
        let mut offset = 0;
        for (row, e) in self.events.iter().enumerate() {
            for hit in e.hits.iter() {
                rows.push(row as u64);
                rules.push(*hit.rule_id);
                sources.push(hit.source_entity.to_bits());
                let (kind, id) = match *hit.target {
                    WatchTarget::Branch { identity, .. } => {
                        (format!("{:?}", identity.kind), identity.id)
                    }
                    WatchTarget::Bus { id, .. } => ("Bus".into(), id),
                };
                kinds.push(kind);
                ids.push(id);
                values.push(hit.value.unwrap_or(f64::NAN));
                result_indices.push(hit.result_index.map_or(-1, |i| (offset + i) as i64));
            }
            offset += e.hits.branch_results().len();
        }
        d.set_item("event_row", rows.into_pyarray(py))?;
        d.set_item("rule_id", rules.into_pyarray(py))?;
        d.set_item("source_entity", sources.into_pyarray(py))?;
        d.set_item("target_kind", kinds)?;
        d.set_item("target_id", ids.into_pyarray(py))?;
        d.set_item("value", values.into_pyarray(py))?;
        d.set_item("result_index", result_indices.into_pyarray(py))?;
        Ok(d)
    }
    #[getter]
    fn branch_results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let rows: Vec<_> = self
            .events
            .iter()
            .enumerate()
            .flat_map(|(i, e)| std::iter::repeat_n(i as u64, e.hits.branch_results().len()))
            .collect();
        d.set_item("event_row", rows.into_pyarray(py))?;
        macro_rules! column { ($($field:ident),+) => { $(
            d.set_item(stringify!($field),self.events.iter().flat_map(|e|e.hits.branch_results().iter().map(|r|r.$field)).collect::<Vec<_>>().into_pyarray(py))?;
        )+ }; }
        column!(
            p_from_mw,
            q_from_mvar,
            p_to_mw,
            q_to_mvar,
            pl_mw,
            ql_mvar,
            i_from_ka,
            i_to_ka,
            i_ka,
            vm_from_pu,
            va_from_degree,
            vm_to_pu,
            va_to_degree,
            loading_percent
        );
        Ok(d)
    }
}

pub fn prepare(
    py: Python<'_>,
    owner: Py<PowerGrid>,
    faults: Option<Bound<'_, PyAny>>,
    mode: &str,
    voltage_limits: bool,
    line_loading_limit: Option<f64>,
) -> PyResult<Study> {
    use crate::basic::ecs::{
        elements::{
            bus::{BusID, VmLimit},
            line::LineID,
            units::PerUnit,
        },
        powerflow::{prepare::prepare_solver_inputs, qlim::QLimPlugin},
    };
    let initialization = initialization(mode)?;
    if line_loading_limit.is_some_and(|v| !v.is_finite() || v < 0.) {
        return Err(value_error(
            "line_loading_limit must be finite and nonnegative",
        ));
    }
    let selection = faults
        .as_ref()
        .map(|f| resolve_faults(&owner, f))
        .transpose()?;
    let started = Instant::now();
    let mut grid = owner.borrow_mut(py);
    if grid.inner.app().is_plugin_added::<QLimPlugin>() {
        return Err(value_error(
            "N-1 uses a fixed bus partition; create this grid with qlim=False",
        ));
    }
    prepare_solver_inputs(grid.inner.world_mut());
    let world = grid.inner.world_mut();
    let mut restore = Vec::new();
    let targets: Vec<_> = world
        .query::<(
            Entity,
            Option<&LineID>,
            Option<&BusID>,
            Option<&VmLimit<PerUnit>>,
        )>()
        .iter(world)
        .filter_map(|(e, line, bus, limit)| {
            let rule = if line.is_some() {
                line_loading_limit.map(WatchCondition::BranchLoadingAbove)
            } else if voltage_limits && bus.is_some() && limit.is_some() {
                Some(WatchCondition::BusVoltageOutsideLimits)
            } else {
                None
            };
            rule.map(|r| (e, r))
        })
        .collect();
    for (entity, condition) in targets {
        let old = world.get::<N1Watch>(entity).cloned();
        let mut watch = old.clone().unwrap_or_default();
        watch.rules.push(WatchRule {
            rule_id: if matches!(condition, WatchCondition::BusVoltageOutsideLimits) {
                2
            } else {
                1
            },
            condition,
        });
        world.entity_mut(entity).insert(watch);
        restore.push((entity, old));
    }
    let prepared = prepare_n1_with_initialization(world, initialization);
    for (entity, old) in restore {
        if let Some(old) = old {
            world.entity_mut(entity).insert(old);
        } else {
            world.entity_mut(entity).remove::<N1Watch>();
        }
    }
    let mut prepared = prepared.map_err(value_error)?;
    if let Some(selection) = selection {
        prepared.load_faults(selection).map_err(value_error)?;
    }
    let study = prepared.study().map_err(value_error)?;
    let faults = prepared
        .faults()
        .cloned()
        .unwrap_or_else(|| study.all_branch_outages());
    drop(grid);
    Ok(Study {
        owner,
        prepared,
        faults,
        preparation_seconds: started.elapsed().as_secs_f64(),
    })
}
