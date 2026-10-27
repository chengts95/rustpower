//! Streaming single-branch contingencies on worker-local Apps.
use super::{
    BranchIdentity, CaseSolution, LinearSolveStats, N1Topology, N1Worker, PreparedBase, SolveStage,
    outage::{OutageTask, WorkerPatchState},
    runner::prepare_base,
    watch::{WatchHits, WatchPlan},
};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// Default initial-guess policy; all workers in a run use the same policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum N1Initialization {
    /// Solve the unchanged base once; restart every outage from this AC voltage.
    #[default]
    BaseAc,
    /// Solve the scenario's patched DCPF before its Newton solve.
    DcPerCase,
}

pub struct PreparedN1 {
    base: PreparedBase,
    watch: WatchPlan,
    base_solution: Option<CaseSolution>,
    initialization: N1Initialization,
    topology: N1Topology,
    pub(super) faults: Option<super::study::N1FaultSet>,
}
impl PreparedN1 {
    pub fn base(&self) -> &PreparedBase {
        &self.base
    }
    pub fn watch(&self) -> &WatchPlan {
        &self.watch
    }
    /// One-time bridge/DFS analysis shared by all contingency queries.
    pub fn topology(&self) -> &N1Topology {
        &self.topology
    }
    /// Main-thread AC solution used as the immutable start voltage for every outage.
    pub fn base_solution(&self) -> Option<&CaseSolution> {
        self.base_solution.as_ref()
    }
    pub fn initialization(&self) -> N1Initialization {
        self.initialization
    }
    /// Dense indices of initially in-service branches, streamed from the shared catalog.
    pub fn active_branches(&self) -> impl Iterator<Item = usize> + '_ {
        self.base
            .catalog()
            .in_service()
            .iter()
            .enumerate()
            .filter(|(_, active)| **active)
            .map(|(i, _)| i)
    }
    pub fn branch_index(&self, source: Entity) -> Option<usize> {
        self.base
            .catalog()
            .source_entities()
            .iter()
            .position(|&e| e == source)
    }
    /// Validate the unchanged base against numerical constraints. Diagnostic
    /// Always/SolveFailed rules do not count as constraint violations.
    pub fn check_baseline(&self) -> Result<(), String> {
        let solution;
        let solved = if let Some(base) = self.base_solution() {
            base
        } else {
            solution =
                N1Worker::new(self.base.baseline()).solve_base(self.base.baseline(), 50, 1e-10);
            &solution
        };
        if !solved.converged {
            return Err(format!(
                "base power flow failed: {:?}: {:?}",
                solved.stage, solved.error
            ));
        }
        let hits = self
            .watch
            .evaluate(self.base.catalog(), &solved.voltage, None, true)?;
        if hits.violation_count() == 0 {
            return Ok(());
        }
        let first = hits
            .first_violation()
            .expect("constraint count has a numerical hit");
        Err(format!(
            "base violates {} monitored constraints; first target {:?}, rule {}, value {:?}; adjust the base before running N-1",
            hits.violation_count(),
            first.target,
            first.rule_id,
            first.value
        ))
    }
}
pub fn prepare_n1(world: &mut World) -> Result<PreparedN1, String> {
    prepare_n1_with_initialization(world, N1Initialization::BaseAc)
}
/// Choose base-AC initialization or per-scenario DCPF before starting workers.
pub fn prepare_n1_with_initialization(
    world: &mut World,
    initialization: N1Initialization,
) -> Result<PreparedN1, String> {
    let mut base = prepare_base(world)?;
    let topology = N1Topology::from_baseline(base.baseline(), base.catalog())?;
    let base_solution = match initialization {
        N1Initialization::BaseAc => Some(base.initialize_ac_start(50, 1e-10)?),
        N1Initialization::DcPerCase => None,
    };
    let watch = WatchPlan::from_world(world, base.catalog())?;
    Ok(PreparedN1 {
        base,
        watch,
        base_solution,
        initialization,
        topology,
        faults: None,
    })
}

/// Output policy for constraint screening and diagnostic postprocessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum N1Reporting {
    /// Preserve configured detailed rule hits, including diagnostic rules.
    Detailed,
    /// Stop at the first violated constraint and emit one status per rejected case.
    #[default]
    StatusOnly,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum N1Outcome {
    Passed,
    ConstraintViolation,
    NumericalFailure,
}

#[derive(Clone, Copy, Debug)]
pub struct N1Config {
    pub workers: usize,
    pub batch_size: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    /// Upper bound on per-group voltage/trigger-result scratch. Excludes persistent LU.
    pub max_batch_bytes: usize,
    pub reporting: N1Reporting,
}
impl Default for N1Config {
    fn default() -> Self {
        Self {
            workers: 1,
            batch_size: 8,
            max_iter: 30,
            tolerance: 1e-10,
            max_batch_bytes: 64 * 1024 * 1024,
            reporting: N1Reporting::StatusOnly,
        }
    }
}
impl N1Config {
    fn validate(self, prepared: &PreparedN1) -> Result<(), String> {
        if self.workers == 0 || self.workers > usize::MAX / 2 {
            return Err("N-1 worker count must be positive and representable".into());
        }
        if self.batch_size == 0 || self.batch_size > 32 {
            return Err("N-1 batch_size must be in 1..=32".into());
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0. {
            return Err("N-1 tolerance must be finite and positive".into());
        }
        let per_case = prepared
            .base
            .baseline()
            .v_init()
            .len()
            .checked_mul(std::mem::size_of::<num_complex::Complex64>())
            .and_then(|n| {
                let output = match self.reporting {
                    N1Reporting::Detailed => prepared.watch.max_output_bytes(),
                    N1Reporting::StatusOnly => Some(0),
                };
                output.and_then(|output| n.checked_add(output))
            })
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<OutageTask>() + std::mem::size_of::<N1Event>())
            })
            .ok_or("N-1 group byte count overflow")?;
        let need = per_case
            .checked_mul(self.batch_size)
            .ok_or("N-1 group byte count overflow")?;
        if need > self.max_batch_bytes {
            return Err(format!(
                "N-1 batch needs at least {need} scratch bytes; lower batch_size or increase max_batch_bytes"
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct N1Event {
    pub case_id: usize,
    pub outage_index: usize,
    pub outage: BranchIdentity,
    pub outage_entity: Entity,
    pub worker_id: usize,
    pub stage: SolveStage,
    pub outcome: N1Outcome,
    pub iterations: usize,
    pub residual: f64,
    pub error: Option<String>,
    pub hits: WatchHits,
    pub linear: LinearSolveStats,
}
#[derive(Debug)]
pub struct N1EventBatch {
    pub events: Vec<N1Event>,
}
#[derive(Default, Debug, Clone, Copy)]
pub struct N1Summary {
    pub submitted: usize,
    pub processed: usize,
    pub failed: usize,
    pub emitted_cases: usize,
    pub emitted_hits: usize,
    pub violated_cases: usize,
    /// Detailed threshold-hit payload count; StatusOnly emits no hit payloads.
    pub violation_hits: usize,
    pub workers_started: usize,
    pub total_iterations: usize,
    pub attempted_iterations: usize,
    pub dc_failed: usize,
    pub newton_failed: usize,
    pub dc_time: Duration,
    pub ac_time: Duration,
    pub ac_iteration_time: Duration,
    pub patch_time: Duration,
    pub watch_time: Duration,
    pub linear: LinearSolveStats,
}
impl N1Summary {
    pub fn rejected(&self) -> usize {
        self.failed + self.violated_cases
    }
    pub fn passed(&self) -> usize {
        self.processed - self.rejected()
    }
    fn add(&mut self, r: Self) {
        self.processed += r.processed;
        self.failed += r.failed;
        self.emitted_cases += r.emitted_cases;
        self.emitted_hits += r.emitted_hits;
        self.violated_cases += r.violated_cases;
        self.violation_hits += r.violation_hits;
        self.workers_started += r.workers_started;
        self.total_iterations += r.total_iterations;
        self.attempted_iterations += r.attempted_iterations;
        self.dc_failed += r.dc_failed;
        self.newton_failed += r.newton_failed;
        self.dc_time += r.dc_time;
        self.ac_time += r.ac_time;
        self.ac_iteration_time += r.ac_iteration_time;
        self.patch_time += r.patch_time;
        self.watch_time += r.watch_time;
        self.linear = self.linear.add(r.linear);
    }
}
// Only the scoped runner constructs the raw references. Worker Apps drop before
// their threads join; the immutable PreparedN1 borrow outlives the whole scope.
#[derive(Resource, Clone, Copy)]
struct Input(*const PreparedN1);
unsafe impl Send for Input {}
unsafe impl Sync for Input {}
#[derive(Resource)]
struct State {
    id: usize,
    config: N1Config,
    initialization: N1Initialization,
    emit_all: bool,
    engine: N1Worker,
    patches: WorkerPatchState,
    inbox: Option<Vec<OutageTask>>,
    solved: Vec<(OutageTask, CaseSolution)>,
    events: Vec<N1Event>,
    summary: N1Summary,
    fatal_error: Option<String>,
}
#[derive(Message)]
struct ExecuteGroup;
struct N1WorkerPlugin;
impl Plugin for N1WorkerPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ExecuteGroup>();
        app.add_systems(Update, (solve_group, emit_group).chain());
    }
}
fn solve_group(
    mut messages: MessageReader<ExecuteGroup>,
    input: Res<Input>,
    mut state: ResMut<State>,
) {
    if messages.is_empty() {
        return;
    }
    messages.clear();
    let Some(tasks) = state.inbox.take() else {
        return;
    };
    let prepared = unsafe { &*input.0 };
    let base = prepared.base.baseline();
    let config = state.config;
    for task in tasks {
        let state = &mut *state;
        let patch_started = Instant::now();
        let patched = state.patches.apply(&mut state.engine, base, &task);
        state.summary.patch_time += patch_started.elapsed();
        let solution = match patched {
            Ok(()) => match state.initialization {
                N1Initialization::BaseAc => {
                    state
                        .engine
                        .solve_ac_current(base, config.max_iter, config.tolerance)
                }
                N1Initialization::DcPerCase => {
                    state
                        .engine
                        .solve_current(base, config.max_iter, config.tolerance)
                }
            },
            Err(error) => CaseSolution {
                converged: false,
                stage: SolveStage::NewtonFailed,
                iterations: 0,
                residual: f64::INFINITY,
                voltage: base.v_init().to_vec(),
                error: Some(error),
                timings: Default::default(),
                linear: Default::default(),
            },
        };
        state.solved.push((task, solution));
    }
}
fn emit_group(input: Res<Input>, mut state: ResMut<State>) {
    let prepared = unsafe { &*input.0 };
    let catalog = prepared.base.catalog();
    let solved = std::mem::take(&mut state.solved);
    for (task, solution) in solved {
        state.summary.processed += 1;
        state.summary.linear = state.summary.linear.add(solution.linear);
        state.summary.total_iterations += solution.iterations;
        state.summary.attempted_iterations += solution.timings.attempted_iterations;
        state.summary.dc_time += solution.timings.dc;
        state.summary.ac_time += solution.timings.ac;
        state.summary.ac_iteration_time += solution.timings.ac_iterations;
        match solution.stage {
            SolveStage::DcFailed => state.summary.dc_failed += 1,
            SolveStage::NewtonFailed => state.summary.newton_failed += 1,
            SolveStage::Converged => {}
        }
        if !solution.converged {
            state.summary.failed += 1;
        }
        let watch_started = Instant::now();
        let reporting = state.config.reporting;
        let evaluated = match reporting {
            N1Reporting::Detailed => prepared
                .watch
                .evaluate(
                    catalog,
                    &solution.voltage,
                    Some(task.branch_index),
                    solution.converged,
                )
                .map(|hits| {
                    let violated = hits.violation_count() > 0;
                    (hits, violated)
                }),
            N1Reporting::StatusOnly => {
                if solution.converged {
                    prepared
                        .watch
                        .is_violated(catalog, &solution.voltage, Some(task.branch_index))
                        .map(|violated| (WatchHits::default(), violated))
                } else {
                    Ok((WatchHits::default(), false))
                }
            }
        };
        state.summary.watch_time += watch_started.elapsed();
        let (hits, violated) = match evaluated {
            Ok(result) => result,
            Err(error) => {
                state.summary.failed += usize::from(solution.converged);
                state.fatal_error = Some(error);
                continue;
            }
        };
        let outcome = if !solution.converged {
            N1Outcome::NumericalFailure
        } else if violated {
            N1Outcome::ConstraintViolation
        } else {
            N1Outcome::Passed
        };
        state.summary.violated_cases += usize::from(violated);
        state.summary.violation_hits += hits.violation_count();
        let emit = state.emit_all
            || match reporting {
                N1Reporting::Detailed => !hits.is_empty(),
                N1Reporting::StatusOnly => outcome != N1Outcome::Passed,
            };
        if !emit {
            continue;
        }
        state.summary.emitted_cases += 1;
        state.summary.emitted_hits += hits.len();
        let event = N1Event {
            case_id: task.case_id,
            outage_index: task.branch_index,
            outage: catalog.identities()[task.branch_index],
            outage_entity: catalog.source_entities()[task.branch_index],
            worker_id: state.id,
            stage: solution.stage,
            outcome,
            iterations: solution.iterations,
            residual: solution.residual,
            error: solution.error,
            hits,
            linear: solution.linear,
        };
        state.events.push(event);
    }
}
fn worker(
    prepared: &PreparedN1,
    config: N1Config,
    initialization: N1Initialization,
    emit_all: bool,
    id: usize,
    rx: Receiver<Vec<OutageTask>>,
    tx: Sender<N1EventBatch>,
    done: &AtomicBool,
) -> Result<N1Summary, String> {
    let mut app = App::new();
    app.insert_resource(Input(prepared));
    app.insert_resource(State {
        id,
        config,
        initialization,
        emit_all,
        engine: match initialization {
            N1Initialization::BaseAc => N1Worker::new_ac(prepared.base.baseline()),
            N1Initialization::DcPerCase => N1Worker::new(prepared.base.baseline()),
        },
        patches: WorkerPatchState::default(),
        inbox: None,
        solved: vec![],
        events: vec![],
        fatal_error: None,
        summary: N1Summary {
            workers_started: 1,
            ..Default::default()
        },
    });
    app.add_plugins(N1WorkerPlugin);
    let mut empty = 0usize;
    loop {
        let finished = done.load(Ordering::Acquire);
        match rx.try_recv() {
            Ok(group) => {
                empty = 0;
                app.world_mut().resource_mut::<State>().inbox = Some(group);
                app.world_mut().write_message(ExecuteGroup);
                app.update();
                let events = std::mem::take(&mut app.world_mut().resource_mut::<State>().events);
                if !events.is_empty() && tx.send(N1EventBatch { events }).is_err() {
                    return Err("N-1 collector disconnected".into());
                }
                if app.world().resource::<State>().fatal_error.is_some() {
                    break;
                }
            }
            Err(TryRecvError::Empty) if finished => break,
            Err(TryRecvError::Disconnected) => break,
            Err(TryRecvError::Empty) => {
                empty = empty.saturating_add(1);
                if empty < 64 {
                    std::thread::yield_now()
                } else {
                    std::thread::sleep(Duration::from_micros(50));
                }
            }
        }
    }
    let mut state = app.world_mut().resource_mut::<State>();
    if let Some(error) = state.fatal_error.take() {
        Err(error)
    } else {
        Ok(state.summary)
    }
}
/// Run selected in-service branch indices as independent single-device outages.
/// Omit watchers for a silent run; callback receives only configured rule hits.
/// Task construction is streamed on the producer thread, bounded by group size.
pub fn run_n1<I, F>(
    prepared: &PreparedN1,
    config: N1Config,
    candidates: I,
    collect: F,
) -> Result<N1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch) + Send,
{
    run_n1_with_initialization(
        prepared,
        prepared.initialization,
        config,
        candidates,
        collect,
    )
}

/// Run another round with an explicit initial guess and the same shared baseline.
pub fn run_n1_with_initialization<I, F>(
    prepared: &PreparedN1,
    initialization: N1Initialization,
    config: N1Config,
    candidates: I,
    collect: F,
) -> Result<N1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch) + Send,
{
    run_round(prepared, initialization, config, candidates, false, collect)
}

pub(super) fn run_round<I, F>(
    prepared: &PreparedN1,
    initialization: N1Initialization,
    config: N1Config,
    candidates: I,
    emit_all: bool,
    mut collect: F,
) -> Result<N1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch) + Send,
{
    config.validate(prepared)?;
    if initialization == N1Initialization::BaseAc && prepared.base.baseline().ac_start().is_none() {
        return Err("BaseAc initialization requires a prepared base AC voltage".into());
    }
    let (task_tx, task_rx) = bounded::<Vec<OutageTask>>(config.workers * 2);
    let (result_tx, result_rx) = bounded::<N1EventBatch>(config.workers * 2);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let collector = std::thread::Builder::new()
            .name("n1-collector".into())
            .spawn_scoped(scope, move || {
                while let Ok(batch) = result_rx.recv() {
                    collect(batch);
                }
            })
            .map_err(|e| format!("start N-1 collector: {e}"))?;
        let mut handles = vec![];
        let mut error = None;
        for id in 0..config.workers {
            let rx = task_rx.clone();
            let tx = result_tx.clone();
            let finished = &done;
            match std::thread::Builder::new()
                .name(format!("n1-worker-{id}"))
                .spawn_scoped(scope, move || {
                    worker(
                        prepared,
                        config,
                        initialization,
                        emit_all,
                        id,
                        rx,
                        tx,
                        finished,
                    )
                }) {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    error = Some(format!("start N-1 worker: {e}"));
                    break;
                }
            }
        }
        drop(task_rx);
        drop(result_tx);
        let mut submitted = 0;
        if error.is_none() {
            let mut group = Vec::with_capacity(config.batch_size);
            for (case_id, branch_index) in candidates.into_iter().enumerate() {
                let compile = match initialization {
                    N1Initialization::BaseAc => OutageTask::for_ac_branch,
                    N1Initialization::DcPerCase => OutageTask::for_branch,
                };
                match compile(
                    prepared.base.baseline(),
                    prepared.base.catalog(),
                    branch_index,
                    case_id,
                ) {
                    Ok(task) => group.push(task),
                    Err(e) => {
                        error = Some(e);
                        break;
                    }
                }
                if group.len() == config.batch_size {
                    submitted += group.len();
                    if task_tx
                        .send(std::mem::replace(
                            &mut group,
                            Vec::with_capacity(config.batch_size),
                        ))
                        .is_err()
                    {
                        error = Some("N-1 workers stopped before all tasks were submitted".into());
                        break;
                    }
                }
            }
            if error.is_none() && !group.is_empty() {
                submitted += group.len();
                if task_tx.send(group).is_err() {
                    error = Some("N-1 workers stopped before all tasks were submitted".into());
                }
            }
        }
        done.store(true, Ordering::Release);
        drop(task_tx);
        let mut summary = N1Summary {
            submitted,
            ..Default::default()
        };
        for handle in handles {
            match handle.join() {
                Ok(Ok(s)) => summary.add(s),
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("N-1 worker panicked".into()),
            }
        }
        if collector.join().is_err() {
            error = Some("N-1 collector panicked".into());
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(summary)
        }
    })
}
