//! Base-case application/queue foundation, with explicit diagnostic emit policy.
use super::{BranchCatalog, CaseSolution, N1Worker, SharedBaseline};
use crate::basic::branch::BranchFlowResult;
use crate::basic::ecs::{elements::*, powerflow::systems::PowerFlowMat};
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use crossbeam_channel::{TryRecvError, bounded};
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

pub const MAX_BASE_BATCH: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct BaseTask {
    pub case_id: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    /// Explicit unconditional diagnostic trigger. False produces no result message.
    pub emit: bool,
}
pub struct BaseBatch(Vec<BaseTask>);
impl BaseBatch {
    pub fn new(tasks: Vec<BaseTask>) -> Result<Self, String> {
        if tasks.is_empty() || tasks.len() > MAX_BASE_BATCH {
            return Err(format!(
                "base task group must contain 1..={MAX_BASE_BATCH} cases"
            ));
        }
        Ok(Self(tasks))
    }
}
#[derive(Debug)]
pub struct BaseResult {
    pub case_id: usize,
    pub worker_id: usize,
    pub solution: CaseSolution,
    pub branches: Vec<BranchFlowResult>,
    pub postprocess_error: Option<String>,
    pub symbolic_ids: (usize, usize),
    pub pattern_ids: (usize, usize, usize, usize),
}
#[derive(Default, Clone, Copy, Debug)]
pub struct RunSummary {
    pub processed: usize,
    pub failed: usize,
    pub emitted: usize,
    pub workers_started: usize,
}
impl RunSummary {
    fn add(&mut self, other: Self) {
        self.processed += other.processed;
        self.failed += other.failed;
        self.emitted += other.emitted;
        self.workers_started += other.workers_started;
    }
}

/// Prepare one shared baseline and postprocessing catalog from an initialized World.
pub struct PreparedBase {
    baseline: SharedBaseline,
    catalog: BranchCatalog,
}
impl PreparedBase {
    pub fn baseline(&self) -> &SharedBaseline {
        &self.baseline
    }
    pub fn catalog(&self) -> &BranchCatalog {
        &self.catalog
    }
    /// Solve the unchanged base once, then freeze its converged AC voltage for workers.
    pub(super) fn initialize_ac_start(
        &mut self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<CaseSolution, String> {
        let mut engine = N1Worker::new(&self.baseline);
        let solution = engine.solve_base(&self.baseline, max_iter, tolerance);
        if !solution.converged {
            return Err(format!(
                "N-1 base AC solve failed at {:?}: {}",
                solution.stage,
                solution
                    .error
                    .as_deref()
                    .unwrap_or("residual exceeds tolerance")
            ));
        }
        self.baseline.set_ac_start(&solution.voltage)?;
        Ok(solution)
    }
}

pub fn prepare_base(world: &mut World) -> Result<PreparedBase, String> {
    if !world.contains_resource::<PowerFlowMat>() {
        return Err("initialize the power-flow model before preparing N-1".into());
    }
    let mut q = world.query_filtered::<(
        &Port4MatPatch,
        &TransformerDevice,
        &FromBusMatrixIndex,
        &ToBusMatrixIndex,
    ), Without<OutOfService>>();
    let mat = world.resource::<PowerFlowMat>();
    let common = world
        .get_resource::<PFCommonData>()
        .ok_or("N-1 baseline requires PFCommonData")?;
    let agg = world.get_resource::<NodeAggRes>();
    let mut shift = vec![0.; mat.y_bus.ncols()];
    for (patch, dev, from, to) in q.iter(world) {
        let angle = dev.effective_shift_degree();
        if angle.abs() > 1e-6 {
            let b = patch.0[(0, 1)].norm() * dev.vn_lv_kv.powi(2) / common.sbase;
            let injection = b * -angle.to_radians();
            let f = agg.map_or(from.0, |a| a.original_to_merged[from.0]);
            let t = agg.map_or(to.0, |a| a.original_to_merged[to.0]);
            shift[f] += injection;
            shift[t] -= injection;
        }
    }
    let baseline = SharedBaseline::from_mat(mat, &shift)?;
    let catalog = BranchCatalog::from_world(world)?;
    Ok(PreparedBase { baseline, catalog })
}

// Scoped runner is the sole constructor of this resource. Pointees remain alive
// and immutable until every worker App has been dropped and its thread joined.
#[derive(Resource, Clone, Copy)]
struct SharedInput {
    base: *const SharedBaseline,
    catalog: *const BranchCatalog,
}
unsafe impl Send for SharedInput {}
unsafe impl Sync for SharedInput {}
#[derive(Resource)]
struct WorkerState {
    id: usize,
    engine: N1Worker,
    inbox: Option<BaseBatch>,
    solved: Vec<(BaseTask, CaseSolution)>,
    results: Vec<BaseResult>,
    progress: RunSummary,
}
#[derive(Message)]
struct ExecuteBaseBatch;
struct BaseWorkerPlugin;
impl Plugin for BaseWorkerPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ExecuteBaseBatch>();
        app.add_systems(Update, (solve_batch, postprocess_batch).chain());
    }
}
fn solve_batch(
    mut events: MessageReader<ExecuteBaseBatch>,
    shared: Res<SharedInput>,
    mut state: ResMut<WorkerState>,
) {
    if events.is_empty() {
        return;
    }
    events.clear();
    let Some(batch) = state.inbox.take() else {
        return;
    };
    let base = unsafe { &*shared.base };
    for task in batch.0 {
        let result = state.engine.solve_base(base, task.max_iter, task.tolerance);
        state.solved.push((task, result));
    }
}
fn postprocess_batch(shared: Res<SharedInput>, mut state: ResMut<WorkerState>) {
    let catalog = unsafe { &*shared.catalog };
    let solved = std::mem::take(&mut state.solved);
    for (task, solution) in solved {
        state.progress.processed += 1;
        if !solution.converged {
            state.progress.failed += 1;
        }
        if !task.emit {
            continue;
        }
        let mut branches = Vec::new();
        let postprocess_error = if solution.converged {
            catalog
                .compute(&solution.voltage, None, catalog.sbase(), &mut branches)
                .err()
        } else {
            None
        };
        if postprocess_error.is_some() {
            state.progress.failed += 1;
        }
        let result = BaseResult {
            case_id: task.case_id,
            worker_id: state.id,
            solution,
            branches,
            postprocess_error,
            symbolic_ids: state.engine.symbolic_ids(),
            pattern_ids: state.engine.pattern_ids(),
        };
        state.results.push(result);
        state.progress.emitted += 1;
    }
}

/// Run real DCPF/Newton base cases on worker-local Apps. Groups and both queues
/// have fixed bounds. Callback executes on a dedicated collector thread.
/// This foundation's emit flag is an explicit always/never rule; entity-derived
/// threshold rules and actual outage task descriptors are subsequent layers.
pub fn run_base_batches<I, F>(
    prepared: &PreparedBase,
    workers: usize,
    batches: I,
    mut collect: F,
) -> Result<RunSummary, String>
where
    I: IntoIterator<Item = BaseBatch>,
    F: FnMut(BaseResult) + Send,
{
    if workers == 0 || workers > usize::MAX / 2 {
        return Err("worker count must be positive and representable".into());
    }
    let base = prepared.baseline();
    let catalog = prepared.catalog();
    let (task_tx, task_rx) = bounded::<BaseBatch>(workers * 2);
    let (result_tx, result_rx) = bounded::<Vec<BaseResult>>(workers * 2);
    let producer_done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let collector = scope.spawn(move || {
            while let Ok(results) = result_rx.recv() {
                for result in results {
                    collect(result);
                }
            }
        });
        let mut handles = Vec::new();
        for id in 0..workers {
            let rx = task_rx.clone();
            let tx = result_tx.clone();
            let done = &producer_done;
            handles.push(scope.spawn(move || {
                let mut app = App::new();
                app.insert_resource(SharedInput { base, catalog });
                app.insert_resource(WorkerState {
                    id,
                    engine: N1Worker::new(base),
                    inbox: None,
                    solved: Vec::new(),
                    results: Vec::new(),
                    progress: RunSummary {
                        workers_started: 1,
                        ..Default::default()
                    },
                });
                app.add_plugins(BaseWorkerPlugin);
                let mut empty_polls = 0;
                loop {
                    let finished = done.load(Ordering::Acquire);
                    match rx.try_recv() {
                        Ok(batch) => {
                            empty_polls = 0;
                            app.world_mut().resource_mut::<WorkerState>().inbox = Some(batch);
                            app.world_mut().write_message(ExecuteBaseBatch);
                            app.update();
                            let results = std::mem::take(
                                &mut app.world_mut().resource_mut::<WorkerState>().results,
                            );
                            if !results.is_empty() && tx.send(results).is_err() {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) if finished => break,
                        Err(TryRecvError::Disconnected) => break,
                        Err(TryRecvError::Empty) => {
                            empty_polls += 1;
                            if empty_polls < 64 {
                                std::thread::yield_now();
                            } else {
                                std::thread::sleep(Duration::from_micros(50));
                            }
                        }
                    }
                }
                app.world().resource::<WorkerState>().progress
            }));
        }
        drop(task_rx);
        drop(result_tx);
        let mut send_failed = false;
        for batch in batches {
            if task_tx.send(batch).is_err() {
                send_failed = true;
                break;
            }
        }
        producer_done.store(true, Ordering::Release);
        drop(task_tx);
        let mut summary = RunSummary::default();
        let mut worker_failed = false;
        for handle in handles {
            match handle.join() {
                Ok(progress) => summary.add(progress),
                Err(_) => worker_failed = true,
            }
        }
        let collector_failed = collector.join().is_err();
        if send_failed || worker_failed || collector_failed {
            Err("base batch execution thread failed".into())
        } else {
            Ok(summary)
        }
    })
}
