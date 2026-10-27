//! Experimental fixed-partition N-1 foundation. Shared analyses, private workers.
pub mod postprocess;
pub mod runner;
pub mod solver;
pub use postprocess::{BranchCatalog, BranchIdentity, BranchKind};
pub use runner::{
    BaseBatch, BaseResult, BaseTask, PreparedBase, RunSummary, prepare_base, run_base_batches,
};
pub use solver::{
    CaseSolution, LinearSolveStats, N1Worker, SharedBaseline, SolveStage, SolveTimings,
};

pub mod execution;
pub mod outage;
pub mod study;
pub mod topology;
pub mod watch;
pub use execution::{
    N1Config, N1Event, N1EventBatch, N1Initialization, N1Outcome, N1Reporting, N1Summary,
    PreparedN1, prepare_n1, prepare_n1_with_initialization, run_n1, run_n1_with_initialization,
};
pub use study::{
    N1FaultSet, N1Island, N1RunOptions, N1Study, N1StudyError, N1StudyReport, prepare_n1_for,
};
pub use topology::{BridgeCut, N1Topology};
pub use watch::{N1Watch, WatchCondition, WatchHit, WatchHits, WatchPlan, WatchRule, WatchTarget};
