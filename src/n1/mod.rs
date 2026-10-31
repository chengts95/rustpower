//! Experimental fixed-partition N-1 contingency studies.
//!
//! The high-level entry point is [`prepare_n1_for`], followed by
//! [`PreparedN1::study`] and [`N1Study::run`] or [`N1Study::details`]. Preparation
//! freezes an initialized source [`World`](bevy_ecs::world::World) into numerical
//! arrays, a branch catalog, monitoring rules, and topology metadata. Later source
//! edits do not affect that snapshot; prepare a new study after changing the model.
//!
//! Every selected branch is removed independently while all other devices retain
//! their baseline state. Bus types remain fixed throughout a study, so Q-limit
//! PV-to-PQ switching is not part of this executor. Bridge outages that would
//! create a partition without a fixed Slack are rejected before numerical solve.
//!
//! The implementation is split by lifetime rather than by solver stage:
//!
//! - [`study`]: public fault selection, dispatch, reports, and reruns;
//! - [`execution`]: bounded CPU scheduling and aggregate/event collection;
//! - [`solver`] and `numeric`: shared symbolic analyses and private Newton/KLU state;
//! - [`outage`]: fixed-size CSC value patches, without structural matrix changes;
//! - [`topology`]: one-time bridge analysis and constant-time island queries;
//! - [`watch`] and [`postprocess`]: compiled monitoring rules and branch results;
//! - [`runner`]: lower-level unchanged-base batch execution used by preparation.
//!
//! `N1Study` is the recommended API. Types re-exported from [`runner`] and the
//! lower-level worker interfaces remain available for custom numerical pipelines.
mod numeric;
pub mod postprocess;
pub mod runner;
pub mod solver;
pub use crate::basic::branch::BranchFlowResult;
pub use postprocess::{BranchCatalog, BranchIdentity, BranchKind};
pub use runner::{
    BaseBatch, BaseResult, BaseTask, PreparedBase, RunSummary, prepare_base, run_base_batches,
};
pub use solver::{
    CaseSolution, LinearSolveStats, N1Worker, SharedBaseline, SolveStage, SolveTimings,
};

#[cfg(feature = "experimental_n1_cuda")]
pub mod cuda;
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
