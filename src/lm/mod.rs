//! Symbolic-pattern KKT/LM assembly (see `Symbolic_KKT_LM_Architecture.md`).
//!
//! Same principle as `JacobianPattern2` (new_dsdvbus2.rs): the Ybus CSC is
//! pre-ordered `[PQ | PV | slack]` with sorted row indices, so one
//! `partition_point` cuts each column into type segments and every write
//! position in the numeric phase is start-plus-offset arithmetic — no slot
//! maps, no per-edge tables, no runtime search.
//!
//! | Layer | Type | Role |
//! |---|---|---|
//! | 0 | [`cache::YbusAnalysisCache`] | every offset derived from the Ybus CSC |
//! | 1 | [`block::BlockDesc`] | one block matrix, identified by its `base` |
//! | — | [`pattern::KktPattern`] | the four blocks of the LM augmented system |
//! | 2 | [`flat::FlatLayout`] | the flat global CSC triple (direct-solver view) |
//!
//! The numeric kernels take a `FLAT` const generic selecting the storage
//! view; both views share one traversal and one set of formulas (doc §3.3).
//!
//! Retention conventions (architecture doc §1.6): the reduced PF/LM system
//! cuts each Ybus column at `pq_end` / `active_end`; the full-retention OPF
//! system is the degenerate case `n_pq = n_active = n_bus` where the cuts
//! resolve to whole columns.

pub mod baseline;
pub mod block;
pub mod cache;
pub mod exact;
pub mod flat;
pub mod gn_flat;
pub mod gn_triu;
pub mod kernels;
pub mod ldl_vs_klu;
pub mod normal_eq;
pub mod pattern;
pub mod residual;

pub use block::BlockDesc;
pub use cache::YbusAnalysisCache;
pub use exact::driver::{LmDriver, LmResult};
pub use flat::{fill_kkt, fill_kkt_flat, FlatLayout};
pub use kernels::{apply_mu_delta, fill_h, fill_jt};
pub use pattern::KktPattern;

/// Default GN-LM entry point for the selected LM backend. Mirrors the
/// [`crate::basic::solver::DefaultLmSolver`] feature ladder exactly:
///
/// * pure-Rust QDLDL (and no `ldl`) consumes the plain upper triangle, so
///   the plugin fills it directly — row-oriented, half the storage;
/// * SuiteSparse LDL accesses the upper triangle of the **permuted** PAP′,
///   so it must see the full symmetric pattern — triu-only input silently
///   drops entries that become upper after AMD permutation (caught by
///   `gn_plugin_ieee39_standard_case`);
/// * LU backends (KLU family) likewise need the full symmetric slim layout.
#[cfg(all(not(feature = "ldl"), feature = "qdldl"))]
pub use gn_triu::newton_pf_gn_triu as newton_pf_gn_default;
#[cfg(any(feature = "ldl", not(feature = "qdldl")))]
pub use gn_flat::newton_pf_gn as newton_pf_gn_default;

#[cfg(test)]
mod tests;
