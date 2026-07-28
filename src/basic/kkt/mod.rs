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

pub mod block;
pub mod cache;
pub mod flat;
pub mod kernels;
pub mod pattern;

pub use block::BlockDesc;
pub use cache::YbusAnalysisCache;
pub use flat::{fill_kkt, fill_kkt_flat, FlatLayout};
pub use kernels::{apply_mu_delta, fill_h, fill_jt};
pub use pattern::KktPattern;

#[cfg(test)]
mod tests;
