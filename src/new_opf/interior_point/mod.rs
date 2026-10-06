//! Interior-point execution shared by the named configurations.
//!
//! The reference driver remains in `opf`. Optimized driver variants are retained
//! here unchanged so this module move does not alter initialization or iterations.
//! They reuse the reference constraint/stopping helpers; their loops can be
//! consolidated separately once an evaluator-workspace interface is established.
use crate::new_opf::solution::{PipsOpt, PipsResult, PipsTiming};
use crate::opf::pips::{
    build_linear_constraints, convergence_measures, matvec_add_to, merge_constraints, step_size,
};
use linear_system::solve_kkt_fused_timed;
use nalgebra_sparse::CscMatrix;
mod fused;
mod linear_system;
mod merged;
mod nonlinear;
pub use crate::opf::pips::pips_with_solver;
pub use fused::pips_with_fused_assembly;
pub use merged::pips_with_fused_assembly_v56;
pub use nonlinear::pips_with_fused_assembly_v55;
