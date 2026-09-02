#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "faer")]
pub use faer::*;

#[cfg(any(feature = "klu", feature = "klu_dyn"))]
mod klu;
#[cfg(any(feature = "klu", feature = "klu_dyn"))]
pub use klu::*;

#[cfg(feature = "ldl")]
mod ldl;
#[cfg(feature = "ldl")]
pub use ldl::*;

#[cfg(feature = "qdldl")]
mod qdldl;
#[cfg(feature = "qdldl")]
pub use qdldl::*;

#[cfg(feature = "rsparse")]
mod rsparse;
#[cfg(feature = "rsparse")]
pub use rsparse::*;

#[cfg(all(
    not(feature = "klu"),
    not(feature = "klu_dyn"),
    not(feature = "faer"),
    feature = "rsparse"
))]
pub type DefaultSolver = RSparseSolver;

#[cfg(any(feature = "klu", feature = "klu_dyn"))]
pub type DefaultSolver = KLUSolver;

#[cfg(all(not(feature = "klu"), not(feature = "klu_dyn"), feature = "faer"))]
pub type DefaultSolver = FaerSolver;

/// Default backend for the LM-family augmented KKT system.
///
/// The augmented system `[μI Jᵀ; J −I]` is symmetric indefinite: a general
/// LU (`DefaultSolver`) solves it but wastes the symmetry, so the LM path
/// defaults to an LDLᵀ factorization instead. SuiteSparse LDL (feature
/// `ldl`) wins when available; otherwise pure-Rust QDLDL (in the default
/// feature set — no external libraries, no SuiteSparse). With both off the
/// alias degrades to the global `DefaultSolver`.
#[cfg(feature = "ldl")]
pub type DefaultLmSolver = LDLSolver;
#[cfg(all(not(feature = "ldl"), feature = "qdldl"))]
pub type DefaultLmSolver = QDLDLSolver;
#[cfg(all(not(feature = "ldl"), not(feature = "qdldl")))]
pub type DefaultLmSolver = DefaultSolver;

#[allow(non_snake_case)]
/// A trait for solving sparse linear systems.
pub trait Solve {
    /// Solves the sparse linear system.
    ///
    /// # Parameters
    ///
    /// * `Ap` - Column pointers of the matrix.
    /// * `Ai` - Row indices of the matrix.
    /// * `Ax` - Non-zero values of the matrix.
    /// * `_b` - Right-hand side vector.
    /// * `_n` - Dimension of the system.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        _b: &mut [f64],
        _n: usize,
    ) -> Result<(), &'static str>;

    fn reset(&mut self);
}
