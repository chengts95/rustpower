#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "faer")]
pub use faer::*;

#[cfg(feature = "klu")]
mod klu;
#[cfg(feature = "klu")]
pub use klu::*;

#[cfg(feature = "rsparse")]
mod rsparse;
#[cfg(feature = "rsparse")]
pub use rsparse::*;

#[cfg(feature = "klu")]
pub type DefaultSolver = KluSolver;

#[cfg(all(not(feature = "klu"), feature = "faer"))]
pub type DefaultSolver = FaerSolver;

#[cfg(all(not(feature = "klu"), not(feature = "faer"), feature = "rsparse"))]
pub type DefaultSolver = RSparseSolver;

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
}
