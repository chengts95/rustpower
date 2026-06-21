#[cfg(feature = "faer")]
mod faer;
#[cfg(feature = "faer")]
pub use faer::*;

#[cfg(any(feature = "klu", feature = "klu_dyn"))]
mod klu;
#[cfg(any(feature = "klu", feature = "klu_dyn"))]
pub use klu::*;

#[cfg(feature = "rsparse")]
mod rsparse;
#[cfg(feature = "rsparse")]
pub use rsparse::*;

#[cfg(any(feature = "klu", feature = "klu_dyn"))]
pub type DefaultSolver = KLUSolver;

 
#[cfg(all(not(feature = "klu"), not(feature = "klu_dyn"), feature = "faer"))]
pub type DefaultSolver = FaerSolver;

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
