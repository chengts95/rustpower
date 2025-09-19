use faer::{
    MatMut,
    linalg::solvers::Solve,
    sparse::{
        SparseColMatRef, SymbolicSparseColMatRef,
        linalg::solvers::{Lu, SymbolicLu},
    },
};

use super::Solve as PoSolve;
#[derive(Default)]
pub struct FaerSolver {
    lu: Option<Lu<usize, f64>>,
    symbolic: Option<SymbolicLu<usize>>,
}

#[allow(non_snake_case)]
impl PoSolve for FaerSolver {
    #[allow(unused)]
    /// Solves the sparse linear system using the Faer solver.
    ///
    /// # Parameters
    ///
    /// * `Ap` - Column pointers of the matrix.
    /// * `Ai` - Row indices of the matrix.
    /// * `Ax` - Non-zero values of the matrix.
    /// * `b` - Right-hand side vector.
    /// * `n` - Dimension of the system.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        let s = unsafe { SymbolicSparseColMatRef::new_unchecked(n, n, Ap, None, Ai) };
        let mat = SparseColMatRef::new(s, Ax);
        if self.symbolic.is_none() {
            self.symbolic = Some(SymbolicLu::try_new(s).map_err(|_| "Faer symbolic error")?);
        }

        self.lu = Some(
            Lu::try_new_with_symbolic(self.symbolic.as_ref().unwrap().clone(), mat)
                .map_err(|_| "Faer numerical error")?,
        );
        let mat_ref = MatMut::from_column_major_slice_mut(b, n, 1);
        self.lu.as_ref().unwrap().solve_in_place(mat_ref);
        Ok(())
    }
    fn reset(&mut self) {
        self.symbolic = None;
    }
}
