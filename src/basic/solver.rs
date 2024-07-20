use rsparse::{self, data, lusol};

#[cfg(feature = "klu")]
#[derive(Default)]
pub struct KLUSolver(pub klu_rs::KLUSolver);

#[derive(Default)]
pub struct RSparseSolver;

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

#[cfg(feature = "klu")]
#[allow(non_snake_case)]
impl Solve for KLUSolver {
    #[allow(unused)]
    /// Solves the sparse linear system using the KLU solver.
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
        unsafe {
            let mut ret = self.0.solve_sym(
                Ap.as_mut_ptr() as *mut i64,
                Ai.as_mut_ptr() as *mut i64,
                n as i64,
            );
            ret |= self.0.factor(
                Ap.as_mut_ptr() as *mut i64,
                Ai.as_mut_ptr() as *mut i64,
                Ax.as_mut_ptr(),
            );
            ret |= self.0.solve(b.as_mut_ptr(), n as i64, 1);
            if ret != 0 {
                return Err("error occurred when calling KLU routines!");
            }
        }
        Ok(())
    }
}

#[cfg(feature = "klu")]
#[test]
/// Tests the drop functionality of the KLU solver.
fn drop_test() {
    let klu = KLUSolver::default();
    drop(klu);
}

#[cfg(feature = "klu")]
#[test]
/// Tests the reset functionality of the KLU solver.
fn reset_test() {
    let mut klu = KLUSolver::default();
    klu.0.reset();
}

#[allow(non_snake_case)]
impl Solve for RSparseSolver {
    #[allow(unused)]
    /// Solves the sparse linear system using the RSparse solver.
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
        let mut mat: data::Sprs = rsparse::data::Sprs::zeros(Ap.len() - 1, Ap.len() - 1, Ai.len());

        let p = unsafe { std::slice::from_raw_parts_mut(Ap.as_mut_ptr() as *mut isize, Ap.len()) };
        unsafe {
            mat.i.clone_from_slice(Ai);
            mat.p.clone_from_slice(p);
            mat.x.clone_from_slice(Ax);
        }

        lusol(&mat, b, 1, 1e-6);
        Ok(())
    }
}
