use super::Solve;
use rustpower_sol_klu as klu_rs;

#[derive(Default)]
pub struct KLUSolver(pub klu_rs::KLUSolver);

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
            if self.0.symbolic.is_null() {
                self.0.solve_sym(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    n as i64,
                );
            }
            
            let mut ret = if self.0.numeric.is_null() {
                self.0.factor(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    Ax.as_mut_ptr(),
                )
            } else {
                // Try refactor first for speed
                let status = self.0.refactor(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    Ax.as_mut_ptr(),
                    n as i64,
                );
                // status > 0 means singular, status < 0 means error. 
                // In both cases, we try a full factor.
                if status != 0 {
                    self.0.factor(
                        Ap.as_mut_ptr() as *mut i64,
                        Ai.as_mut_ptr() as *mut i64,
                        Ax.as_mut_ptr(),
                    )
                } else {
                    0
                }
            };

            ret |= self.0.solve(b.as_mut_ptr(), n as i64, 1);
            if ret != 0 {
                return Err("error occurred when calling KLU routines!");
            }
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.0.reset();
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
