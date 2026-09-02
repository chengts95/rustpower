use super::Solve;
use rustpower_sol_klu as klu_rs;

/// Performance instrumentation (see `crate::timeit!`): symbolic vs refactor
/// vs factor-fallback vs triangular-solve wall time, with per-phase call
/// counts. Exists only with the `probe` feature; compiled out entirely
/// otherwise.
#[cfg(feature = "probe")]
pub mod klu_probe {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static SYM_NS: AtomicU64 = AtomicU64::new(0);
    pub static FACTOR_NS: AtomicU64 = AtomicU64::new(0);
    pub static REFACTOR_NS: AtomicU64 = AtomicU64::new(0);
    pub static SOLVE_NS: AtomicU64 = AtomicU64::new(0);
    pub static N_REFACTOR: AtomicU64 = AtomicU64::new(0);
    pub static N_FACTOR_FALLBACK: AtomicU64 = AtomicU64::new(0);
    pub static N_FIRST_FACTOR: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        for a in [
            &SYM_NS,
            &FACTOR_NS,
            &REFACTOR_NS,
            &SOLVE_NS,
            &N_REFACTOR,
            &N_FACTOR_FALLBACK,
            &N_FIRST_FACTOR,
        ] {
            a.store(0, Ordering::Relaxed);
        }
    }
    pub fn report() -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "KLU breakdown: sym={:.3}ms factor={:.3}ms(first={}) refactor={:.3}ms({}) fallback={} tri_solve={:.3}ms",
            g(&SYM_NS) as f64 / 1e6,
            g(&FACTOR_NS) as f64 / 1e6,
            g(&N_FIRST_FACTOR),
            g(&REFACTOR_NS) as f64 / 1e6,
            g(&N_REFACTOR),
            g(&N_FACTOR_FALLBACK),
            g(&SOLVE_NS) as f64 / 1e6,
        )
    }
}

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
                crate::timeit!(klu_probe::SYM_NS, {
                    self.0.solve_sym(
                        Ap.as_mut_ptr() as *mut i64,
                        Ai.as_mut_ptr() as *mut i64,
                        n as i64,
                    );
                });
            }

            let mut ret = if self.0.numeric.is_null() {
                crate::probe_count!(klu_probe::N_FIRST_FACTOR);
                crate::timeit!(klu_probe::FACTOR_NS, {
                    self.0.factor(
                        Ap.as_mut_ptr() as *mut i64,
                        Ai.as_mut_ptr() as *mut i64,
                        Ax.as_mut_ptr(),
                    )
                })
            } else {
                // Try refactor first for speed.
                crate::probe_count!(klu_probe::N_REFACTOR);
                let status = crate::timeit!(klu_probe::REFACTOR_NS, {
                    self.0.refactor(
                        Ap.as_mut_ptr() as *mut i64,
                        Ai.as_mut_ptr() as *mut i64,
                        Ax.as_mut_ptr(),
                        n as i64,
                    )
                });
                // status > 0 means singular, status < 0 means error.
                // In both cases, we try a full factor.
                if status != 0 {
                    crate::probe_count!(klu_probe::N_FACTOR_FALLBACK);
                    crate::timeit!(klu_probe::FACTOR_NS, {
                        self.0.factor(
                            Ap.as_mut_ptr() as *mut i64,
                            Ai.as_mut_ptr() as *mut i64,
                            Ax.as_mut_ptr(),
                        )
                    })
                } else {
                    0
                }
            };

            ret |= crate::timeit!(klu_probe::SOLVE_NS, {
                self.0.solve(b.as_mut_ptr(), n as i64, 1)
            });
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
