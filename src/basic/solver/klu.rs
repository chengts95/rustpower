use super::Solve;
use rustpower_sol_klu as klu_rs;

/// Lightweight global instrumentation for KLU solve breakdown (refactor vs factor-fallback
/// vs triangular solve). Enabled implicitly; read/reset via the helpers below.
pub mod klu_probe {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static REFACTOR_NS: AtomicU64 = AtomicU64::new(0);
    pub static FACTOR_NS: AtomicU64 = AtomicU64::new(0);
    pub static SOLVE_NS: AtomicU64 = AtomicU64::new(0);
    pub static SYM_NS: AtomicU64 = AtomicU64::new(0);
    pub static N_REFACTOR: AtomicU64 = AtomicU64::new(0);
    pub static N_FACTOR_FALLBACK: AtomicU64 = AtomicU64::new(0);
    pub static N_FIRST_FACTOR: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        for a in [&REFACTOR_NS, &FACTOR_NS, &SOLVE_NS, &SYM_NS,
                  &N_REFACTOR, &N_FACTOR_FALLBACK, &N_FIRST_FACTOR] {
            a.store(0, Ordering::Relaxed);
        }
    }
    pub fn report() -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "KLU breakdown: sym={:.3}ms factor(first)={:.3}ms refactor={:.3}ms({}) fallback={:.3}ms({}) tri_solve={:.3}ms",
            g(&SYM_NS) as f64 / 1e6,
            g(&FACTOR_NS) as f64 / 1e6,
            g(&REFACTOR_NS) as f64 / 1e6, g(&N_REFACTOR),
            0.0, g(&N_FACTOR_FALLBACK),
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
        use klu_probe::*;
        use std::sync::atomic::Ordering::Relaxed;
        unsafe {
            if self.0.symbolic.is_null() {
                let t = std::time::Instant::now();
                self.0.solve_sym(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    n as i64,
                );
                SYM_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
            }

            let mut ret = if self.0.numeric.is_null() {
                let t = std::time::Instant::now();
                let r = self.0.factor(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    Ax.as_mut_ptr(),
                );
                FACTOR_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                N_FIRST_FACTOR.fetch_add(1, Relaxed);
                r
            } else {
                // Try refactor first for speed
                let t = std::time::Instant::now();
                let status = self.0.refactor(
                    Ap.as_mut_ptr() as *mut i64,
                    Ai.as_mut_ptr() as *mut i64,
                    Ax.as_mut_ptr(),
                    n as i64,
                );
                REFACTOR_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                N_REFACTOR.fetch_add(1, Relaxed);
                // status > 0 means singular, status < 0 means error.
                // In both cases, we try a full factor.
                if status != 0 {
                    let t = std::time::Instant::now();
                    let r = self.0.factor(
                        Ap.as_mut_ptr() as *mut i64,
                        Ai.as_mut_ptr() as *mut i64,
                        Ax.as_mut_ptr(),
                    );
                    FACTOR_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                    N_FACTOR_FALLBACK.fetch_add(1, Relaxed);
                    r
                } else {
                    0
                }
            };

            let t = std::time::Instant::now();
            ret |= self.0.solve(b.as_mut_ptr(), n as i64, 1);
            SOLVE_NS.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
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
