//! SuiteSparse LDL backend for the LM augmented (quasi-definite) system.
//!
//! The [`Solve`] trait hands us the **full** symmetric CSC triple; LDL reads
//! the caller's `Ax` directly (no extraction, no gather): with a permutation
//! it accesses exactly the entries that land in the upper triangle of PAP′
//! and ignores the rest. Pattern is converted to `i32` once per analysis.
//!
//! Quasi-definiteness (`[μI Jᵀ; J −I]`, μ > 0) guarantees the no-pivot LDLᵀ
//! exists and is stable for any fill-reducing ordering (Vanderbei 1995), so
//! AMD from `analyze` stays valid across μ and across LM iterations.

use super::Solve;
use rustpower_sol_ldl as ldl_rs;

/// Performance instrumentation (see `crate::timeit!`): symbolic vs numeric
/// vs triangular-solve wall time and numeric-phase call count. Exists only
/// with the `probe` feature; compiled out entirely otherwise.
#[cfg(feature = "probe")]
pub mod ldl_probe {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static SYM_NS: AtomicU64 = AtomicU64::new(0);
    pub static NUMERIC_NS: AtomicU64 = AtomicU64::new(0);
    pub static SOLVE_NS: AtomicU64 = AtomicU64::new(0);
    pub static N_NUMERIC: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        for a in [&SYM_NS, &NUMERIC_NS, &SOLVE_NS, &N_NUMERIC] {
            a.store(0, Ordering::Relaxed);
        }
    }
    pub fn report() -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        format!(
            "LDL breakdown: sym={:.3}ms numeric={:.3}ms({}) tri_solve={:.3}ms",
            g(&SYM_NS) as f64 / 1e6,
            g(&NUMERIC_NS) as f64 / 1e6,
            g(&N_NUMERIC),
            g(&SOLVE_NS) as f64 / 1e6,
        )
    }
}

/// Everything derived from one sparsity pattern. Kept behind a single
/// `Option` in the solver: `None` IS "not analyzed", there is no second
/// flag to disagree with it.
struct Symbolic {
    /// Full symmetric pattern converted to i32 CSC (LDL input is int32).
    /// Values are read from the caller's `Ax` directly at factor time.
    ap32: Vec<i32>,
    ai32: Vec<i32>,
    /// Identity of the caller buffers this symbolic was built from. The
    /// validity check is pointer + length compare only (O(1), what the
    /// Newton hot loop hits every iteration). Correctness for *structural*
    /// changes is the event chain's job: topology / node-type changes always
    /// arrive with a `reset()` from `reset_solvers` before any pointer
    /// comparison would matter. Callers bypassing the event system must call
    /// `reset()` themselves when handing us a different matrix.
    ap_ptr: usize,
    ai_ptr: usize,
}

impl Symbolic {
    fn owns(&self, ap: &[usize], ai: &[usize]) -> bool {
        self.ap_ptr == ap.as_ptr() as usize
            && self.ai_ptr == ai.as_ptr() as usize
            && self.ap32.len() == ap.len()
            && self.ai32.len() == ai.len()
    }
}

#[derive(Default)]
pub struct LDLSolver {
    inner: ldl_rs::LDLSolver,
    symbolic: Option<Symbolic>,
}

#[allow(non_snake_case)]
impl Solve for LDLSolver {
    /// Solves the symmetric quasi-definite system using LDLᵀ.
    ///
    /// `Ap`/`Ai`/`Ax` are the FULL symmetric CSC triple (as produced by the
    /// LM flat layout); `b` is the RHS, overwritten with the solution.
    /// LDL itself picks out the permuted upper triangle and ignores the rest.
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        if let Some(s) = &self.symbolic {
            if !s.owns(Ap, Ai) {
                self.reset();
            }
        }

        if self.symbolic.is_none() {
            crate::timeit!(ldl_probe::SYM_NS, {
            let ap32: Vec<i32> = Ap.iter().map(|&x| x as i32).collect();
            let ai32: Vec<i32> = Ai.iter().map(|&x| x as i32).collect();
            unsafe {
                if self.inner.analyze(n as i32, &ap32, &ai32) != 0 {
                    return Err("LDL AMD ordering / symbolic failed");
                }
            }
            self.symbolic = Some(Symbolic {
                ap32,
                ai32,
                ap_ptr: Ap.as_ptr() as usize,
                ai_ptr: Ai.as_ptr() as usize,
            });
            });
        }

        let s = self.symbolic.as_ref().unwrap();
        crate::timeit!(ldl_probe::NUMERIC_NS, {
            let ret = unsafe { self.inner.factor(&s.ap32, &s.ai32, Ax) };
            if ret != 0 {
                // Zero pivot: violates quasi-definiteness (μ too small / bad data).
                return Err("LDL numeric factorization hit a zero pivot");
            }
            crate::probe_count!(ldl_probe::N_NUMERIC);
        });

        crate::timeit!(ldl_probe::SOLVE_NS, {
            unsafe { self.inner.solve_in_place(b) };
        });
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.symbolic = None;
    }
}

impl LDLSolver {
    /// Inertia of the last factorization: (positive, negative) pivots of D.
    /// For the LM augmented system this must equal (n_delta, n_residual).
    pub fn inertia(&self) -> (usize, usize) {
        self.inner.inertia()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small quasi-definite sanity: K = [μI Jᵀ; J −I] with a known dense
    /// solution cross-checked against nalgebra's dense LU.
    #[test]
    fn ldl_quasidefinite_small() {
        // 3x2 J, μ = 0.5 → K is 5x5 quasi-definite.
        let j: [[f64; 2]; 3] = [[2.0, 0.0], [1.0, -1.0], [0.0, 3.0]];
        let mu = 0.5f64;
        let nd = 2; // δ block
        let nr = 3; // residual block
        let n = nd + nr;
        // Full symmetric CSC of K = [μI Jᵀ; J −I].
        let mut dense = nalgebra::DMatrix::<f64>::zeros(n, n);
        for k in 0..nd {
            dense[(k, k)] = mu;
        }
        for k in 0..nr {
            dense[(nd + k, nd + k)] = -1.0;
        }
        for r in 0..nr {
            for c in 0..nd {
                dense[(c, nd + r)] = j[r][c];
                dense[(nd + r, c)] = j[r][c];
            }
        }
        // CSC from dense.
        let mut ap = vec![0usize; n + 1];
        let mut ai = Vec::new();
        let mut ax = Vec::new();
        for c in 0..n {
            for r in 0..n {
                if dense[(r, c)] != 0.0 {
                    ai.push(r);
                    ax.push(dense[(r, c)]);
                }
            }
            ap[c + 1] = ai.len();
        }

        let b0 = vec![1.0, -2.0, 3.0, 0.5, -1.0];
        let x_ref = dense.clone().lu().solve(&nalgebra::DVector::from_vec(b0.clone())).unwrap();

        let mut solver = LDLSolver::default();
        let mut b = b0.clone();
        let mut ap_m = ap.clone();
        let mut ai_m = ai.clone();
        let mut ax_m = ax.clone();
        solver.solve(&mut ap_m, &mut ai_m, &mut ax_m, &mut b, n).unwrap();

        let err = b.iter().zip(x_ref.iter()).fold(0.0f64, |m, (a, r)| m.max((a - r).abs()));
        println!("LDL small quasi-definite: max|Δx|={err:.3e} inertia={:?}", solver.inertia());
        assert!(err < 1e-12);
        assert_eq!(solver.inertia(), (nd, nr), "quasi-definite inertia must be (nδ, nr)");

        // Refactor with different μ (same pattern) must reuse symbolic.
        // δ-列内对角（行 c = 列 c）升序排在首位，即每列第一个 entry。
        let mut ax2 = ax.clone();
        for c in 0..nd {
            ax2[ap[c]] = 0.75;
        }
        let dense2 = {
            let mut d2 = dense.clone();
            for k in 0..nd {
                d2[(k, k)] = 0.75;
            }
            d2
        };
        let x_ref2 = dense2.lu().solve(&nalgebra::DVector::from_vec(b0.clone())).unwrap();
        let mut b2 = b0.clone();
        solver.solve(&mut ap_m, &mut ai_m, &mut ax2, &mut b2, n).unwrap();
        let err2 = b2.iter().zip(x_ref2.iter()).fold(0.0f64, |m, (a, r)| m.max((a - r).abs()));
        println!("LDL refactor new μ: max|Δx|={err2:.3e}");
        assert!(err2 < 1e-12);
    }
}
