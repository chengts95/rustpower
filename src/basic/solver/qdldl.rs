//! QDLDL backend (Clarabel's native-Rust quasi-definite LDLᵀ) for the LM
//! augmented system — the pure-Rust counterpart of the SuiteSparse LDL
//! backend, same quasi-definite contract, zero bindgen.
//!
//! Clarabel's `QDLDLFactorisation` caches its symbolic phase internally
//! (permutation, elimination tree, L pattern, and an input→permuted-triu
//! gather map). Our [`Solve`] impl feeds the FULL symmetric CSC: values are
//! pushed per solve via `update_values` (all positions) + `refactor`.

use super::Solve;
use clarabel::algebra::CscMatrix;
use clarabel::qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder};

/// Performance instrumentation (see `crate::timeit!`): symbolic vs numeric
/// vs triangular-solve wall time and numeric-phase call count. Exists only
/// with the `probe` feature; compiled out entirely otherwise.
#[cfg(feature = "probe")]
pub mod qdldl_probe {
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
            "QDLDL breakdown: sym={:.3}ms numeric={:.3}ms({}) tri_solve={:.3}ms",
            g(&SYM_NS) as f64 / 1e6,
            g(&NUMERIC_NS) as f64 / 1e6,
            g(&N_NUMERIC),
            g(&SOLVE_NS) as f64 / 1e6,
        )
    }
}

/// Everything derived from one sparsity pattern. Kept behind a single
/// `Option` in the solver: `None` IS "no valid symbolic", there is no
/// second flag to disagree with it.
struct Symbolic {
    factor: QDLDLFactorisation<f64>,
    /// Gather map: triu position → index in the caller's full-symmetric `Ax`.
    map: Vec<usize>,
    /// Scratch triu values, refilled per solve and pushed wholesale.
    triu_vals: Vec<f64>,
    /// `0..nnz_triu`, the "update everything" index set for `update_values`.
    all_idx: Vec<usize>,
    /// Identity of the caller buffers this symbolic was built from. The
    /// validity check is pointer + length compare only (O(1), what the
    /// Newton hot loop hits every iteration). Correctness for *structural*
    /// changes is the event chain's job: topology / node-type changes always
    /// arrive with a `reset()` from `reset_solvers` before any pointer
    /// comparison would matter. Callers bypassing the event system must call
    /// `reset()` themselves when handing us a different matrix.
    ap_ptr: usize,
    ai_ptr: usize,
    ap_len: usize,
    ai_len: usize,
}

impl Symbolic {
    fn owns(&self, ap: &[usize], ai: &[usize]) -> bool {
        self.ap_ptr == ap.as_ptr() as usize
            && self.ai_ptr == ai.as_ptr() as usize
            && self.ap_len == ap.len()
            && self.ai_len == ai.len()
    }
}

#[derive(Default)]
pub struct QDLDLSolver {
    symbolic: Option<Symbolic>,
    /// Expected signs of D (+1 for the δ block, −1 for the residual block).
    /// `None` → first half +1 / second half −1 (the LM augmented layout).
    /// Clarabel's qdldl otherwise assumes ALL pivots positive and, with its
    /// default dynamic regularization enabled, *silently shifts* negative
    /// pivots — fatal for our −I block. We always pass explicit signs and
    /// disable regularization to keep the factorization faithful.
    dsigns: Option<Vec<i8>>,
}

impl QDLDLSolver {
    /// Explicit D-sign vector (e.g. for a non-half/half KKT layout).
    pub fn with_dsigns(dsigns: Vec<i8>) -> Self {
        Self { dsigns: Some(dsigns), ..Default::default() }
    }
}

#[allow(non_snake_case)]
impl Solve for QDLDLSolver {
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
                self.symbolic = None;
            }
        }

        if self.symbolic.is_none() {
            crate::timeit!(qdldl_probe::SYM_NS, {
            // QDLDL input convention: upper triangle only (i <= j), CSC.
            let mut up_p = vec![0usize; n + 1];
            let mut up_i = Vec::new();
            let mut map = Vec::new();
            for j in 0..n {
                for p in Ap[j]..Ap[j + 1] {
                    if Ai[p] <= j {
                        up_i.push(Ai[p]);
                        map.push(p);
                    }
                }
                up_p[j + 1] = up_i.len();
            }
            let triu_vals: Vec<f64> = map.iter().map(|&p| Ax[p]).collect();
            let mat = CscMatrix {
                m: n,
                n,
                colptr: up_p,
                rowval: up_i,
                nzval: triu_vals.clone(),
            };
            let dsigns = self
                .dsigns
                .clone()
                .unwrap_or_else(|| (0..n).map(|k| if k < n / 2 { 1i8 } else { -1i8 }).collect());
            let settings = QDLDLSettingsBuilder::default()
                .Dsigns(dsigns)
                .regularize_enable(false)
                .build()
                .map_err(|_| "QDLDL settings build failed")?;
            let factor = QDLDLFactorisation::new(&mat, Some(settings))
                .map_err(|_| "QDLDL symbolic/first factor failed")?;
            self.symbolic = Some(Symbolic {
                factor,
                all_idx: (0..triu_vals.len()).collect(),
                map,
                triu_vals,
                ap_ptr: Ap.as_ptr() as usize,
                ai_ptr: Ai.as_ptr() as usize,
                ap_len: Ap.len(),
                ai_len: Ai.len(),
            });
            });
        }

        let s = self.symbolic.as_mut().unwrap();
        crate::timeit!(qdldl_probe::NUMERIC_NS, {
            for (k, &src) in s.map.iter().enumerate() {
                s.triu_vals[k] = Ax[src];
            }
            s.factor.update_values(&s.all_idx, &s.triu_vals);
            s.factor
                .refactor()
                .map_err(|_| "QDLDL refactor failed (zero pivot: μ too small?)")?;
            crate::probe_count!(qdldl_probe::N_NUMERIC);
        });
        crate::timeit!(qdldl_probe::SOLVE_NS, {
            s.factor.solve(b);
        });
        Ok(())
    }

    fn reset(&mut self) {
        self.symbolic = None;
    }
}

impl QDLDLSolver {
    /// Positive pivots of D. For the LM augmented system the full inertia
    /// must be (n_delta positive, n_residual negative) — check with
    /// `positive_inertia() == n_delta`.
    pub fn positive_inertia(&self) -> Option<usize> {
        self.symbolic.as_ref().map(|s| s.factor.positive_inertia())
    }
}
