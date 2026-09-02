//! **AUG-COO** — the ablation floor for the augmented GN-LM system.
//!
//! Same mathematics as [`super::gn_flat::GnDriver`] (identical μ policy,
//! identical residual, identical J values from the shared offset kernel),
//! but every linear solve is done the stranger's way:
//!
//! 1. push the whole `[μI Jᵀ; J −I]` into a COO triplet buffer,
//! 2. `CscMatrix::from(&coo)` — the sort/dedup tax, every single μ try,
//! 3. hand a **fresh** `QDLDLSolver` the matrix — symbolic analysis +
//!    numeric factorization from scratch, zero reuse.
//!
//! The gap between this driver and `GnDriver` is exactly what the paper
//! claims: direct CSC fill + symbolic-once/numeric-many. Kept in its own
//! module so the whole baseline can be dropped by deleting the folder.

use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

use super::super::pattern::KktPattern;
use super::super::residual::residual;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::solver::{QDLDLSolver, Solve};

/// AUG-COO driver: naive COO assembly + fresh symbolic per solve.
pub struct AugCooDriver {
    pub pat: KktPattern,
    pub sbus: Vec<Complex64>,
    /// J values in the shared block-CSC layout (kernel output, per sweep).
    j_vals: Vec<f64>,
    /// Column starts **with the nnz sentinel** (graph.col_starts lacks it).
    j_cols: Vec<usize>,
    // Scratch (allocated once).
    ibus: Vec<Complex64>,
    scalc: Vec<Complex64>,
    vnorm: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    vt: Vec<Complex64>,
    n_act: usize,
    npq: usize,
    n_state: usize,
    // Profiling (ns), same convention as `normal_eq::NeDriver`.
    /// J fill (shared kernel — identical cost in every path).
    pub prof_fill_ns: u64,
    /// COO push + sort/convert + triple extraction (the naive-assembly tax).
    pub prof_coo_ns: u64,
    /// Fresh-solver solves: symbolic + numeric + triangular, per μ try.
    pub prof_solve_ns: u64,
    /// Number of linear solves (outer iterations + μ retries).
    pub n_solves: u64,
}

/// Outcome of one run (same shape as `GnResult` / `NeResult`).
pub struct AugCooResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl AugCooDriver {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize, sbus: Vec<Complex64>) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let n_state = pat.graph.n_cols;
        let nnz = pat.graph.nnz;
        let mut j_cols = pat.graph.col_starts.clone();
        j_cols.push(nnz); // sentinel
        Self {
            pat,
            sbus,
            j_vals: vec![0.0; nnz],
            j_cols,
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            scalc: vec![Complex64::new(0.0, 0.0); nb],
            vnorm: vec![Complex64::new(1.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            g: vec![0.0; n_state],
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            n_act: n_pv + n_pq,
            npq: n_pq,
            n_state,
            prof_fill_ns: 0,
            prof_coo_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    /// J fill via the shared offset kernel (identical to every other path).
    fn fill_j(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t = std::time::Instant::now();
        let nb = ybus.ncols();
        for i in 0..nb {
            self.scalc[i] = v[i] * self.ibus[i].conj();
            let m = v[i].norm();
            self.vnorm[i] = if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) };
        }
        let cache = &self.pat.cache;
        fill_jacobian_v4::<false>(
            ybus, v, &self.vnorm, &self.scalc,
            &self.pat.graph.col_starts, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
            self.n_act - self.npq, self.npq, &mut self.j_vals,
        );
        self.prof_fill_ns += t.elapsed().as_nanos() as u64;
    }

    /// The stranger's assembly: push `[μI Jᵀ; J −I]` as COO triplets, then
    /// pay the sort/convert. Jᵀ entries are re-pushed from the same J values
    /// (a naive implementation owns no transpose kernel).
    fn coo_assemble(&mut self, mu: f64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let (cs, rows) = (&self.j_cols, &self.pat.graph.row_indices);
        let mut coo = CooMatrix::new(2 * n, 2 * n);
        // μI block.
        for c in 0..n {
            coo.push(c, c, mu);
        }
        // J block at rows n+r, and Jᵀ block mirrored at (c, n+r).
        for c in 0..n {
            for p in cs[c]..cs[c + 1] {
                let (r, jv) = (rows[p], self.j_vals[p]);
                coo.push(n + r, c, jv);
                coo.push(c, n + r, jv);
            }
        }
        // −I block.
        for c in 0..n {
            coo.push(n + c, n + c, -1.0);
        }
        let csc = CscMatrix::from(&coo);
        let triple = (
            csc.col_offsets().to_vec(),
            csc.row_indices().to_vec(),
            csc.values().to_vec(),
        );
        self.prof_coo_ns += t.elapsed().as_nanos() as u64;
        triple
    }

    /// One fresh-solver solve of the assembled system. The stranger pays
    /// symbolic + numeric + triangular every time (no pattern caching).
    fn fresh_solve(&mut self, ap: &mut [usize], ai: &mut [usize], ax: &mut [f64]) -> bool {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let mut solver = QDLDLSolver::default();
        self.b[..n].fill(0.0);
        for i in 0..n {
            self.b[n + i] = -self.r[i];
        }
        let ok = solver.solve(ap, ai, ax, &mut self.b, 2 * n).is_ok();
        self.prof_solve_ns += t.elapsed().as_nanos() as u64;
        self.n_solves += 1;
        ok
    }

    /// Classical LM loop — μ rules byte-identical to
    /// [`super::super::gn_flat::GnDriver::solve_gn`] so the only difference
    /// under test is how the linear system gets assembled and solved.
    pub fn solve_aug_coo(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> AugCooResult {
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let mut mu = 1e-2f64;
        let mut res_inf;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
            }
            if res_inf < tol {
                return AugCooResult { iterations: it, converged: true, res_inf };
            }
            self.fill_j(ybus, v);

            // g = Jᵀ r from the J block (same formula as the NE driver).
            for c in 0..n {
                let mut acc = 0.0;
                for p in self.j_cols[c]..self.j_cols[c + 1] {
                    acc += self.j_vals[p] * self.r[self.pat.graph.row_indices[p]];
                }
                self.g[c] = acc;
            }

            let mut accepted = false;
            for _ in 0..30 {
                let (mut ap, mut ai, mut ax) = self.coo_assemble(mu);
                let solve_ok = self.fresh_solve(&mut ap, &mut ai, &mut ax);
                let delta: Vec<f64> = self.b[..n].to_vec();
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    mu *= 10.0;
                    if mu > 1e12 {
                        return AugCooResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                // Polar trial update (identical to the GN driver): |V|
                // states live at n_act + k inside the n_state vector.
                self.vt.copy_from_slice(v);
                for k in 0..self.n_act {
                    let mut mag = self.vt[k].norm();
                    let ang = self.vt[k].arg() + delta[k];
                    if k < self.npq {
                        mag += delta[self.n_act + k];
                    }
                    self.vt[k] = Complex64::from_polar(mag, ang);
                }
                if self.vt.iter().any(|x| !x.re.is_finite() || !x.im.is_finite()) {
                    mu *= 10.0;
                    if mu > 1e12 {
                        return AugCooResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                let (_, f_new) = {
                    let (n_act, npq) = (self.n_act, self.npq);
                    residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, &self.vt, &mut self.rt)
                };
                let pred: f64 = -0.5
                    * self.g.iter().zip(delta.iter()).map(|(g, d)| g * d).sum::<f64>();
                let rho = if pred > 0.0 { (f - f_new) / pred } else { -1.0 };
                if debug {
                    eprintln!("it={it} tryμ={mu:.3e} res={res_inf:.3e} f={f:.4e} f_new={f_new:.4e} pred={pred:.4e} ρ={rho:.4}");
                }
                if rho > 1e-4 {
                    v.copy_from_slice(&self.vt);
                    if rho > 0.75 {
                        mu = (mu / 3.0).max(1e-12);
                    }
                    accepted = true;
                    break;
                }
                mu *= 2.0;
                if mu > 1e12 {
                    return AugCooResult { iterations: it, converged: false, res_inf };
                }
            }
            if !accepted {
                return AugCooResult { iterations: it, converged: false, res_inf };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        AugCooResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }
}
