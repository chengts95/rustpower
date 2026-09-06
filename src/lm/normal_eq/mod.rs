//! The normal-equations LM path: solve the LM step as
//!
//! ```text
//! (JᵀJ + μI) δ = −Jᵀ r
//! ```
//!
//! 两种乘积算法共用驱动器和 V4 Jacobian 填充：
//! * `dumb_mode = false`：首次建立乘积结构，之后用列交集计算数值。
//! * `dumb_mode = true`：每轮使用通用稀疏乘法重建乘积。
//! 两者均复用传给线性求解器的矩阵缓冲区，避免重复符号分析。
//! 性能由 `lm::comparison` 对照测试衡量，不预设哪种更快。

use nalgebra::DVector;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use num_complex::Complex64;

use super::pattern::KktPattern;
use super::residual::residual;
use super::step_control::{LmOptions, TrialError, TrustRegion, polar_trial, predicted_reduction};
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::solver::Solve;

/// Normal-equations LM driver, two modes:
///
/// * `dumb_mode = true` — the ablation floor: nalgebra spgemm redoes the
///   JᵀJ pattern **every** iteration (write-it-like-a-stranger baseline);
/// * `dumb_mode = false` (default) — the same architecture discipline as the
///   augmented path: pattern analyzed once, then each sweep is numeric-only
///   (two-pointer column dot products, zero extra storage).
pub struct NeDriver {
    pub pat: KktPattern,
    pub sbus: Vec<Complex64>,
    /// Ablation switch: redo the JᵀJ symbolic pattern every iteration.
    pub dumb_mode: bool,
    // J as a standalone CSC (pattern from the shared graph, values per sweep).
    j_cols: Vec<usize>,
    j_rows: Vec<usize>,
    j_vals: Vec<f64>,
    // A = JᵀJ cached symbolic pattern + values (the product matrix triple).
    a_cols: Vec<usize>,
    a_rows: Vec<usize>,
    a_vals: Vec<f64>,
    /// Diagonal position within each column of A (cached; kills the μ scan).
    diag_pos: Vec<usize>,
    a_symbolic_done: bool,
    diag0: Vec<f64>,
    b: Vec<f64>,
    // Scratch (allocated once).
    ibus: Vec<Complex64>,
    scalc: Vec<Complex64>,
    vnorm: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    g: Vec<f64>,
    vt: Vec<Complex64>,
    n_act: usize,
    npq: usize,
    n_state: usize,
    // Profiling (ns), reset via `reset_prof`.
    pub prof_fill_ns: u64,
    pub prof_spgemm_ns: u64,
    pub prof_numeric_ns: u64,
    pub prof_mu_ns: u64,
    pub prof_solve_ns: u64,
    pub n_solves: u64,
}

/// Outcome of one NE-LM run (same shape as the GN driver's result).
pub struct NeResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl NeDriver {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize, sbus: Vec<Complex64>) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let n_state = pat.graph.n_cols;
        let mut j_cols = pat.graph.col_starts.clone();
        j_cols.push(pat.graph.nnz);
        let j_rows = pat.graph.row_indices.clone();
        let nnz = pat.graph.nnz;
        Self {
            pat,
            sbus,
            dumb_mode: false,
            j_cols,
            j_rows,
            j_vals: vec![0.0; nnz],
            a_cols: Vec::new(),
            a_rows: Vec::new(),
            a_vals: Vec::new(),
            diag_pos: Vec::new(),
            a_symbolic_done: false,
            diag0: vec![0.0; n_state],
            b: vec![0.0; n_state],
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            scalc: vec![Complex64::new(0.0, 0.0); nb],
            vnorm: vec![Complex64::new(1.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            g: vec![0.0; n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            n_act: n_pv + n_pq,
            npq: n_pq,
            n_state,
            prof_fill_ns: 0,
            prof_spgemm_ns: 0,
            prof_numeric_ns: 0,
            prof_mu_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    pub fn reset_prof(&mut self) {
        self.prof_fill_ns = 0;
        self.prof_spgemm_ns = 0;
        self.prof_numeric_ns = 0;
        self.prof_mu_ns = 0;
        self.prof_solve_ns = 0;
        self.n_solves = 0;
    }

    /// J fill (existing offset kernel) — shared with every other path.
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

    /// Symbolic phase: the JᵀJ pattern via nalgebra's generic spgemm
    /// (CSC→CSR round-trips included). In `dumb_mode` this runs every
    /// iteration; otherwise exactly once.
    fn jtj_symbolic(&mut self) {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let j_csc = CscMatrix::try_from_csc_data(
            n, n, self.j_cols.clone(), self.j_rows.clone(), self.j_vals.clone(),
        )
        .expect("J pattern is a valid CSC");
        // transpose() 保持格式（Csc→Csc），需要再转一次格式到 CSR。
        let jt_csr = CsrMatrix::from(&j_csc.transpose()); // Jᵀ as CSR
        let j_csr = CsrMatrix::from(&j_csc); // J as CSR
        let prod = &jt_csr * &j_csr; // spgemm: pattern + numeric
        let a_csc = CscMatrix::from(&prod);
        if !self.a_symbolic_done {
            self.a_cols = a_csc.col_offsets().to_vec();
            self.a_rows = a_csc.row_indices().to_vec();
            self.a_vals = a_csc.values().to_vec();
        } else {
            // 通用乘法可以分配临时矩阵，但传给solver的结构数组保持不动。
            // 固定J模式应产生相同的乘积模式；检查后只复制数值。
            assert!(self.a_cols == a_csc.col_offsets()
                && self.a_rows == a_csc.row_indices(), "JᵀJ pattern changed during solve");
            self.a_vals.copy_from_slice(a_csc.values());
        }
        if self.diag_pos.is_empty() {
            // Diagonal position per column (Ybus diag ⇒ JᵀJ diag always present).
            self.diag_pos = (0..n)
                .map(|c| {
                    (self.a_cols[c]..self.a_cols[c + 1])
                        .find(|&p| self.a_rows[p] == c)
                        .expect("JᵀJ diagonal always present")
                })
                .collect();
        }
        self.a_symbolic_done = true;
        self.prof_spgemm_ns += t.elapsed().as_nanos() as u64;
    }

    /// Numeric phase: A[i,j] = dot(J col i, J col j) by two-pointer merge on
    /// the shared sorted rows. Zero extra storage, pattern untouched.
    fn jtj_numeric(&mut self) {
        let t = std::time::Instant::now();
        let (jc, jr, jv) = (&self.j_cols, &self.j_rows, &self.j_vals);
        for c in 0..self.n_state {
            for p in self.a_cols[c]..self.a_cols[c + 1] {
                let i = self.a_rows[p];
                // dot(J col i, J col c): both row lists ascending.
                let (mut x, mut y) = (jc[i], jc[c]);
                let (xe, ye) = (jc[i + 1], jc[c + 1]);
                let mut acc = 0.0;
                while x < xe && y < ye {
                    let (rx, ry) = (jr[x], jr[y]);
                    if rx == ry {
                        acc += jv[x] * jv[y];
                        x += 1;
                        y += 1;
                    } else if rx < ry {
                        x += 1;
                    } else {
                        y += 1;
                    }
                }
                self.a_vals[p] = acc;
            }
        }
        self.prof_numeric_ns += t.elapsed().as_nanos() as u64;
    }

    /// One A = JᵀJ refresh at `v` (fill + symbolic-as-mode + numeric).
    fn jtj(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        self.fill_j(ybus, v);
        if self.dumb_mode || !self.a_symbolic_done {
            self.jtj_symbolic();
            if self.dumb_mode {
                // Dumb mode: the spgemm already produced values; keep them.
                return;
            }
        }
        self.jtj_numeric();
    }

    /// Classical LM loop with gain-ratio μ adaptation — rules byte-identical
    /// to [`super::gn_flat::GnDriver::solve_gn`] so the comparison is purely
    /// "how the linear system gets solved", not the μ policy.
    pub fn solve_ne<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> NeResult {
        self.solve_ne_with_options(ybus, solver, v, tol, maxit, &LmOptions::default())
    }

    /// Configurable damping and trial acceptance. Invalid options panic;
    /// callers accepting external settings can use `LmOptions::validate` first.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_ne_with_options<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> NeResult {
        options.validate().expect("invalid LM options");
        let damping = options.damping_metric.prepare(ybus, self.n_act);
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let mut mu = options.initial_mu;
        let mut region = TrustRegion::new(options.trust_region.as_ref());
        let mut res_inf;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = residual(ybus, &self.sbus, &mut self.ibus,
                    n_act, npq, v, &mut self.r);
            }
            if res_inf < tol {
                return NeResult { iterations: it, converged: true, res_inf };
            }
            self.jtj(ybus, v);
            for c in 0..n {
                self.diag0[c] = self.a_vals[self.diag_pos[c]];
                self.g[c] = (self.j_cols[c]..self.j_cols[c + 1])
                    .map(|p| self.j_vals[p] * self.r[self.j_rows[p]]).sum();
            }

            let mut accepted = false;
            for _ in 0..options.max_trials {
                let t_mu = std::time::Instant::now();
                for c in 0..n {
                    self.a_vals[self.diag_pos[c]] = self.diag0[c]
                        + mu * damping.weight(v, c, self.n_act);
                    self.b[c] = -self.g[c];
                }
                self.prof_mu_ns += t_mu.elapsed().as_nanos() as u64;
                let t_solve = std::time::Instant::now();
                let solve_ok = solver
                    .solve(
                        &mut self.a_cols,
                        &mut self.a_rows,
                        &mut self.a_vals,
                        &mut self.b,
                        n,
                    )
                    .is_ok();
                self.prof_solve_ns += t_solve.elapsed().as_nanos() as u64;
                self.n_solves += 1;
                let delta = &self.b[..n];
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        return NeResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                let step_norm_squared = damping.step_norm_squared(v, delta, self.n_act);
                if !region.allows(step_norm_squared) {
                    if debug {
                        eprintln!("it={it} tryμ={mu:.3e} rejected=TrustRadius step={:.3e} radius={:.3e}", step_norm_squared.sqrt(), region.radius);
                    }
                    if !options.increase_mu(&mut mu, options.mu_increase) {
                        return NeResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                if let Err(reason) = polar_trial(
                    v, delta, self.n_act, self.npq, options.reject_nonpositive_voltage, &mut self.vt,
                ) {
                    if debug {
                        eprintln!("it={it} tryμ={mu:.3e} rejected={reason:?}");
                    }
                    region.reject();
                    let factor = match reason {
                        TrialError::NonFinite => options.failed_step_increase,
                        TrialError::NonPositiveMagnitude { .. } => options.mu_increase,
                    };
                    if !options.increase_mu(&mut mu, factor) {
                        return NeResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                let (_, f_new) = {
                    let (n_act, npq) = (self.n_act, self.npq);
                    residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, &self.vt, &mut self.rt)
                };
                let pred = predicted_reduction(&self.g, delta, mu, step_norm_squared);
                let rho = if pred.is_finite() && pred > 0.0 && f_new.is_finite() {
                    (f - f_new) / pred
                } else { -1.0 };
                if debug {
                    eprintln!("it={it} tryμ={mu:.3e} res={res_inf:.3e} f={f:.4e} f_new={f_new:.4e} pred={pred:.4e} ρ={rho:.4}");
                }
                if rho.is_finite() && rho > options.acceptance_threshold {
                    region.accept(rho, step_norm_squared, options.good_step_threshold);
                    v.copy_from_slice(&self.vt);
                    mu = options.accepted_mu(mu, rho);
                    accepted = true;
                    break;
                }
                region.reject();
                if !options.increase_mu(&mut mu, options.mu_increase) {
                    return NeResult { iterations: it, converged: false, res_inf };
                }
            }
            if !accepted {
                return NeResult { iterations: it, converged: false, res_inf };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        NeResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }
}

/// Normal-equations LM power flow with the same contract as
/// [`crate::basic::newton_pf`]. The dumb baseline entry point.
#[allow(clippy::too_many_arguments)]
pub fn newton_pf_ne<Solver: Solve>(
    ybus: &CscMatrix<Complex64>,
    sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let tol = tolerance.unwrap_or(1e-6);
    let maxit = max_iter.unwrap_or(100);
    let mut driver = NeDriver::build(ybus, npv, npq, sbus.iter().copied().collect());
    let mut v: Vec<Complex64> = v_init.iter().copied().collect();
    let res = driver.solve_ne(ybus, solver, &mut v, tol, maxit);
    let out = DVector::from_vec(v);
    if res.converged {
        Ok((out, res.iterations))
    } else {
        Err(("normal-equations LM failed to converge".into(), out, res.iterations))
    }
}

// All tests in this module need the SuiteSparse LDL backend (and its
// fixtures need `klu`); gate at module level so the imports stay valid.
#[cfg(all(test, any(feature = "klu", feature = "klu_dyn"), feature = "ldl"))]
mod tests;
