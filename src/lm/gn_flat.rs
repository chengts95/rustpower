//! Classical Gauss–Newton LM on a **slim** flat layout (no Hessian term).
//!
//! The exact-LM [`super::flat::FlatLayout`] carries `[μI+H Jᵀ; J −I]`;
//! without H the δ-columns shrink from `[H col | J col]` to `[μ diag | J col]`:
//!
//! ```text
//! δ-column c     (0..n)  : [μ diag | J col c ]   rows [c | graph rows + n]
//! s-column n + c (0..n)  : [Jᵀ col c | −I diag]  rows [graph rows | n + c]
//! ```
//!
//! Column pointers stay affine in the shared graph pattern:
//! `gp[c] = c + cs[c]`, `gp[n+c] = n + nnz + cs[c] + c` — every position is
//! re-derived from the column's own base and the Ybus structure, and the μ
//! diagonal slot is simply the **first entry of every δ-column**
//! (`nnz_slim = 2·nnz + 2n` vs fat `3·nnz + n`).
//!
//! Fills reuse the **existing block-mode kernels untouched**
//! (`fill_jacobian_v4::<false>` + `fill_jt::<false>` into compact block
//! arrays) plus one `memcpy` per column into the slim CSC. The extra
//! `O(nnz)` copy per sweep is the price for not adding a third view mode to
//! the shared kernels; it is negligible next to one sparse factorization.
//! An in-place slim fill can replace the copy later if profiling says so.
//!
//! The driver loop mirrors the exact-LM driver's `solve_lm` (kept in the
//! uncommitted `exact/` folder) minus the H machinery (no `zero_h`, μ slot
//! = column head); the gain-ratio μ rules are identical (ext_ref `run_lm`).

use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::residual::residual;
use super::kernels::fill_jt;
use super::pattern::KktPattern;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::solver::Solve;

/// Slim global CSC of `[μI Jᵀ; J −I]`, symbolic part (only the triple a
/// direct solver needs, same discipline as [`super::flat::FlatLayout`]).
pub struct GnFlatLayout {
    pub n_state: usize,
    /// `2·nnz + 2·n_state`.
    pub nnz_slim: usize,
    pub col_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
}

impl GnFlatLayout {
    pub fn build(pat: &KktPattern) -> Self {
        let cache = &pat.cache;
        let n = cache.n_active() + cache.n_pq();
        let nnz = pat.graph.nnz;
        let cs = &pat.graph.col_starts;

        let mut col_offsets = Vec::with_capacity(2 * n + 1);
        let mut row_indices = Vec::with_capacity(2 * nnz + 2 * n);

        // δ-columns: [μ diag (row c) | J col c (rows shifted by n)].
        for c in 0..n {
            col_offsets.push(c + cs[c]);
            row_indices.push(c);
            row_indices.extend(pat.graph.col_rows(c).iter().map(|r| r + n));
        }
        // s-columns: [Jᵀ segment | −I entry].
        for c in 0..n {
            col_offsets.push(n + nnz + cs[c] + c);
            row_indices.extend_from_slice(pat.graph.col_rows(c));
            row_indices.push(n + c);
        }
        col_offsets.push(2 * nnz + 2 * n);

        Self {
            n_state: n,
            nnz_slim: 2 * nnz + 2 * n,
            col_offsets,
            row_indices,
        }
    }

    /// Stamp the constant `−I` block (last entry of every s-column), once.
    pub fn stamp_neg_i(&self, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.nnz_slim);
        for c in 0..self.n_state {
            values[self.col_offsets[self.n_state + c + 1] - 1] = -1.0;
        }
    }
}

/// Classical GN-LM driver on the slim layout.
pub struct GnDriver {
    pub pat: KktPattern,
    pub gn: GnFlatLayout,
    pub sbus: Vec<Complex64>,
    values: Vec<f64>,
    // Compact block views (fill output, copied into the slim CSC per sweep).
    j_block: Vec<f64>,
    jt_block: Vec<f64>,
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
    // Profiling (ns), same convention as `normal_eq::NeDriver`: the slim
    // fill (J + Jᵀ kernels + column copies) and the solver calls.
    pub prof_fill_ns: u64,
    pub prof_solve_ns: u64,
    pub n_solves: u64,
}

/// Outcome of one GN-LM run (same shape as the exact-LM driver's result).
pub struct GnResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl GnDriver {
    /// Read-only view of the slim CSC values (test cross-checks).
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize, sbus: Vec<Complex64>) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let gn = GnFlatLayout::build(&pat);
        let n_state = gn.n_state;
        let nnz = pat.graph.nnz;
        let mut values = vec![0.0; gn.nnz_slim];
        gn.stamp_neg_i(&mut values);
        Self {
            pat,
            gn,
            sbus,
            values,
            j_block: vec![0.0; nnz],
            jt_block: vec![0.0; nnz],
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
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    pub fn reset_prof(&mut self) {
        self.prof_fill_ns = 0;
        self.prof_solve_ns = 0;
        self.n_solves = 0;
    }

    /// One slim fill at `v`: block J + block Jᵀ (existing kernels), then one
    /// column-wise copy into the slim CSC. The μ diagonal slots and the −I
    /// slots are not touched (μ is stamped by the inner loop).
    fn fill(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t_prof = std::time::Instant::now();
        let nb = ybus.ncols();
        for i in 0..nb {
            self.scalc[i] = v[i] * self.ibus[i].conj();
            let m = v[i].norm();
            self.vnorm[i] = if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) };
        }
        let cache = &self.pat.cache;
        let (npv, npq) = (self.n_act - self.npq, self.npq);
        let cs = &self.pat.graph.col_starts;
        fill_jacobian_v4::<false>(
            ybus, v, &self.vnorm, &self.scalc,
            cs, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
            npv, npq, &mut self.j_block,
        );
        fill_jt::<false>(ybus, &self.pat, self.j_block.as_ptr(), self.jt_block.as_mut_ptr());

        let (n, gp) = (self.n_state, &self.gn.col_offsets);
        // δ-column c: J segment right after the μ slot.
        for c in 0..n {
            let l = gp[c + 1] - gp[c] - 1;
            self.values[gp[c] + 1..gp[c] + 1 + l].copy_from_slice(&self.j_block[cs[c]..cs[c] + l]);
        }
        // s-column c: Jᵀ segment (the −I tail is write-once, untouched).
        for c in 0..n {
            let l = gp[n + c + 1] - gp[n + c] - 1;
            self.values[gp[n + c]..gp[n + c] + l].copy_from_slice(&self.jt_block[cs[c]..cs[c] + l]);
        }
        self.prof_fill_ns += t_prof.elapsed().as_nanos() as u64;
    }

    /// `g = Jᵀ·r` from the slim CSC: δ-column c's trailing segment is J
    /// column c (rows `n+i` ↔ residual `i`). Same formula as the (fixed)
    /// fat path, with `L_c = 1`.
    fn jt_times_r(&mut self) {
        let (n, gp, ri) = (self.n_state, &self.gn.col_offsets, &self.gn.row_indices);
        for c in 0..n {
            let mut acc = 0.0;
            for p in gp[c] + 1..gp[c + 1] {
                acc += self.values[p] * self.r[ri[p] - n];
            }
            self.g[c] = acc;
        }
    }

    /// Classical GN-LM with gain-ratio μ adaptation (rules identical to
    /// the exact-LM driver's `solve_lm`: accept ρ > 1e-4; ρ > 0.75 →
    /// μ/3; reject → μ×2; non-finite → μ×10; μ > 1e12 → stall). The μ slot
    /// is the head of every δ-column, so a μ change is `n` adds.
    pub fn solve_gn<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnResult {
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
                return GnResult { iterations: it, converged: true, res_inf };
            }
            self.fill(ybus, v);
            self.jt_times_r();

            let mut accepted = false;
            for _ in 0..30 {
                // μ diagonal slot = head of every δ-column: set absolute μ.
                {
                    let gp = &self.gn.col_offsets;
                    for c in 0..n {
                        self.values[gp[c]] = mu;
                    }
                }

                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                let t_solve = std::time::Instant::now();
                let solve_ok = solver
                    .solve(
                        &mut self.gn.col_offsets,
                        &mut self.gn.row_indices,
                        &mut self.values,
                        &mut self.b,
                        2 * n,
                    )
                    .is_ok();
                self.prof_solve_ns += t_solve.elapsed().as_nanos() as u64;
                self.n_solves += 1;
                let delta = &self.b[..n];
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    mu *= 10.0;
                    if mu > 1e12 {
                        return GnResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                // Polar trial update (identical to the exact-LM driver).
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
                        return GnResult { iterations: it, converged: false, res_inf };
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
                    return GnResult { iterations: it, converged: false, res_inf };
                }
            }
            if !accepted {
                return GnResult { iterations: it, converged: false, res_inf };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        GnResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }
}

/// Classical GN-LM power flow with the same contract as
/// [`crate::basic::newton_pf`] — the ECS-plugin entry point. Slim-layout
/// counterpart of the exact-LM driver's `newton_pf_lm`.
#[allow(clippy::too_many_arguments)]
pub fn newton_pf_gn<Solver: Solve>(
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
    let mut driver = GnDriver::build(ybus, npv, npq, sbus.iter().copied().collect());
    let mut v: Vec<Complex64> = v_init.iter().copied().collect();
    let res = driver.solve_gn(ybus, solver, &mut v, tol, maxit);
    let dv = DVector::from_vec(v);
    if res.converged {
        Ok((dv, res.iterations))
    } else {
        Err((
            format!("GN-LM did not converge (res_inf = {:.3e})", res.res_inf),
            dv,
            res.iterations,
        ))
    }
}

#[cfg(all(test, feature = "klu"))]
mod tests {
    use super::*;
    use crate::lm::residual::fixtures::load_ieee39_mat;
    use crate::basic::solver::KLUSolver;

    /// IEEE39 收敛性（与 fat-GN 同数学，同迭代数）。
    #[test]
    fn slim_gn_ieee39() {
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();
        let mut driver = GnDriver::build(ybus, npv, npq, sbus);
        let mut solver = KLUSolver::default();
        let mut v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let res = driver.solve_gn(ybus, &mut solver, &mut v, 1e-8, 100);
        println!(
            "IEEE39 slim-GN: converged={} it={} res={:.2e} | nnz slim={} fat={}",
            res.converged,
            res.iterations,
            res.res_inf,
            driver.gn.nnz_slim,
            crate::lm::flat::FlatLayout::build(&driver.pat).nnz_flat
        );
        assert!(res.converged);
    }

    /// 非病态标准算例：NR / GN-LM / exact-LM 三家在严格容差（1e-12）下
    /// 解的逐点一致性——验证 LM 两侧与生产 NR 算的是同一个东西。
    #[test]
    fn three_way_matches_nr_tight() {
        use crate::basic::newtonpf::newton_pf;
        use crate::lm::exact::driver::newton_pf_lm;
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus = DVector::from_vec(mat.s_bus.iter().copied().collect::<Vec<_>>());
        let v_init = DVector::from_vec(mat.v_bus_init.iter().copied().collect::<Vec<_>>());

        let mut s_nr = KLUSolver::default();
        let (v_nr, it_nr) =
            newton_pf(ybus, &sbus, &v_init, npv, npq, Some(1e-12), Some(100), &mut s_nr)
                .expect("NR should converge");
        let mut s_gn = KLUSolver::default();
        let (v_gn, it_gn) =
            newton_pf_gn(ybus, &sbus, &v_init, npv, npq, Some(1e-12), Some(100), &mut s_gn)
                .expect("GN-LM should converge");
        let mut s_lm = KLUSolver::default();
        let (v_lm, it_lm) =
            newton_pf_lm(ybus, &sbus, &v_init, npv, npq, Some(1e-12), Some(100), &mut s_lm)
                .expect("exact-LM should converge");

        let diff = |a: &DVector<Complex64>, b: &DVector<Complex64>| {
            a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
        };
        println!(
            "严格容差 1e-12: NR it={it_nr} | GN-LM it={it_gn} | exact-LM it={it_lm}"
        );
        println!(
            "max|ΔV|: GN vs NR = {:.3e} | exact vs NR = {:.3e} | exact vs GN = {:.3e}",
            diff(&v_gn, &v_nr),
            diff(&v_lm, &v_nr),
            diff(&v_lm, &v_gn)
        );
        assert!(diff(&v_gn, &v_nr) < 1e-9, "GN-LM and NR disagree at tight tolerance");
        assert!(diff(&v_lm, &v_nr) < 1e-9, "exact-LM and NR disagree at tight tolerance");
    }
}
