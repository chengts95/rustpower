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
//! The baseline fills reuse the **existing block-mode kernels untouched**
//! (`fill_jacobian_v4::<false>` + `fill_jt::<false>` into compact block
//! arrays) plus one `memcpy` per column into the slim CSC. The extra
//! `O(nnz)` copy per sweep is the price for not adding a third view mode to
//! the shared kernels; it is negligible next to one sparse factorization.
//! `build_operator` fills both blocks directly from borrowed pattern views,
//! using V3 arithmetic reuse and live Y values. It allocates neither J nor Jᵀ
//! scratch blocks. Both modes use the same LM loop and damping policy.
//!
//! The driver loop mirrors the exact-LM driver's `solve_lm` (kept in the
//! uncommitted `exact/` folder) minus the H machinery (no `zero_h`, μ slot
//! = column head); the gain-ratio μ rules are identical (ext_ref `run_lm`).

use crate::lm::step_control::{
    LmOptions, TrialError, TrustRegion, polar_trial, predicted_reduction,
};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::kernels::fill_jt;
use super::pattern::KktPattern;
use super::residual::residual;
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

// 装配方式及其专属工作区；迭代控制不依赖装配方式。
enum Assembly {
    BlockCopy {
        j_block: Vec<f64>,
        jt_block: Vec<f64>,
        vnorm: Vec<Complex64>,
    },
    Direct {
        inv_vmag: Vec<f64>,
    },
}

/// Classical GN-LM driver on the slim layout.
pub struct GnDriver {
    pub pat: KktPattern,
    pub gn: GnFlatLayout,
    pub sbus: Vec<Complex64>,
    assembly: Assembly,
    values: Vec<f64>,
    // Scratch (allocated once).
    ibus: Vec<Complex64>,
    scalc: Vec<Complex64>,
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
    /// μ对角及右端准备的累计时间。
    pub prof_mu_ns: u64,
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

    pub fn build(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        Self::build_with_assembly(ybus, n_pv, n_pq, sbus, |pat| Assembly::BlockCopy {
            j_block: vec![0.0; pat.graph.nnz],
            jt_block: vec![0.0; pat.graph.nnz],
            vnorm: vec![Complex64::new(1.0, 0.0); ybus.ncols()],
        })
    }

    /// Use the borrowed Jacobian block operator; retain the original driver as baseline.
    pub fn build_operator(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        Self::build_with_assembly(ybus, n_pv, n_pq, sbus, |_| Assembly::Direct {
            inv_vmag: vec![0.0; ybus.ncols()],
        })
    }

    fn build_with_assembly(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
        make_assembly: impl FnOnce(&KktPattern) -> Assembly,
    ) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let assembly = make_assembly(&pat);
        let gn = GnFlatLayout::build(&pat);
        let n_state = gn.n_state;
        let mut values = vec![0.0; gn.nnz_slim];
        gn.stamp_neg_i(&mut values);
        Self {
            pat,
            gn,
            sbus,
            assembly,
            values,
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            scalc: vec![Complex64::new(0.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            g: vec![0.0; n_state],
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            n_act: n_pv + n_pq,
            npq: n_pq,
            n_state,
            prof_fill_ns: 0,
            prof_mu_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    pub fn reset_prof(&mut self) {
        self.prof_fill_ns = 0;
        self.prof_mu_ns = 0;
        self.prof_solve_ns = 0;
        self.n_solves = 0;
    }

    /// 填充 J 和 Jᵀ：原路径先填紧凑块再拷贝，直接算子写入最终 CSC。
    /// 两者复用残差计算得到的 scalc，不修改 μ 和 −I。
    fn fill(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t_prof = std::time::Instant::now();
        match &mut self.assembly {
            Assembly::Direct { inv_vmag } => {
                for (inv, voltage) in inv_vmag.iter_mut().zip(v) {
                    *inv = 1.0 / voltage.norm();
                }
                use crate::basic::jacobian_operator::{JacobianBlock, JacobianOperator};
                let cache = &self.pat.cache;
                let op = JacobianOperator {
                    ybus,
                    v,
                    inv_vmag,
                    scalc: &self.scalc,
                    pq_ends: cache.pq_ends(),
                    active_ends: cache.active_ends(),
                    diag_ptrs: cache.diag_ptrs(),
                    mirror: cache.y_trans(),
                    npq: self.npq,
                    npv: self.n_act - self.npq,
                };
                op.fill::<false, false>(
                    JacobianBlock {
                        column_starts: &self.gn.col_offsets,
                        base: 0,
                        prefix: 1,
                    },
                    &mut self.values,
                );
                op.fill::<false, true>(
                    JacobianBlock {
                        column_starts: &self.gn.col_offsets[self.n_state..],
                        base: 0,
                        prefix: 0,
                    },
                    &mut self.values,
                );
            }
            Assembly::BlockCopy {
                j_block,
                jt_block,
                vnorm,
            } => {
                let nb = ybus.ncols();
                for i in 0..nb {
                    let m = v[i].norm();
                    vnorm[i] = if m > 1e-12 {
                        v[i] / m
                    } else {
                        Complex64::new(1.0, 0.0)
                    };
                }
                let cache = &self.pat.cache;
                let (npv, npq) = (self.n_act - self.npq, self.npq);
                let cs = &self.pat.graph.col_starts;
                fill_jacobian_v4::<false>(
                    ybus,
                    v,
                    vnorm,
                    &self.scalc,
                    cs,
                    cache.pq_ends(),
                    cache.active_ends(),
                    cache.diag_ptrs(),
                    npv,
                    npq,
                    j_block,
                );
                fill_jt::<false>(ybus, &self.pat, j_block.as_ptr(), jt_block.as_mut_ptr());

                let (n, gp) = (self.n_state, &self.gn.col_offsets);
                // δ-column c: J segment right after the μ slot.
                for c in 0..n {
                    let l = gp[c + 1] - gp[c] - 1;
                    self.values[gp[c] + 1..gp[c] + 1 + l]
                        .copy_from_slice(&j_block[cs[c]..cs[c] + l]);
                }
                // s-column c: Jᵀ segment (the −I tail is write-once, untouched).
                for c in 0..n {
                    let l = gp[n + c + 1] - gp[n + c] - 1;
                    self.values[gp[n + c]..gp[n + c] + l]
                        .copy_from_slice(&jt_block[cs[c]..cs[c] + l]);
                }
            }
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

    /// Classical GN-LM using default damping settings and positive trial
    /// magnitudes. See `solve_gn_with_options` for configurable settings.
    pub fn solve_gn<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnResult {
        self.solve_gn_with_options(ybus, solver, v, tol, maxit, &LmOptions::default())
    }

    /// Configurable damping and trial acceptance. Invalid options panic;
    /// callers accepting external settings can use `LmOptions::validate` first.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_gn_with_options<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> GnResult {
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
                (res_inf, f) = super::residual::residual_with_power(
                    ybus,
                    &self.sbus,
                    &mut self.ibus,
                    &mut self.scalc,
                    n_act,
                    npq,
                    v,
                    &mut self.r,
                );
            }
            if res_inf < tol {
                return GnResult {
                    iterations: it,
                    converged: true,
                    res_inf,
                };
            }
            self.fill(ybus, v);
            self.jt_times_r();

            let mut accepted = false;
            for _ in 0..options.max_trials {
                let t_mu = std::time::Instant::now();
                // μ diagonal slot = head of every δ-column: set absolute μ.
                {
                    let gp = &self.gn.col_offsets;
                    for c in 0..n {
                        self.values[gp[c]] = mu * damping.weight(v, c, self.n_act);
                    }
                }

                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.prof_mu_ns += t_mu.elapsed().as_nanos() as u64;
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
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        return GnResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                let step_norm_squared = damping.step_norm_squared(v, delta, self.n_act);
                if !region.allows(step_norm_squared) {
                    if debug {
                        eprintln!(
                            "it={it} tryμ={mu:.3e} rejected=TrustRadius step={:.3e} radius={:.3e}",
                            step_norm_squared.sqrt(),
                            region.radius
                        );
                    }
                    if !options.increase_mu(&mut mu, options.mu_increase) {
                        return GnResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                if let Err(reason) = polar_trial(
                    v,
                    delta,
                    self.n_act,
                    self.npq,
                    options.reject_nonpositive_voltage,
                    &mut self.vt,
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
                        return GnResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                let (_, f_new) = {
                    let (n_act, npq) = (self.n_act, self.npq);
                    residual(
                        ybus,
                        &self.sbus,
                        &mut self.ibus,
                        n_act,
                        npq,
                        &self.vt,
                        &mut self.rt,
                    )
                };
                let pred = predicted_reduction(&self.g, delta, mu, step_norm_squared);
                let rho = if pred.is_finite() && pred > 0.0 && f_new.is_finite() {
                    (f - f_new) / pred
                } else {
                    -1.0
                };
                if debug {
                    eprintln!(
                        "it={it} tryμ={mu:.3e} res={res_inf:.3e} f={f:.4e} f_new={f_new:.4e} pred={pred:.4e} ρ={rho:.4}"
                    );
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
                    return GnResult {
                        iterations: it,
                        converged: false,
                        res_inf,
                    };
                }
            }
            if !accepted {
                return GnResult {
                    iterations: it,
                    converged: false,
                    res_inf,
                };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        GnResult {
            iterations: maxit,
            converged: res_inf < tol,
            res_inf,
        }
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

#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use super::*;
    use crate::basic::solver::KLUSolver;
    use crate::lm::residual::fixtures::load_ieee39_mat;

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
        let (v_nr, it_nr) = newton_pf(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_nr,
            None,
        )
        .expect("NR should converge");
        let mut s_gn = KLUSolver::default();
        let (v_gn, it_gn) = newton_pf_gn(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_gn,
        )
        .expect("GN-LM should converge");
        let mut s_lm = KLUSolver::default();
        let (v_lm, it_lm) = newton_pf_lm(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_lm,
        )
        .expect("exact-LM should converge");

        let diff = |a: &DVector<Complex64>, b: &DVector<Complex64>| {
            a.iter()
                .zip(b.iter())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
        };
        println!("严格容差 1e-12: NR it={it_nr} | GN-LM it={it_gn} | exact-LM it={it_lm}");
        println!(
            "max|ΔV|: GN vs NR = {:.3e} | exact vs NR = {:.3e} | exact vs GN = {:.3e}",
            diff(&v_gn, &v_nr),
            diff(&v_lm, &v_nr),
            diff(&v_lm, &v_gn)
        );
        assert!(
            diff(&v_gn, &v_nr) < 1e-9,
            "GN-LM and NR disagree at tight tolerance"
        );
        assert!(
            diff(&v_lm, &v_nr) < 1e-9,
            "exact-LM and NR disagree at tight tolerance"
        );
    }
}
