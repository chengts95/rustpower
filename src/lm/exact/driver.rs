//! Phase 3 — Exact-LM driver on the flat augmented system (doc §1.3, §1.5).
//!
//! Per outer iteration the flat CSC is filled once (`fill_kkt_flat`) and
//! handed to a sparse direct solver (KLU in production, anything
//! implementing [`Solve`] in tests):
//!
//! ```text
//! ┌ μI + H(r)    Jᵀ ┐ ┌ δ ┐   ┌  0 ┐
//! │                 │ │   │ = │    │
//! └ J            −I ┘ └ s ┘   └ −r ┘
//! ```
//!
//! Row 2 gives `s = Jδ + r`; substituting into row 1 yields the exact-LM
//! normal step `(JᵀJ + H(r) + μI)δ = −Jᵀr` — the normal equations are never
//! formed (ext_ref `run_lm`: "生产路径走增广系统保持 1-hop").
//!
//! The μ inner loop only re-stamps the `aa`/`vv` diagonal slots
//! (`apply_mu_delta`) and re-factors — the main fill never re-runs for a
//! μ change. With `exact = false` the H region is kept at zero and the
//! driver degenerates to Gauss–Newton LM (the control group of the
//! convergence-window experiment).

use nalgebra::DVector;
use crate::lm::step_control::{LmOptions, TrialError, TrustRegion, polar_trial, predicted_reduction};
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use crate::lm::flat::{FlatLayout, fill_kkt_flat};
use crate::lm::kernels::{apply_mu_delta_weighted, fill_jt};
use crate::lm::pattern::KktPattern;
use crate::lm::residual::residual;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::solver::Solve;

pub struct LmDriver {
    pub pat: KktPattern,
    pub flat: FlatLayout,
    /// Specified injections `S_spec = P + iQ` for **all** buses (slack's and
    /// PV buses' Q are unused — slack enters through the physics channel).
    pub sbus: Vec<Complex64>,
    values: Vec<f64>,
    // Scratch (allocated once, reused every iteration).
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
}

/// Outcome of one LM run.
pub struct LmResult {
    pub iterations: usize,
    pub converged: bool,
    /// Final ‖r‖∞.
    pub res_inf: f64,
}

impl LmDriver {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize, sbus: Vec<Complex64>) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let flat = FlatLayout::build(&pat);
        let n_state = flat.n_state;
        let mut values = vec![0.0; flat.nnz_flat];
        flat.stamp_neg_i(&mut values);
        Self {
            pat,
            flat,
            sbus,
            values,
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
        }
    }

    /// Read-only view of the current flat values (inspection/testing).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// `g = Jᵀ·r` from the flat CSC: δ-column `c`'s **trailing** segment is
    /// the lower-left `J` block = J column `c` (rows `n+i` ↔ residual `i`).
    /// (The s-column's leading segment holds J **row** `c` — dotting that
    /// with `r` yields `J·r`, the wrong transpose.)
    fn jt_times_r(&mut self) {
        let (n, gp, ri) = (self.n_state, &self.flat.col_offsets, &self.flat.row_indices);
        for c in 0..n {
            let mut acc = 0.0;
            let l_c = (gp[c + 1] - gp[c]) / 2; // leading H segment length
            for p in gp[c] + l_c..gp[c + 1] {
                acc += self.values[p] * self.r[ri[p] - n];
            }
            self.g[c] = acc;
        }
    }

    /// One fill of the flat system at `v` (H region zero when `!exact`).
    fn fill(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64], exact: bool) {
        let nb = ybus.ncols();
        if !exact {
            // GN-LM: H must be zero **before every μ cycle** — μ accumulates
            // on the H diagonal, so re-zero here (nothing else writes H).
            self.zero_h();
        }
        for i in 0..nb {
            self.scalc[i] = v[i] * self.ibus[i].conj();
            let m = v[i].norm();
            self.vnorm[i] = if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) };
        }
        let cache = &self.pat.cache;
        let (npv, npq) = (self.n_act - self.npq, self.npq);
        let cs = &self.pat.graph.col_starts;
        if exact {
            fill_kkt_flat(
                ybus, &self.pat, &self.flat, v, &self.vnorm, &self.scalc, &self.r,
                &mut self.values,
            );
        } else {
            fill_jacobian_v4::<true>(
                ybus, v, &self.vnorm, &self.scalc,
                cs, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
                npv, npq, &mut self.values,
            );
            let ptr = self.values.as_mut_ptr();
            fill_jt::<true>(ybus, &self.pat, ptr, ptr);
        }
    }

    /// Zero the H region: δ-column c's leading segment, length derived from
    /// the global column pointers alone (`L_c = (gp[c+1]−gp[c])/2`).
    fn zero_h(&mut self) {
        let gp = &self.flat.col_offsets;
        for c in 0..self.n_state {
            let l_c = (gp[c + 1] - gp[c]) / 2;
            self.values[gp[c]..gp[c] + l_c].fill(0.0);
        }
    }

    /// Exact-LM (or GN-LM when `exact = false`) with default step control.
    /// See `solve_lm_with_options` for configurable settings.
    ///
    /// `v` is the flat-start voltage (slack fixed, PV magnitudes at spec) and
    /// is updated only after accepted steps, including when a later step fails.
    pub fn solve_lm<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        exact: bool,
        tol: f64,
        maxit: usize,
    ) -> LmResult {
        self.solve_lm_with_options(ybus, solver, v, exact, tol, maxit, &LmOptions::default())
    }

    /// Configurable damping and trial acceptance. Invalid options panic;
    /// callers accepting external settings can use `LmOptions::validate` first.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_lm_with_options<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        exact: bool,
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> LmResult {
        options.validate().expect("invalid LM options");
        let damping = options.damping_metric.prepare(ybus, self.n_act);
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let trace_voltage = std::env::var_os("RUSTPOWER_LM_TRACE_VOLTAGE").is_some();
        let mut mu = options.initial_mu;
        let mut region = TrustRegion::new(options.trust_region.as_ref());
        let mut res_inf;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
            }
            if res_inf < tol {
                return LmResult { iterations: it, converged: true, res_inf };
            }
            self.fill(ybus, v, exact);
            self.jt_times_r();

            // μ inner loop: only the diagonal slots move between tries.
            let mut mu_applied = 0.0;
            let mut accepted = false;
            for _ in 0..options.max_trials {
                apply_mu_delta_weighted::<true>(&self.pat, &mut self.values, mu - mu_applied,
                    |k| damping.weight(v, k, self.n_act));
                mu_applied = mu;

                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                let solve_ok = solver
                    .solve(
                        &mut self.flat.col_offsets,
                        &mut self.flat.row_indices,
                        &mut self.values,
                        &mut self.b,
                        2 * n,
                    )
                    .is_ok();
                let delta = &self.b[..n];
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        return LmResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                let step_norm_squared = damping.step_norm_squared(v, delta, self.n_act);
                if !region.allows(step_norm_squared) {
                    if debug {
                        eprintln!("it={it} tryμ={mu:.3e} rejected=TrustRadius step={:.3e} radius={:.3e}", step_norm_squared.sqrt(), region.radius);
                    }
                    if !options.increase_mu(&mut mu, options.mu_increase) {
                        return LmResult { iterations: it, converged: false, res_inf };
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
                        return LmResult { iterations: it, converged: false, res_inf };
                    }
                    continue;
                }

                // Trial residual goes to rt — r (the accepted point's) stays
                // intact for the next try's right-hand side.
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
                    eprintln!("trial_metrics it={it} mu={mu:.17e} step={:.17e} radius={:.17e} rho={rho:.17e} accepted={}",
                        step_norm_squared.sqrt(), region.radius,
                        rho.is_finite() && rho > options.acceptance_threshold);
                }
                if rho.is_finite() && rho > options.acceptance_threshold {
                    // Opt-in audit only; the reference solution never enters the solver.
                    if trace_voltage {
                        eprintln!("accepted_voltage {}", serde_json::json!({
                            "iteration": it + 1, "mu": mu, "radius": region.radius,
                            "rho": rho, "delta": delta,
                            "v_re": self.vt.iter().map(|v| v.re).collect::<Vec<_>>(),
                            "v_im": self.vt.iter().map(|v| v.im).collect::<Vec<_>>(),
                        }));
                    }
                    region.accept(rho, step_norm_squared, options.good_step_threshold);
                    v.copy_from_slice(&self.vt);
                    mu = options.accepted_mu(mu, rho);
                    accepted = true;
                    break;
                }
                region.reject();
                if !options.increase_mu(&mut mu, options.mu_increase) {
                    return LmResult { iterations: it, converged: false, res_inf };
                }
            }
            if !accepted {
                return LmResult { iterations: it, converged: false, res_inf };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        LmResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }
}

/// Drop-in LM power flow with the same contract as
/// [`crate::basic::newton_pf`] (`[PQ | PV | slack]` ordering, permuted
/// inputs/outputs): exact-LM on the flat augmented system with gain-ratio
/// μ adaptation. This is the ECS-plugin entry point (see
/// `ecs::lm_plugin`), mirroring `newton_pf_iwamoto`.
///
/// Note: the symbolic pattern + flat layout are rebuilt per call (one
/// `O(nnz)` pass); caching them as an ECS resource is a later optimization.
#[allow(clippy::too_many_arguments)]
pub fn newton_pf_lm<Solver: Solve>(
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
    let mut driver = LmDriver::build(ybus, npv, npq, sbus.iter().copied().collect());
    let mut v: Vec<Complex64> = v_init.iter().copied().collect();
    let res = driver.solve_lm(ybus, solver, &mut v, true, tol, maxit);
    let dv = DVector::from_vec(v);
    if res.converged {
        Ok((dv, res.iterations))
    } else {
        Err((
            format!("LM did not converge (res_inf = {:.3e})", res.res_inf),
            dv,
            res.iterations,
        ))
    }
}

/// Classical GN-LM lives on its own slim layout: see
/// [`super::gn_flat::newton_pf_gn`]. The fat layout's GN mode
/// (`solve_lm(exact = false)`) stays as the exact-LM control group.

#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use super::*;
    use crate::lm::residual::fixtures::ill_conditioned_case;
    use crate::basic::solver::KLUSolver;

    /// Slim 与 fat（GN 模式）逐元素对照：同一 J/Jᵀ 内容、−I 槽不被触碰、
    /// μ 槽在列头。（从 gn_flat 挪来：它是 exact 侧的对照测试。）
    #[test]
    fn slim_matches_fat_gn() {
        use crate::lm::gn_flat::GnDriver;
        let (ybus, n_pv, n_pq, _v_star, s_spec) = ill_conditioned_case();
        let mut fat = LmDriver::build(&ybus, n_pv, n_pq, s_spec.clone());
        let mut slim = GnDriver::build(&ybus, n_pv, n_pq, s_spec);
        let v: Vec<Complex64> = (0..ybus.ncols())
            .map(|k| Complex64::from_polar(1.0 + 0.004 * (k as f64), -0.01 * k as f64))
            .collect();

        // 两个 driver 各自填一次（fat 走 GN 分支，slim 走 block+copy）。
        let mut s1 = KLUSolver::default();
        let mut s2 = KLUSolver::default();
        let mut vf = v.clone();
        let mut vs = v.clone();
        // 只跑到第一次 fill 之后比对：用 1 次迭代上限。
        fat.solve_lm(&ybus, &mut s1, &mut vf, false, 0.0, 1);
        slim.solve_gn(&ybus, &mut s2, &mut vs, 0.0, 1);

        let (n, cs) = (slim.gn.n_state, &slim.pat.graph.col_starts);
        let (gp_s, gp_f) = (&slim.gn.col_offsets, &fat.flat.col_offsets);
        let slim_v = slim.values();
        let fat_v = fat.values();
        // δ-列：slim 的 J 段 == fat 的 J 段（fat 前段是 H=0 + μ）。
        for c in 0..n {
            let l_s = gp_s[c + 1] - gp_s[c] - 1;
            let l_f = (gp_f[c + 1] - gp_f[c]) / 2;
            assert_eq!(l_s, l_f, "column {c} J segment length mismatch");
            assert_eq!(
                &slim_v[gp_s[c] + 1..gp_s[c] + 1 + l_s],
                &fat_v[gp_f[c] + l_f..gp_f[c] + l_f + l_f],
                "column {c} J values mismatch"
            );
        }
        // s-列：Jᵀ 段逐元素相等；−I 槽都是 −1。
        for c in 0..n {
            let l = gp_s[n + c + 1] - gp_s[n + c] - 1;
            assert_eq!(
                &slim_v[gp_s[n + c]..gp_s[n + c] + l],
                &fat_v[gp_f[n + c]..gp_f[n + c] + l],
                "s-column {c} Jᵀ values mismatch"
            );
            assert_eq!(slim_v[gp_s[n + c + 1] - 1], -1.0);
        }
    }

}
