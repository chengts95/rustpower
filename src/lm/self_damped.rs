//! **Self-damped GN-LM**: residual-scaled damping `γ = ‖F(x_k)‖²` with full
//! steps and no step-control machinery.
//!
//! # Method
//!
//! Each iteration solves the same augmented system as [`super::gn_triu`]
//! (shared operator, buffers and solver backends — this file only owns the
//! loop), with the damping taken as a *function of the current state*
//! instead of a history-driven schedule:
//!
//! ```text
//! γ_k = ‖F(x_k)‖²      (F = power mismatch, merit f = ½‖F‖²)
//! δ   = −[γI Jᵀ; J −I]⁻¹ [0; F]
//! x_{k+1} = x_k + δ      (always the full step; nothing is ever rejected)
//! ```
//!
//! The damping is therefore **state-driven, not history-driven**: far from
//! the solution ‖F‖ is large and the step is heavily damped (short,
//! gradient-like); as ‖F‖ → 0 the damping vanishes *quadratically* and the
//! method recovers the plain Gauss-Newton step with its quadratic finish.
//! There is no μ schedule, no acceptance ratio and no trust-region radius —
//! and no rejected step ever wastes a factorization.
//!
//! # References
//!
//! - The `γ = ‖F‖²` choice is classical for power-system Levenberg-Marquardt:
//!   Rao ND, Tripathy SC, "Power system static state estimation by the
//!   Levenberg-Marquardt algorithm", IEEE Trans. PAS-99(2), 1980; see also
//!   the LM power-flow of Iwamoto & Tamura (1981) for the same family.
//! - The single-step form ([`GnTriuDriver::solve_self_damped`]) is the first
//!   step of the three-step composite (TSLM) of Chen L., "A high-order
//!   modified Levenberg-Marquardt method for systems of nonlinear equations"
//!   (as used in Derakhshandeh, Pourbagher, Kargar, "A novel fuzzy logic
//!   Levenberg-Marquardt method to solve the ill-conditioned power flow
//!   problem", IJEPES 99:299-308, 2018). The composite is implemented as
//!   [`GnTriuDriver::solve_self_damped_ts`]; the fuzzy layer of that paper
//!   is deliberately not adopted (no convergence content).
//!
//! # Deviations from the literature (deliberate, documented)
//!
//! - **Numerical safeguard**: the literature never faces a failed or
//!   non-positive trial because it assumes the damped system is well-posed.
//!   On real factorization failures (non-finite solve, or a polar trial
//!   driving |V| non-positive) we escalate `γ ×= 10` up to
//!   [`MAX_RETRIES`] times within the same iteration; beyond that the run
//!   is declared failed. This only shortens the step — no history is kept
//!   between iterations, so the method remains state-driven.
//! - **Stationarity stop**: besides `res_inf < tol` we stop when the merit
//!   stagnates (`|Δf| ≤ 1e-12·f` after an accepted step). That is
//!   the practical form of the literature's `‖JᵀF‖ = 0` test and lets the
//!   loop exit at least-squares points of infeasible specs instead of
//!   burning the remaining iterations. The threshold is relative to `f`
//!   itself — an absolute floor (e.g. `max(1, f)`) misreads the slow final
//!   approach on ill-conditioned cases as stagnation and kills runs that
//!   were still converging (observed on case6495rte α=1.05, stopped at
//!   f ≈ 2e-13 one step before the quadratic finish).

use crate::basic::solver::Solve;
use crate::lm::residual::residual_with_power;
use crate::lm::step_control::polar_trial;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::gn_triu::{GnTriuDriver, GnTriuResult};

/// γ escalations within one iteration after a failed/non-positive trial.
const MAX_RETRIES: usize = 6;
/// Merit stagnation tolerance for the stationarity stop.
const STAGNATION: f64 = 1e-12;

impl GnTriuDriver {
    /// Self-damped LM power flow (`γ = ‖F‖²`, full steps).
    /// Same contract as [`GnTriuDriver::solve_gn`]: `v` holds the start on
    /// entry and the terminal iterate on return.
    pub fn solve_self_damped<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnTriuResult {
        let n = self.n_state;
        let mut res_inf;
        let mut merit = f64::INFINITY;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = residual_with_power(
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
                return GnTriuResult { iterations: it, converged: true, res_inf };
            }
            // Stationarity (least-squares point): merit no longer decreases.
            // Relative to f: an absolute floor misjudges the slow final
            // approach as stagnation (see module docs).
            if (merit - f).abs() <= STAGNATION * f {
                return GnTriuResult { iterations: it, converged: false, res_inf };
            }
            merit = f;
            self.fill(ybus, v);

            // γ = ‖F‖² = 2f; escalated ×10 only after numerical trial failures.
            let mut gamma = 2.0 * f;
            let mut stepped = false;
            for _ in 0..=MAX_RETRIES {
                // μ slot = the lone entry of δ-column c (identity metric).
                for c in 0..n {
                    self.values[c] = gamma;
                }
                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.n_solves += 1;
                let ok = solver
                    .solve(
                        &mut self.triu.col_offsets,
                        &mut self.triu.row_indices,
                        &mut self.values,
                        &mut self.b,
                        2 * n,
                    )
                    .is_ok()
                    && self.b.iter().all(|x| x.is_finite());
                let trial_ok = ok
                    && polar_trial(
                        v,
                        &self.b[..n],
                        self.n_act,
                        self.npq,
                        true, // reject non-positive |V| trial: escalate γ instead
                        &mut self.vt,
                    )
                    .is_ok();
                if trial_ok {
                    v.copy_from_slice(&self.vt);
                    stepped = true;
                    break;
                }
                gamma *= 10.0;
            }
            if !stepped {
                return GnTriuResult { iterations: it, converged: false, res_inf };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) =
            crate::lm::residual::residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        GnTriuResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }

    /// Three-step composite self-damped LM (TSLM): one factorization per
    /// outer iteration, three corrective shots fired through it. Same
    /// contract as [`GnTriuDriver::solve_self_damped`].
    ///
    /// Shot 1 is exactly the single-step method (`y = x + δ₁`, including
    /// the γ-escalation safeguard). Shots 2–3 then reuse the *same*
    /// factorization with refreshed residuals:
    ///
    /// ```text
    /// δ₂ = −[γI Jᵀ; J −I]⁻¹ [0; F(y)]        z      = y + δ₂
    /// δ₃ = −[γI Jᵀ; J −I]⁻¹ [0; F(z)]        x_{k+1} = z + δ₃
    /// ```
    ///
    /// J and γ stay frozen at `x_k` across all three shots, so each extra
    /// shot costs one residual evaluation plus one triangular solve — no
    /// refactorization. This is the composite of Chen (fourth-order once
    /// the damping vanishes) referenced in the module docs, and is what
    /// makes the otherwise iteration-hungry self-damped method practical:
    /// the single-step variant needs hundreds of outer iterations on
    /// ill-conditioned cases precisely because every correction pays a
    /// full refactorization.
    ///
    /// Deliberate deviations, same spirit as the single-step version:
    /// composite shots allow non-positive |V| trials — they have no
    /// damping of their own to escalate (changing γ would break the shared
    /// factorization), and a non-physical intermediate is pulled back by
    /// the next outer iteration's state-driven γ. A failed shot aborts the
    /// remaining composite; the outer loop then refactors anyway. Backends
    /// without [`Solve::resolve`] degrade each extra shot to a full
    /// `solve` — numerically identical, just not cheaper.
    pub fn solve_self_damped_ts<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnTriuResult {
        let n = self.n_state;
        let mut res_inf;
        let mut merit = f64::INFINITY;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = residual_with_power(
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
                return GnTriuResult { iterations: it, converged: true, res_inf };
            }
            if (merit - f).abs() <= STAGNATION * f {
                return GnTriuResult { iterations: it, converged: false, res_inf };
            }
            merit = f;
            self.fill(ybus, v);

            // Shot 1: identical to the single-step method (γ = ‖F‖² = 2f,
            // escalated ×10 only after numerical trial failures).
            let mut gamma = 2.0 * f;
            let mut stepped = false;
            for _ in 0..=MAX_RETRIES {
                for c in 0..n {
                    self.values[c] = gamma;
                }
                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.n_solves += 1;
                let ok = solver
                    .solve(
                        &mut self.triu.col_offsets,
                        &mut self.triu.row_indices,
                        &mut self.values,
                        &mut self.b,
                        2 * n,
                    )
                    .is_ok()
                    && self.b.iter().all(|x| x.is_finite());
                let trial_ok = ok
                    && polar_trial(
                        v,
                        &self.b[..n],
                        self.n_act,
                        self.npq,
                        true,
                        &mut self.vt,
                    )
                    .is_ok();
                if trial_ok {
                    v.copy_from_slice(&self.vt);
                    stepped = true;
                    break;
                }
                gamma *= 10.0;
            }
            if !stepped {
                return GnTriuResult { iterations: it, converged: false, res_inf };
            }

            // Shots 2 & 3: refreshed residual, frozen J and γ, shared factor.
            for _ in 0..2 {
                {
                    let (n_act, npq) = (self.n_act, self.npq);
                    residual_with_power(
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
                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.n_solves += 1;
                // Prefer factorization reuse; backends without `resolve`
                // fall back to a full solve (same matrix, same result).
                let ok = solver.resolve(&mut self.b).is_ok()
                    || solver
                        .solve(
                            &mut self.triu.col_offsets,
                            &mut self.triu.row_indices,
                            &mut self.values,
                            &mut self.b,
                            2 * n,
                        )
                        .is_ok();
                let trial_ok = ok
                    && self.b.iter().all(|x| x.is_finite())
                    && polar_trial(
                        v,
                        &self.b[..n],
                        self.n_act,
                        self.npq,
                        false, // composite shots: no γ to escalate, see docs
                        &mut self.vt,
                    )
                    .is_ok();
                if !trial_ok {
                    break; // abort the remaining composite; outer loop refactors
                }
                v.copy_from_slice(&self.vt);
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) =
            crate::lm::residual::residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        GnTriuResult { iterations: maxit, converged: res_inf < tol, res_inf }
    }

    /// Cauchy 护送 steps 步：handoff 实验的**免分解**护送路径。
    /// 每步 = 3 次 Ybus matvec + 逐元素运算，无填充、无分解。
    /// 返回实际走成的步数（某步彻底失败则提前停止）。
    pub fn cauchy_escort(&mut self, ybus: &CscMatrix<Complex64>, v: &mut [Complex64], steps: usize) -> usize {
        let mut done = 0;
        for _ in 0..steps {
            let (n_act, npq) = (self.n_act, self.npq);
            let (_, f) = residual_with_power(
                ybus, &self.sbus, &mut self.ibus, &mut self.scalc, n_act, npq, v, &mut self.r,
            );
            if !self.cauchy_step(ybus, v, f) {
                break;
            }
            done += 1;
        }
        done
    }

    /// 一步 Cauchy 最速下降：δ = −σ·∇f，σ = ‖g‖²/‖Jg‖²（沿梯度方向对
    /// 二次模型 ½‖F+Jδ‖² 的精确线搜索）。g = JᵀF 与 Jg 均由 dSbus_dV 的
    /// Hermitian 形式解析得出（与 [`super::gn_triu::fill_jt_rows`] 同一 J
    /// 约定：|V| 列为绝对幅值导数），共 3 次 Ybus matvec，零分解。
    /// merit 不下降或试步非正则 σ/=4 重试（重试只需一次残差）。
    /// 入口要求 `f` = 当前 merit ½‖F‖²（调用方刚算过残差）。
    fn cauchy_step(&mut self, ybus: &CscMatrix<Complex64>, v: &mut [Complex64], f: f64) -> bool {
        let (n_act, npq, n) = (self.n_act, self.npq, self.n_state);
        self.merit_gradient(ybus, v);

        // Jg = dS_dVa·g_θ + dS_dVm·ĝ_V（复数，P 行取 Re、Q 行取 Im）。
        let (cp, ri, yv) = (ybus.col_offsets(), ybus.row_indices(), ybus.values());
        // scalc = Ybus·(V̄⊙ĝ_θ)
        for i in 0..v.len() {
            self.cw2[i] = if i < n_act {
                v[i].conj() * self.b[i]
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
        crate::basic::newtonpf::csc_matvec_complex(cp, ri, yv, &self.cw2, &mut self.scalc);
        // cw1 = Ybus·(Vn⊙ĝ_V)
        for i in 0..v.len() {
            self.cw2[i] = if i < npq {
                vnorm_at(v[i]) * self.b[n_act + i]
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
        crate::basic::newtonpf::csc_matvec_complex(cp, ri, yv, &self.cw2, &mut self.cw1);
        for k in 0..n_act {
            let g_v = if k < npq { self.b[n_act + k] } else { 0.0 };
            // j·V·Ībus·g_θ − j·V·conj(Y(V̄ĝ_θ)) + V·conj(Y(Vnĝ_V)) + Ībus·Vn·ĝ_V
            let t = Complex64::i() * v[k] * self.ibus[k].conj() * self.b[k]
                - Complex64::i() * v[k] * self.scalc[k].conj()
                + v[k] * self.cw1[k].conj()
                + self.ibus[k].conj() * vnorm_at(v[k]) * g_v;
            self.b[n + k] = t.re; // Jg_P
            if k < npq {
                self.b[n + n_act + k] = t.im; // Jg_Q
            }
        }

        let gg: f64 = self.b[..n].iter().map(|x| x * x).sum();
        let gjg: f64 = (0..n).map(|i| self.b[i] * self.b[n + i]).sum();
        if !gg.is_finite() || !gjg.is_finite() || gjg <= 0.0 || gg == 0.0 {
            return false; // JᵀF = 0（驻点）或数值异常：护送无路可走
        }
        let mut sigma = gg / gjg;
        for _ in 0..5 {
            // δ = −σ·g，盖在已用完的 Jg 槽位上。
            for i in 0..n {
                self.b[n + i] = -sigma * self.b[i];
            }
            if polar_trial(v, &self.b[n..], n_act, npq, true, &mut self.vt).is_err() {
                sigma /= 4.0;
                continue;
            }
            let (_, f_trial) = residual_with_power(
                ybus, &self.sbus, &mut self.ibus, &mut self.scalc, n_act, npq, &self.vt, &mut self.rt,
            );
            if f_trial < f {
                v.copy_from_slice(&self.vt);
                return true;
            }
            sigma /= 4.0;
        }
        false
    }

    /// ∇f = JᵀF，dSbus_dV 的 Hermitian 形式（matpower 约定的转置解析式）：
    /// ```text
    /// g_θ  = Re(−j·V̄⊙Ibus⊙mis + j·V⊙(Yᵀ(V̄⊙mis)))
    /// g_|V| = Re(Vn⊙(Yᵀ(V̄⊙mis)) + Ibus⊙V̄n⊙mis)      (仅 PQ 节点)
    /// ```
    /// 入口要求 self.ibus = Ibus、self.r = F（刚调过 residual_with_power）。
    /// 结果写入 self.b[..n_state]。成本：1 次 Yᵀ matvec + 逐元素运算。
    fn merit_gradient(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let (n_act, npq) = (self.n_act, self.npq);
        // mis = ΔP + iΔQ，逐节点构造：P 行只对 active 节点、Q 行只对 PQ
        // 节点存在，slack 无任何残差行——r 是 n_state 维的紧凑向量，
        // 节点 i ≥ n_act 时 r[i] 是别人的 Q 行，严禁直接索引。
        for i in 0..v.len() {
            let p = if i < n_act { self.r[i] } else { 0.0 };
            let q = if i < npq { self.r[n_act + i] } else { 0.0 };
            self.cw1[i] = Complex64::new(p, q);
        }
        // scalc = Ybusᵀ·(V̄⊙mis)：CSC 列与向量点积。
        for i in 0..v.len() {
            self.cw2[i] = v[i].conj() * self.cw1[i];
        }
        yt_matvec(ybus, &self.cw2, &mut self.scalc);
        for k in 0..n_act {
            let t1 = -Complex64::i() * v[k].conj() * self.ibus[k] * self.cw1[k];
            let t2 = Complex64::i() * v[k] * self.scalc[k];
            self.b[k] = (t1 + t2).re;
        }
        for k in 0..npq {
            let vn = vnorm_at(v[k]);
            self.b[n_act + k] =
                (vn * self.scalc[k] + self.ibus[k] * vn.conj() * self.cw1[k]).re;
        }
    }
}

#[inline(always)]
fn vnorm_at(vk: Complex64) -> Complex64 {
    let m = vk.norm();
    if m > 1e-12 { vk / m } else { Complex64::new(1.0, 0.0) }
}

/// (Yᵀ·x)：CSC 列 j 与 x 的点积（列主序下 Yᵀ 乘 = 列点积）。
fn yt_matvec(ybus: &CscMatrix<Complex64>, x: &[Complex64], out: &mut [Complex64]) {
    let (cp, ri, vals) = (ybus.col_offsets(), ybus.row_indices(), ybus.values());
    for j in 0..x.len() {
        let mut acc = Complex64::new(0.0, 0.0);
        for p in cp[j]..cp[j + 1] {
            acc += vals[p] * x[ri[p]];
        }
        out[j] = acc;
    }
}

// ─── tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 节点：bus0 PQ、bus1 PV、bus2 slack（与 step_control 测试同一 Ybus）。
    fn toy() -> (CscMatrix<Complex64>, Vec<Complex64>, Vec<Complex64>) {
        let y = CscMatrix::try_from_csc_data(
            3,
            3,
            vec![0, 3, 6, 9],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            [
                (2.0, 3.0), (-1.0, -1.0), (-1.0, -2.0),
                (-1.0, -1.0), (4.0, -2.0), (-3.0, 3.0),
                (-1.0, -2.0), (-3.0, 3.0), (4.0, -1.0),
            ]
            .map(|(re, im)| Complex64::new(re, im))
            .to_vec(),
        )
        .unwrap();
        let s = vec![
            Complex64::new(0.5, 0.2),
            Complex64::new(-0.3, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let v = vec![
            Complex64::from_polar(0.95, 0.05),
            Complex64::from_polar(1.02, -0.03),
            Complex64::from_polar(1.0, 0.0),
        ];
        (y, s, v)
    }

    /// merit_gradient 的解析 ∇f 必须等于 f = ½‖F‖² 的有限差分梯度
    ///（Hermitian 公式极易推错符号/共轭，本测试就是裁决者）。
    #[test]
    fn merit_gradient_matches_finite_differences() {
        let (y, s, v) = toy();
        let mut d = GnTriuDriver::build(&y, 1, 1, s);
        let (n_act, n) = (d.n_act, d.n_state);
        {
            let (na, nq) = (d.n_act, d.npq);
            residual_with_power(&y, &d.sbus, &mut d.ibus, &mut d.scalc, na, nq, &v, &mut d.r);
        }
        d.merit_gradient(&y, &v);
        let g: Vec<f64> = d.b[..n].to_vec();

        let eps = 1e-6;
        for c in 0..n {
            let k = if c < n_act { c } else { c - n_act };
            let (mag, ang) = (v[k].norm(), v[k].arg());
            let (mut vp, mut vm) = (v.clone(), v.clone());
            if c < n_act {
                vp[k] = Complex64::from_polar(mag, ang + eps);
                vm[k] = Complex64::from_polar(mag, ang - eps);
            } else {
                vp[k] = Complex64::from_polar(mag + eps, ang);
                vm[k] = Complex64::from_polar(mag - eps, ang);
            }
            let fp = {
                let (na, nq) = (d.n_act, d.npq);
                residual_with_power(&y, &d.sbus, &mut d.ibus, &mut d.scalc, na, nq, &vp, &mut d.r).1
            };
            let fm = {
                let (na, nq) = (d.n_act, d.npq);
                residual_with_power(&y, &d.sbus, &mut d.ibus, &mut d.scalc, na, nq, &vm, &mut d.r).1
            };
            let fd = (fp - fm) / (2.0 * eps);
            let tol = 1e-5 * g[c].abs().max(1.0);
            assert!(
                (g[c] - fd).abs() < tol,
                "component {c}: analytic {} vs finite-diff {}",
                g[c],
                fd
            );
        }
    }
}
