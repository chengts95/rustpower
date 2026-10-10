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
}
