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
//! - The single-step form here is the first step of the three-step composite
//!   (TSLM) of Chen L., "A high-order modified Levenberg-Marquardt method for
//!   systems of nonlinear equations" (as used in Derakhshandeh, Pourbagher,
//!   Kargar, "A novel fuzzy logic Levenberg-Marquardt method to solve the
//!   ill-conditioned power flow problem", IJEPES 99:299-308, 2018). The
//!   composite variant may follow; the fuzzy layer of that paper is
//!   deliberately not adopted (no convergence content).
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
//!   stagnates (`|Δf| ≤ 1e-12·max(1, f)` after an accepted step). That is
//!   the practical form of the literature's `‖JᵀF‖ = 0` test and lets the
//!   loop exit at least-squares points of infeasible specs instead of
//!   burning the remaining iterations.

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
            if (merit - f).abs() <= STAGNATION * f.max(1.0) {
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
}
