//! Continuation power flow (CPF) using predictor-corrector tracing of a PV curve.
//!
//! # Method
//!
//! Tangent prediction and perpendicular-intersection correction follow the default
//! parameterization used by MATPOWER `runcpf`:
//!
//! - V. Ajjarapu, C. Christy, "The continuation power flow: a tool for steady
//!   state voltage stability analysis", IEEE Trans. Power Systems, 7(1), 1992.
//! - H.-D. Chiang et al., "CPFLOW: a practical tool for tracing power system
//!   steady-state stationary behavior due to load and generation variations",
//!   IEEE Trans. Power Systems, 10(2), 1995.
//!
//! # Formulation
//!
//! The loading parameter lambda scales a fixed injection direction:
//! `F(x, lambda) = s_calc(x) - lambda * d = 0`, where `d` is the reduced base
//! injection (P at all active buses and Q at PQ buses), so `dF/dlambda = -d` is
//! constant. The state `x = [theta(active); |V|(PQ)]` uses the same
//! `[PQ | PV | slack]` ordering as `newton_pf` and `fill_jacobian_v4`, allowing
//! direct reuse of the V4 Jacobian assembly and sparsity pattern.
//!
//! - Prediction solves `z = J^-1 d`, normalizes `[z; 1]`, and advances along the
//!   directed tangent. A directed `t_lambda` changing from positive to negative
//!   identifies passage through the nose.
//! - Correction solves `F = 0` on the hyperplane through the predicted point with
//!   the tangent as its normal. Each iteration factors `J y = F` and reuses that
//!   factorization for `J z = d` when the backend supports `Solve::resolve`.
//!
//! Unlike repeated independent power-flow solves, continuation follows the solution
//! manifold. This avoids interpreting Newton's shrinking convergence region near a
//! nose as proof that no power-flow solution exists.
//!
//! Costs are reported as factorizations and resolves rather than outer iterations.
//! Every tangent solve also counts as one factorization.

use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use std::time::{Duration, Instant};

use crate::basic::jacobian_cache::JacobianCache;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::newtonpf::{csc_matvec_and_scalc, fill_f_from_scalc, newton_pf};
use crate::basic::solver::Solve;

/// Predictor-corrector settings for continuation power flow.
#[derive(Clone, Copy, Debug)]
pub struct CpfOptions {
    /// Maximum absolute power mismatch accepted at a curve point.
    pub tol: f64,
    pub step0: f64,
    pub step_max: f64,
    pub step_min: f64,
    pub step_grow: f64,
    pub step_shrink: f64,
    pub max_corr_iter: usize,
    /// Maximum accepted points, including the base point.
    pub max_points: usize,
    /// Maximum iterations for the initial ordinary Newton solve.
    pub base_max_iter: usize,
}

impl Default for CpfOptions {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            step0: 0.2,
            step_max: 1.0,
            step_min: 1e-3,
            step_grow: 1.3,
            step_shrink: 0.5,
            max_corr_iter: 10,
            max_points: 300,
            base_max_iter: 100,
        }
    }
}

impl CpfOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err("CPF tolerance must be finite and positive".into());
        }
        if ![self.step0, self.step_max, self.step_min]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
            || self.step_min > self.step0
            || self.step0 > self.step_max
        {
            return Err("CPF steps must satisfy 0 < step_min <= step0 <= step_max".into());
        }
        if !self.step_grow.is_finite() || self.step_grow < 1.0 {
            return Err("CPF step_grow must be finite and at least 1".into());
        }
        if !self.step_shrink.is_finite() || !(0.0..1.0).contains(&self.step_shrink) {
            return Err("CPF step_shrink must be finite and in (0, 1)".into());
        }
        if self.max_corr_iter == 0 || self.max_points == 0 || self.base_max_iter == 0 {
            return Err("CPF iteration and point limits must be positive".into());
        }
        Ok(())
    }
}

/// One accepted point on the PV curve, in solver bus order at the low-level API.
#[derive(Clone, Debug)]
pub struct CpfPoint {
    pub lambda: f64,
    pub voltage: DVector<Complex64>,
    pub corrector_iterations: usize,
}

/// Three-point quadratic estimate of a detected nose point.
#[derive(Clone, Debug)]
pub struct CpfNose {
    pub lambda: f64,
    pub voltage: DVector<Complex64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpfTermination {
    NosePassed,
    MaxPointsReached,
    StepBelowMinimum,
}

impl CpfTermination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NosePassed => "nose-passed",
            Self::MaxPointsReached => "max-points-reached",
            Self::StepBelowMinimum => "step-below-minimum",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpfStats {
    pub factorizations: usize,
    pub resolves: usize,
    pub elapsed: Duration,
}

/// Result of tracing a fixed-direction, fixed-bus-partition PV curve.
#[derive(Clone, Debug)]
pub struct CpfReport {
    pub points: Vec<CpfPoint>,
    /// Present only after the directed tangent crosses through `t_lambda = 0`.
    pub nose: Option<CpfNose>,
    pub termination: CpfTermination,
    pub stats: CpfStats,
}

impl CpfReport {
    pub fn passed_nose(&self) -> bool {
        self.termination == CpfTermination::NosePassed
    }

    pub fn lambda_max(&self) -> f64 {
        self.points
            .iter()
            .map(|point| point.lambda)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn stop_reason(&self) -> &'static str {
        self.termination.as_str()
    }
}

/// Stateful CPF solver owning its Jacobian pattern, numerical workspaces, and
/// linear solver. The borrowed Ybus structure remains fixed throughout tracing.
pub struct CpfSolver<S: Solve> {
    npv: usize,
    npq: usize,
    n_active: usize,
    n_state: usize,
    /// Base injections at lambda = 1 in `[PQ | PV | slack]` order.
    s_base: DVector<Complex64>,
    /// Initial voltage supplying PV magnitudes and Slack references.
    v_spec: DVector<Complex64>,
    /// Reduced injection direction: P at active buses followed by Q at PQ buses.
    d: Vec<f64>,
    pat: JacobianCache,
    j_values: Vec<f64>,
    v_cur: DVector<Complex64>,
    v_m: DVector<f64>,
    v_a: DVector<f64>,
    v_norm: DVector<Complex64>,
    ibus: DVector<Complex64>,
    s_calc: DVector<Complex64>,
    s_lam: DVector<Complex64>,
    /// Corrector right-hand side F; overwritten by `y = J^-1 F` during solve.
    fbuf: Vec<f64>,
    /// Second right-hand side for `z = J^-1 d`.
    zbuf: Vec<f64>,
    solver: S,
}

impl<S: Solve> CpfSolver<S> {
    pub fn new(
        ybus: &CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
        s_base: DVector<Complex64>,
        v_spec: DVector<Complex64>,
        solver: S,
    ) -> Self {
        let n_active = npv + npq;
        let n_state = npv + 2 * npq;
        let n_bus = s_base.len();
        let pat =
            JacobianCache::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);
        let j_values = vec![0.0; pat.j_row_indices.len()];
        let mut d = vec![0.0; n_state];
        for i in 0..n_active {
            d[i] = s_base[i].re;
        }
        for j in 0..npq {
            d[n_active + j] = s_base[j].im;
        }
        Self {
            npv,
            npq,
            n_active,
            n_state,
            s_base,
            v_spec,
            d,
            pat,
            j_values,
            v_cur: DVector::zeros(n_bus),
            v_m: DVector::zeros(n_bus),
            v_a: DVector::zeros(n_bus),
            v_norm: DVector::zeros(n_bus),
            ibus: DVector::zeros(n_bus),
            s_calc: DVector::zeros(n_bus),
            s_lam: DVector::zeros(n_bus),
            fbuf: vec![0.0; n_state],
            zbuf: vec![0.0; n_state],
            solver,
        }
    }

    /// Rebuild complex voltages from the current magnitudes and angles.
    fn refresh_v(&mut self) {
        for i in 0..self.v_cur.len() {
            self.v_cur[i] = Complex64::from_polar(self.v_m[i], self.v_a[i]);
            self.v_norm[i] = Complex64::from_polar(1.0, self.v_a[i]);
        }
    }

    /// Current state `[theta(active); |V|(PQ)]`.
    fn state_x(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.n_state];
        x[..self.n_active].copy_from_slice(&self.v_a.as_slice()[..self.n_active]);
        x[self.n_active..].copy_from_slice(&self.v_m.as_slice()[..self.npq]);
        x
    }

    /// Fill Jacobian values at the current point using the fixed pattern.
    fn fill_jacobian(&mut self, ybus: &CscMatrix<Complex64>) {
        fill_jacobian_v4::<false>(
            ybus,
            self.v_cur.as_slice(),
            self.v_norm.as_slice(),
            self.s_calc.as_slice(),
            &self.pat.j_col_ptrs,
            &self.pat.pq_ends,
            &self.pat.active_ends,
            &self.pat.diag_ptrs,
            self.npv,
            self.npq,
            &mut self.j_values,
        );
    }

    /// Return the infinity norm of `F = s_calc - lambda * S_base`.
    fn mismatch(&mut self, ybus: &CscMatrix<Complex64>, lambda: f64) -> f64 {
        csc_matvec_and_scalc(
            ybus.col_offsets(),
            ybus.row_indices(),
            ybus.values(),
            self.v_cur.as_slice(),
            self.ibus.as_mut_slice(),
            self.s_calc.as_mut_slice(),
        );
        for i in 0..self.s_lam.len() {
            self.s_lam[i] = self.s_base[i] * lambda;
        }
        fill_f_from_scalc::<false>(
            self.s_calc.as_slice(),
            self.s_lam.as_slice(),
            self.npq,
            self.n_active,
            &mut self.fbuf,
        )
    }

    /// Compute the directed tangent `[t_x; t_lambda]`; each call factors J once.
    fn tangent(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        factorizations: &mut usize,
    ) -> Result<Vec<f64>, String> {
        self.fill_jacobian(ybus);
        self.zbuf.copy_from_slice(&self.d);
        self.solver
            .solve(
                &mut self.pat.j_col_ptrs,
                &mut self.pat.j_row_indices,
                &mut self.j_values,
                &mut self.zbuf,
                self.n_state,
            )
            .map_err(|e| format!("tangent solve failed: {e}"))?;
        *factorizations += 1;
        let norm = (self.zbuf.iter().map(|e| e * e).sum::<f64>() + 1.0).sqrt();
        let mut t = Vec::with_capacity(self.n_state + 1);
        t.extend(self.zbuf.iter().map(|e| e / norm));
        t.push(1.0 / norm);
        Ok(t)
    }

    /// Solve on the hyperplane through the predicted point with normal `t`.
    /// Return the corrector iteration count, or `Err` for non-convergence,
    /// non-finite states, negative lambda, or a failed linear solve.
    #[allow(clippy::too_many_arguments)]
    fn corrector(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        t: &[f64],
        x_pred: &[f64],
        lambda_pred: f64,
        lambda: &mut f64,
        opts: &CpfOptions,
        factorizations: &mut usize,
        resolves: &mut usize,
    ) -> Result<usize, ()> {
        let n_act = self.n_active;
        let n_st = self.n_state;
        let (tx, tl) = (&t[..n_st], t[n_st]);

        for it in 0..opts.max_corr_iter {
            let norm = self.mismatch(ybus, *lambda);
            if !norm.is_finite() || *lambda < 0.0 {
                return Err(());
            }
            if norm < opts.tol {
                return Ok(it);
            }

            // Plane residual p = t_x^T (x - x_pred) + t_lambda (lambda - lambda_pred).
            let x = self.state_x();
            let mut p = tl * (*lambda - lambda_pred);
            for i in 0..n_st {
                p += tx[i] * (x[i] - x_pred[i]);
            }

            self.fill_jacobian(ybus);
            if self
                .solver
                .solve(
                    &mut self.pat.j_col_ptrs,
                    &mut self.pat.j_row_indices,
                    &mut self.j_values,
                    &mut self.fbuf,
                    n_st,
                )
                .is_err()
            {
                return Err(());
            }
            *factorizations += 1;

            // Reuse the factorization for z = J^-1 d, or refactor when unsupported.
            self.zbuf.copy_from_slice(&self.d);
            if self.solver.resolve(&mut self.zbuf).is_ok() {
                *resolves += 1;
            } else {
                self.fill_jacobian(ybus);
                if self
                    .solver
                    .solve(
                        &mut self.pat.j_col_ptrs,
                        &mut self.pat.j_row_indices,
                        &mut self.j_values,
                        &mut self.zbuf,
                        n_st,
                    )
                    .is_err()
                {
                    return Err(());
                }
                *factorizations += 1;
            }

            // Augmented Newton update for the power equations and plane constraint.
            let (y, z) = (&self.fbuf, &self.zbuf);
            let mut tx_y = 0.0;
            let mut tx_z = 0.0;
            for i in 0..n_st {
                tx_y += tx[i] * y[i];
                tx_z += tx[i] * z[i];
            }
            let denom = tl + tx_z;
            if denom.abs() < 1e-14 {
                return Err(());
            }
            let d_lambda = (p - tx_y) / denom;
            for i in 0..n_act {
                self.v_a[i] -= y[i] + d_lambda * z[i];
            }
            for j in 0..self.npq {
                self.v_m[j] -= y[n_act + j] + d_lambda * z[n_act + j];
            }
            *lambda -= d_lambda;
            self.refresh_v();
        }
        // Check convergence once more after the final update.
        let norm = self.mismatch(ybus, *lambda);
        if norm.is_finite() && norm < opts.tol && *lambda >= 0.0 {
            Ok(opts.max_corr_iter)
        } else {
            Err(())
        }
    }

    /// Trace from lambda = 1 after solving the base point with ordinary Newton.
    pub fn run(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        opts: &CpfOptions,
    ) -> Result<CpfReport, String> {
        opts.validate()?;
        let (base, iterations) = newton_pf(
            ybus,
            &self.s_base,
            &self.v_spec,
            self.npv,
            self.npq,
            Some(opts.tol),
            Some(opts.base_max_iter),
            &mut self.solver,
            None,
        )
        .map_err(|(msg, _, it)| {
            format!("base point lambda=1 NR failed after {it} iterations: {msg}")
        })?;
        self.run_from_base(ybus, base, iterations, opts)
    }

    /// Trace from a converged base voltage in the same solver bus order.
    pub fn run_from_base(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        base: DVector<Complex64>,
        base_iterations: usize,
        opts: &CpfOptions,
    ) -> Result<CpfReport, String> {
        opts.validate()?;
        if base.len() != self.v_cur.len() || base.iter().any(|v| !v.is_finite()) {
            return Err("CPF base voltage is invalid for the prepared model".into());
        }
        let started = Instant::now();
        let mut factorizations = base_iterations;
        let mut resolves = 0usize;

        self.v_cur = base.clone();
        for i in 0..base.len() {
            self.v_m[i] = base[i].norm();
            self.v_a[i] = base[i].arg();
        }
        self.refresh_v();
        let _ = self.mismatch(ybus, 1.0);

        let mut points = vec![CpfPoint {
            lambda: 1.0,
            voltage: base,
            corrector_iterations: 0,
        }];
        let mut lambda = 1.0;
        let mut tangent = self.tangent(ybus, &mut factorizations)?;
        let t_lambda0 = tangent[self.n_state].abs().max(1e-12);
        let mut sigma = opts.step0;
        let mut termination = CpfTermination::MaxPointsReached;

        loop {
            if points.len() >= opts.max_points {
                break;
            }
            let n_st = self.n_state;
            let sigma_eff = sigma * (tangent[n_st].abs() / t_lambda0).clamp(0.05, 1.0);
            let x = self.state_x();
            let mut x_pred = x.clone();
            for i in 0..n_st {
                x_pred[i] += sigma_eff * tangent[i];
            }
            let lambda_pred = lambda + sigma_eff * tangent[n_st];

            for i in 0..self.n_active {
                self.v_a[i] = x_pred[i];
            }
            for j in 0..self.npq {
                self.v_m[j] = x_pred[self.n_active + j];
            }
            self.refresh_v();
            let mut lambda_corr = lambda_pred;

            match self.corrector(
                ybus,
                &tangent,
                &x_pred,
                lambda_pred,
                &mut lambda_corr,
                opts,
                &mut factorizations,
                &mut resolves,
            ) {
                Ok(iterations) => {
                    points.push(CpfPoint {
                        lambda: lambda_corr,
                        voltage: self.v_cur.clone(),
                        corrector_iterations: iterations,
                    });
                    lambda = lambda_corr;
                    sigma = (sigma * opts.step_grow).min(opts.step_max);

                    let mut next = self.tangent(ybus, &mut factorizations)?;
                    let dot: f64 = tangent.iter().zip(next.iter()).map(|(a, b)| a * b).sum();
                    if dot < 0.0 {
                        for value in next.iter_mut() {
                            *value = -*value;
                        }
                    }
                    if next[self.n_state] < 0.0 {
                        termination = CpfTermination::NosePassed;
                        break;
                    }
                    tangent = next;
                }
                Err(()) => {
                    sigma *= opts.step_shrink;
                    if sigma < opts.step_min {
                        termination = CpfTermination::StepBelowMinimum;
                        break;
                    }
                    let last = points.last().expect("base point is always present");
                    self.v_cur = last.voltage.clone();
                    for i in 0..last.voltage.len() {
                        self.v_m[i] = last.voltage[i].norm();
                        self.v_a[i] = last.voltage[i].arg();
                    }
                    self.refresh_v();
                    lambda = last.lambda;
                }
            }
        }

        let nose = if termination == CpfTermination::NosePassed {
            let (lambda, voltage) = refine_nose(&points);
            Some(CpfNose { lambda, voltage })
        } else {
            None
        };
        Ok(CpfReport {
            points,
            nose,
            termination,
            stats: CpfStats {
                factorizations,
                resolves,
                elapsed: started.elapsed(),
            },
        })
    }
}

/// Fit a quadratic lambda curve against cumulative chord length over the last
/// three accepted points. Interpolate voltage with the same parameterization.
/// Fall back to the maximum-lambda point if the fit is degenerate or has no
/// in-range maximum.
fn refine_nose(points: &[CpfPoint]) -> (f64, DVector<Complex64>) {
    let fallback = |points: &[CpfPoint]| {
        let p = points
            .iter()
            .max_by(|a, b| a.lambda.partial_cmp(&b.lambda).unwrap())
            .unwrap();
        (p.lambda, p.voltage.clone())
    };
    if points.len() < 3 {
        return fallback(points);
    }
    let n = points.len();
    let (p0, p1, p2) = (&points[n - 3], &points[n - 2], &points[n - 1]);
    let dist = |a: &CpfPoint, b: &CpfPoint| -> f64 {
        let mut s = (b.lambda - a.lambda).powi(2);
        for i in 0..a.voltage.len() {
            s += (b.voltage[i].norm() - a.voltage[i].norm()).powi(2)
                + (b.voltage[i].arg() - a.voltage[i].arg()).powi(2);
        }
        s.sqrt()
    };
    let s1 = dist(p0, p1);
    let s2 = s1 + dist(p1, p2);
    if s1 < 1e-12 || s2 - s1 < 1e-12 {
        return fallback(points);
    }
    let (l0, l1, l2) = (p0.lambda, p1.lambda, p2.lambda);
    let a = ((l2 - l0) / s2 - (l1 - l0) / s1) / (s2 - s1);
    let b = (l1 - l0) / s1 - a * s1;
    if a >= 0.0 {
        return fallback(points);
    }
    let s_star = -b / (2.0 * a);
    if !(0.0..=s2).contains(&s_star) {
        return fallback(points);
    }
    let lambda_nose = a * s_star * s_star + b * s_star + l0;
    let n_bus = p0.voltage.len();
    let mut v_nose = p0.voltage.clone();
    for i in 0..n_bus {
        let (v0, v1, v2) = (p0.voltage[i], p1.voltage[i], p2.voltage[i]);
        let av = ((v2 - v0) / s2 - (v1 - v0) / s1) / (s2 - s1);
        let bv = (v1 - v0) / s1 - av * s1;
        v_nose[i] = av * s_star * s_star + bv * s_star + v0;
    }
    (lambda_nose, v_nose)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_reject_invalid_steps_and_limits() {
        assert!(
            CpfOptions {
                step0: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            CpfOptions {
                step_shrink: 1.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            CpfOptions {
                max_points: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
