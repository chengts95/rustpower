//! Shared damping settings and polar trial-step checks for all LM layouts.

use num_complex::Complex64;
use nalgebra_sparse::CscMatrix;
use serde::{Deserialize, Serialize};

/// Metric for the *linearized* voltage step. This does not change coordinates
/// or add an explicit trust-region radius.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DampingMetric {
    #[default]
    Polar,
    /// M'M: angle weights |V|²; magnitude weights 1.
    CartesianVoltage,
    /// Fixed network metric: |Y_jj|² for both retained variables of bus j.
    YbusDiagonal,
}

impl DampingMetric {
    /// Read the network once per solve; no per-iteration Ybus traversal.
    pub(crate) fn prepare(self, ybus: &CscMatrix<Complex64>, n_act: usize) -> DampingWeights {
        let bus_weights = if matches!(self, Self::YbusDiagonal) {
            Some((0..n_act).map(|k| {
                let start = ybus.col_offsets()[k];
                let end = ybus.col_offsets()[k + 1];
                let offset = ybus.row_indices()[start..end].binary_search(&k)
                    .expect("Ybus damping requires diagonal entries");
                let weight = ybus.values()[start + offset].norm_sqr();
                assert!(weight.is_finite() && weight > 0.0,
                    "Ybus damping requires finite positive diagonal weights");
                weight
            }).collect())
        } else { None };
        DampingWeights { metric: self, bus_weights }
    }
}

/// One definition of the diagonal, used for matrix updates, step norms and
/// predicted reduction. Only the network metric owns a vector (per bus).
pub(crate) struct DampingWeights {
    metric: DampingMetric,
    bus_weights: Option<Vec<f64>>,
}

impl DampingWeights {
    pub(crate) fn weight(&self, v: &[Complex64], column: usize, n_act: usize) -> f64 {
        match self.metric {
            DampingMetric::Polar => 1.0,
            DampingMetric::CartesianVoltage if column < n_act => v[column].norm_sqr(),
            DampingMetric::CartesianVoltage => 1.0,
            DampingMetric::YbusDiagonal => {
                let bus = if column < n_act { column } else { column - n_act };
                self.bus_weights.as_ref().unwrap()[bus]
            }
        }
    }

    pub(crate) fn step_norm_squared(&self, v: &[Complex64], delta: &[f64], n_act: usize) -> f64 {
        delta
            .iter()
            .enumerate()
            .map(|(c, d)| self.weight(v, c, n_act) * d * d)
            .sum()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrustRegionOptions {
    /// Radius in the selected metric's norm, across all retained states.
    pub initial_radius: f64,
    pub min_radius: f64,
    pub max_radius: f64,
    pub shrink_factor: f64,
    pub grow_factor: f64,
    pub poor_step_threshold: f64,
}

impl Default for TrustRegionOptions {
    fn default() -> Self {
        Self {
            initial_radius: 1.0,
            min_radius: 1e-12,
            max_radius: 1e6,
            shrink_factor: 0.5,
            grow_factor: 2.0,
            poor_step_threshold: 0.25,
        }
    }
}

pub(crate) struct TrustRegion<'a> {
    options: Option<&'a TrustRegionOptions>,
    pub(crate) radius: f64,
}

impl<'a> TrustRegion<'a> {
    pub(crate) fn new(options: Option<&'a TrustRegionOptions>) -> Self {
        Self {
            radius: options.map_or(f64::INFINITY, |o| o.initial_radius),
            options,
        }
    }

    pub(crate) fn allows(&self, norm_squared: f64) -> bool {
        norm_squared.is_finite() && norm_squared.sqrt() <= self.radius
    }

    pub(crate) fn reject(&mut self) {
        if let Some(o) = self.options {
            self.radius = (self.radius * o.shrink_factor).max(o.min_radius);
        }
    }

    pub(crate) fn accept(&mut self, rho: f64, norm_squared: f64, good_threshold: f64) {
        if let Some(o) = self.options {
            if rho < o.poor_step_threshold {
                self.reject();
            } else if rho > good_threshold && norm_squared.sqrt() > self.radius / o.grow_factor {
                self.radius = (self.radius * o.grow_factor).min(o.max_radius);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LmOptions {
    pub damping_metric: DampingMetric,
    /// None preserves damping-only control for matched comparisons.
    pub trust_region: Option<TrustRegionOptions>,
    pub initial_mu: f64,
    pub min_mu: f64,
    pub max_mu: f64,
    /// Multiplier after a rejected residual or voltage-domain trial.
    pub mu_increase: f64,
    /// Divisor after an accepted step with rho above good_step_threshold.
    pub mu_decrease: f64,
    pub failed_step_increase: f64,
    pub acceptance_threshold: f64,
    pub good_step_threshold: f64,
    pub max_trials: usize,
    /// Reject |V| + delta_|V| <= 0 before converting to complex voltage.
    /// Disable only for comparison with the former unconstrained update.
    pub reject_nonpositive_voltage: bool,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            damping_metric: DampingMetric::Polar,
            trust_region: None,
            initial_mu: 1e-2,
            min_mu: 1e-12,
            max_mu: 1e12,
            mu_increase: 2.0,
            mu_decrease: 3.0,
            failed_step_increase: 10.0,
            acceptance_threshold: 1e-4,
            good_step_threshold: 0.75,
            max_trials: 30,
            reject_nonpositive_voltage: true,
        }
    }
}

impl LmOptions {
    pub fn validate(&self) -> Result<(), &'static str> {
        if ![self.initial_mu, self.min_mu, self.max_mu]
            .iter()
            .all(|x| x.is_finite())
            || !(0.0 < self.min_mu
                && self.min_mu <= self.initial_mu
                && self.initial_mu <= self.max_mu)
        {
            return Err("require finite 0 < min_mu <= initial_mu <= max_mu");
        }
        if ![
            self.mu_increase,
            self.mu_decrease,
            self.failed_step_increase,
        ]
        .iter()
        .all(|x| x.is_finite() && *x > 1.0)
        {
            return Err("damping increase multipliers and decrease divisor must be finite and > 1");
        }
        if !self.acceptance_threshold.is_finite()
            || !self.good_step_threshold.is_finite()
            || !(0.0 <= self.acceptance_threshold
                && self.acceptance_threshold < self.good_step_threshold
                && self.good_step_threshold < 1.0)
        {
            return Err("require 0 <= acceptance_threshold < good_step_threshold < 1");
        }
        if self.max_trials == 0 {
            return Err("max_trials must be positive");
        }
        if let Some(o) = &self.trust_region {
            if ![
                o.initial_radius,
                o.min_radius,
                o.max_radius,
                o.shrink_factor,
                o.grow_factor,
                o.poor_step_threshold,
            ]
            .iter()
            .all(|x| x.is_finite())
                || !(0.0 < o.min_radius
                    && o.min_radius <= o.initial_radius
                    && o.initial_radius <= o.max_radius
                    && 0.0 < o.shrink_factor
                    && o.shrink_factor < 1.0
                    && o.grow_factor > 1.0
                    && self.acceptance_threshold <= o.poor_step_threshold
                    && o.poor_step_threshold < self.good_step_threshold)
            {
                return Err("invalid trust-region radii, factors or poor-step threshold");
            }
        }
        Ok(())
    }

    /// Try the configured upper bound once before giving up.
    pub(crate) fn increase_mu(&self, mu: &mut f64, factor: f64) -> bool {
        if *mu >= self.max_mu {
            return false;
        }
        *mu = (*mu * factor).min(self.max_mu);
        true
    }

    pub(crate) fn accepted_mu(&self, mu: f64, rho: f64) -> f64 {
        if rho > self.good_step_threshold {
            (mu / self.mu_decrease).max(self.min_mu)
        } else {
            mu
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum TrialError {
    NonFinite,
    NonPositiveMagnitude { bus: usize },
}

/// The accepted voltage is read-only. A failed trial is never copied back.
pub(crate) fn polar_trial(
    v: &[Complex64],
    delta: &[f64],
    n_act: usize,
    npq: usize,
    reject_nonpositive: bool,
    trial: &mut [Complex64],
) -> Result<(), TrialError> {
    trial.copy_from_slice(v);
    for k in 0..n_act {
        let mag = v[k].norm() + if k < npq { delta[n_act + k] } else { 0.0 };
        let angle = v[k].arg() + delta[k];
        if !mag.is_finite() || !angle.is_finite() {
            return Err(TrialError::NonFinite);
        }
        if reject_nonpositive && mag <= 0.0 {
            return Err(TrialError::NonPositiveMagnitude { bus: k });
        }
        trial[k] = Complex64::from_polar(mag, angle);
    }
    if trial.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err(TrialError::NonFinite);
    }
    Ok(())
}

/// Reduction of the *undamped* quadratic model. For either B = J'J or
/// B = J'J + H(r), (B + mu W) delta = -g gives
/// -g'delta - delta'B delta / 2 = (mu delta'W delta - g'delta) / 2.
pub(crate) fn predicted_reduction(
    g: &[f64],
    delta: &[f64],
    mu: f64,
    step_norm_squared: f64,
) -> f64 {
    let gd: f64 = g.iter().zip(delta).map(|(g, d)| g * d).sum();
    0.5 * (mu * step_norm_squared - gd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::solver::Solve;
    use crate::lm::{LmDriver, gn_flat::GnDriver, gn_triu::GnTriuDriver};
    use nalgebra_sparse::CscMatrix;

    #[test]
    fn ybus_metric_preserves_newton_limit_and_weights_both_diagonals() {
        use nalgebra::{Matrix3, Vector3};
        let y = CscMatrix::try_from_csc_data(3, 3, vec![0, 3, 6, 9],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            [(2.0, 3.0), (-1.0, -1.0), (-1.0, -2.0),
             (-1.0, -1.0), (4.0, -2.0), (-3.0, 3.0),
             (-1.0, -2.0), (-3.0, 3.0), (4.0, -1.0)]
                .map(|(re, im)| Complex64::new(re, im)).to_vec()).unwrap();
        let damping = DampingMetric::YbusDiagonal.prepare(&y, 2);
        let v = [Complex64::new(0.2, 0.1); 3];
        // PQ angle, PV angle, PQ magnitude: one network weight per bus.
        let weights = Vector3::new(13.0, 20.0, 13.0);
        for c in 0..3 { assert_eq!(damping.weight(&v, c, 2), weights[c]); }
        let w = Matrix3::from_diagonal(&weights);
        let inverse_scale = Matrix3::from_diagonal(&weights.map(|x| 1.0 / x.sqrt()));
        let j = Matrix3::new(2.0, -0.3, 0.2, 0.7, 1.4, -0.1, 0.2, 0.3, 1.8);
        let r = Vector3::new(0.4, -0.9, 0.2);
        let scaled_j = j * inverse_scale;
        for mu in [0.0, 0.03, 1.0] {
            let step = (j.transpose() * j + mu * w).lu().solve(&(-j.transpose() * r)).unwrap();
            let scaled_step = (scaled_j.transpose() * scaled_j + mu * Matrix3::identity())
                .lu().solve(&(-scaled_j.transpose() * r)).unwrap();
            assert!((step - inverse_scale * scaled_step).norm() < 1e-13);
            if mu == 0.0 { assert!((step - j.lu().solve(&(-r)).unwrap()).norm() < 1e-13); }
            let norm = damping.step_norm_squared(&v, step.as_slice(), 2);
            assert!((norm - step.dot(&(w * step))).abs() < 1e-13);
            let actual_model_decrease = 0.5 * (r.norm_squared() - (r + j * step).norm_squared());
            let prediction = predicted_reduction((j.transpose() * r).as_slice(), step.as_slice(), mu, norm);
            assert!((prediction - actual_model_decrease).abs() < 1e-13);
        }
        // Check the actual augmented-diagonal fill, including the PQ magnitude.
        let pat = crate::lm::KktPattern::build(&y, 1, 1);
        let layout = crate::lm::FlatLayout::build(&pat);
        let before: Vec<f64> = (0..layout.nnz_flat).map(|k| k as f64 / 10.0).collect();
        let mut values = before.clone();
        crate::lm::kernels::apply_mu_delta_weighted::<true>(&pat, &mut values, 0.3,
            |c| damping.weight(&v, c, 2));
        for c in 0..6 {
            for p in layout.col_offsets[c]..layout.col_offsets[c + 1] {
                let change = if c < 3 && layout.row_indices[p] == c { 0.3 * weights[c] } else { 0.0 };
                assert!((values[p] - before[p] - change).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn trust_radius_controls_both_coordinates_and_adapts() {
        let settings = TrustRegionOptions::default();
        let mut region = TrustRegion::new(Some(&settings));
        let v = [Complex64::new(1.0, 0.0)];
        let damping = DampingWeights { metric: DampingMetric::CartesianVoltage, bus_weights: None };
        for step in [[2.0, 0.0], [0.0, 2.0]] {
            assert!(
                !region.allows(damping.step_norm_squared(&v, &step, 1))
            );
        }
        region.accept(0.95, 0.75 * 0.75, 0.75);
        assert_eq!(region.radius, 2.0);
        region.accept(0.1, 0.25, 0.75);
        assert_eq!(region.radius, 1.0);
        region.reject();
        assert_eq!(region.radius, 0.5);
        let options = LmOptions {
            trust_region: Some(settings),
            ..Default::default()
        };
        options.validate().unwrap();
    }

    #[test]
    fn cartesian_metric_reproduces_cartesian_gn_step() {
        use nalgebra::{Matrix2, Vector2};
        let j_rect = Matrix2::<f64>::new(2.0, -0.3, 0.7, 1.4);
        let r = Vector2::new(0.4, -0.9);
        let mu = 0.6;
        let damping = DampingWeights { metric: DampingMetric::CartesianVoltage, bus_weights: None };
        let cartesian_step = (j_rect.transpose() * j_rect + mu * Matrix2::<f64>::identity())
            .lu()
            .solve(&(-j_rect.transpose() * r))
            .unwrap();
        for magnitude in [0.2, 1.0, 1.6] {
            let theta: f64 = 0.7;
            let m = Matrix2::new(
                -magnitude * theta.sin(),
                theta.cos(),
                magnitude * theta.cos(),
                theta.sin(),
            );
            let voltage = Complex64::from_polar(magnitude, theta);
            let w = Matrix2::from_diagonal(&Vector2::new(
                voltage.norm_sqr(),
                1.0,
            ));
            assert!((m.transpose() * m - w).norm() < 1e-14);
            let j_polar = j_rect * m;
            let d = (j_polar.transpose() * j_polar + mu * w)
                .lu()
                .solve(&(-j_polar.transpose() * r))
                .unwrap();
            assert!((m * d - cartesian_step).norm() < 1e-14);
            let norm =
                damping.step_norm_squared(&[voltage], d.as_slice(), 1);
            assert!((norm - (m * d).norm_squared()).abs() < 1e-14);
        }
    }

    struct TrialSolver {
        always_negative: bool,
        diagonals: Vec<f64>,
        initial_rhs: Option<Vec<f64>>,
    }

    impl Solve for TrialSolver {
        fn solve(
            &mut self,
            cp: &mut [usize],
            ri: &mut [usize],
            values: &mut [f64],
            rhs: &mut [f64],
            n: usize,
        ) -> Result<(), &'static str> {
            assert_eq!(n, 4);
            if let Some(initial) = &self.initial_rhs {
                assert_eq!(rhs, initial, "retry must use the same accepted residual");
            } else {
                self.initial_rhs = Some(rhs.to_vec());
            }
            let p = (cp[0]..cp[1]).find(|&p| ri[p] == 0).unwrap();
            self.diagonals.push(values[p]);
            rhs.fill(0.0);
            rhs[1] = if self.always_negative || self.diagonals.len() == 1 {
                -2.2
            } else {
                -0.1
            };
            Ok(())
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn all_drivers_retry_from_accepted_voltage_and_respect_acceptance_threshold() {
        // PQ bus connected to a 1 pu slack through a unit conductance.
        // Starting at 1.1 pu, a -0.1 pu magnitude step solves P = Q = 0.
        let y = CscMatrix::try_from_csc_data(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            [1.0, -1.0, -1.0, 1.0]
                .map(|x| Complex64::new(x, 0.0))
                .to_vec(),
        )
        .unwrap();
        for method in [
            "full",
            "full_operator",
            "upper",
            "upper_operator",
            "exact_gn",
            "exact_hessian",
        ] {
            for (always_negative, acceptance_threshold, should_converge) in [
                (false, 1e-4, true),
                (true, 1e-4, false),
                (false, 0.6, false),
            ] {
                let options = LmOptions {
                    initial_mu: 0.2,
                    mu_increase: 4.0,
                    max_mu: 1.0,
                    max_trials: 2,
                    acceptance_threshold,
                    ..Default::default()
                };
                let mut solver = TrialSolver {
                    always_negative,
                    diagonals: vec![],
                    initial_rhs: None,
                };
                let initial = [Complex64::new(1.1, 0.0), Complex64::new(1.0, 0.0)];
                let mut v = initial;
                let sbus = vec![Complex64::new(0.0, 0.0); 2];
                let converged = match method {
                    "full" | "full_operator" => {
                        let mut d = if method == "full" {
                            GnDriver::build(&y, 0, 1, sbus)
                        } else {
                            GnDriver::build_operator(&y, 0, 1, sbus)
                        };
                        d.solve_gn_with_options(&y, &mut solver, &mut v, 1e-8, 1, &options)
                            .converged
                    }
                    "upper" | "upper_operator" => {
                        let mut d = if method == "upper" {
                            GnTriuDriver::build(&y, 0, 1, sbus)
                        } else {
                            GnTriuDriver::build_operator(&y, 0, 1, sbus)
                        };
                        d.solve_gn_with_options(&y, &mut solver, &mut v, 1e-8, 1, &options)
                            .converged
                    }
                    _ => {
                        LmDriver::build(&y, 0, 1, sbus)
                            .solve_lm_with_options(
                                &y,
                                &mut solver,
                                &mut v,
                                method == "exact_hessian",
                                1e-8,
                                1,
                                &options,
                            )
                            .converged
                    }
                };
                assert_eq!(
                    solver.diagonals.len(),
                    2,
                    "{method}: must retry within the same iteration"
                );
                assert!((solver.diagonals[1] - solver.diagonals[0] - 0.6).abs() < 1e-14);
                assert_eq!(converged, should_converge, "{method}");
                if should_converge {
                    assert!(
                        (v[0].re - 1.0).abs() < 1e-14,
                        "{method}: retry from original 1.1 pu"
                    );
                } else {
                    assert_eq!(
                        v, initial,
                        "{method}: rejected trials must not change accepted state"
                    );
                }
            }
        }
    }

    #[test]
    fn rejected_polar_trial_preserves_base_and_can_retry() {
        let v = [Complex64::new(1.0, 0.0), Complex64::new(1.05, 0.0)];
        let saved = v;
        let mut trial = v;
        for magnitude_step in [-1.5, -1.0] {
            assert_eq!(
                polar_trial(&v, &[0.2, magnitude_step], 1, 1, true, &mut trial),
                Err(TrialError::NonPositiveMagnitude { bus: 0 })
            );
            assert_eq!(v, saved);
        }
        polar_trial(&v, &[0.2, -0.1], 1, 1, true, &mut trial).unwrap();
        assert!((trial[0].norm() - 0.9).abs() < 1e-14);
        assert!((trial[0].arg() - 0.2).abs() < 1e-14);
        assert_eq!(trial[1], v[1]);
    }

    #[test]
    fn prediction_matches_explicit_quadratic_model() {
        // Includes an indefinite B, as can occur with the exact Hessian.
        for b in [[2.0, 3.0], [-2.0, 3.0]] {
            let mu = 4.0;
            let g = [1.0, -2.0];
            let d = [-g[0] / (b[0] + mu), -g[1] / (b[1] + mu)];
            let explicit: f64 = (0..2)
                .map(|i| -g[i] * d[i] - 0.5 * b[i] * d[i] * d[i])
                .sum();
            assert!(
                (predicted_reduction(&g, &d, mu, d.iter().map(|x| x * x).sum()) - explicit).abs()
                    < 1e-14
            );
        }
    }

    #[test]
    fn damping_honors_configured_bounds() {
        let options = LmOptions {
            initial_mu: 0.2,
            min_mu: 0.1,
            max_mu: 1.0,
            ..Default::default()
        };
        options.validate().unwrap();
        let mut mu = 0.2;
        assert!(options.increase_mu(&mut mu, 10.0));
        assert_eq!(mu, 1.0);
        assert!(!options.increase_mu(&mut mu, 2.0));
        assert_eq!(options.accepted_mu(0.2, 0.9), 0.1);
        assert!(
            LmOptions {
                initial_mu: f64::NAN,
                ..options.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            LmOptions {
                mu_increase: 1.0,
                ..options.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            LmOptions {
                acceptance_threshold: 0.8,
                ..options.clone()
            }
            .validate()
            .is_err()
        );
        assert!(
            LmOptions {
                max_trials: 0,
                ..options
            }
            .validate()
            .is_err()
        );
    }
}
