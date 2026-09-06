//! Same-model LM/Newton comparison using performance/audit_lm_6515.py exports.
//! cargo run --release --features klu_dyn --example audit_lm -- INPUT.json [MAX_ITER] [LM_OPTIONS.json] [METHOD]
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::basic::{
    newton_pf,
    solver::{KLUSolver, QDLDLSolver, Solve},
};
use rustpower::lm::{LmDriver, LmOptions, gn_flat::GnDriver, gn_triu::GnTriuDriver};
use serde::Deserialize;
use serde_json::json;
use std::time::Instant;

/// Diagnostic only: check A*x=b, optionally discarding cached KLU pivots.
#[derive(Default)]
struct AuditKlu {
    inner: KLUSolver,
    fresh: bool,
    calls: u64,
    max_relative_residual: f64,
    max_backward_error: f64,
}

impl Solve for AuditKlu {
    fn solve(
        &mut self,
        cp: &mut [usize],
        ri: &mut [usize],
        values: &mut [f64],
        x: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        let rhs = x.to_vec();
        if self.fresh {
            self.inner.reset();
        }
        self.calls += 1;
        self.inner.solve(cp, ri, values, x, n)?;
        let mut residual: Vec<f64> = rhs.iter().map(|b| -b).collect();
        let mut row_sums = vec![0.0; n];
        for c in 0..n {
            for p in cp[c]..cp[c + 1] {
                residual[ri[p]] += values[p] * x[c];
                row_sums[ri[p]] += values[p].abs();
            }
        }
        let inf = |v: &[f64]| {
            v.iter()
                .map(|x| {
                    if x.is_finite() {
                        x.abs()
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0_f64, f64::max)
        };
        let r = inf(&residual);
        let relative = r / inf(&rhs).max(f64::MIN_POSITIVE);
        let backward = r / (inf(&row_sums) * inf(x) + inf(&rhs)).max(f64::MIN_POSITIVE);
        self.max_relative_residual = self.max_relative_residual.max(relative);
        self.max_backward_error = self.max_backward_error.max(backward);
        eprintln!(
            "linear trial={} fresh={} relative_residual={relative:.3e} backward_error={backward:.3e}",
            self.calls, self.fresh
        );
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[derive(Deserialize)]
struct Input {
    case: String,
    init: String,
    nb: usize,
    npv: usize,
    npq: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    y_re: Vec<f64>,
    y_im: Vec<f64>,
    s_re: Vec<f64>,
    s_im: Vec<f64>,
    v_re: Vec<f64>,
    v_im: Vec<f64>,
    tolerance_pu: f64,
    max_iter: usize,
}
fn complex(re: &[f64], im: &[f64]) -> Vec<Complex64> {
    re.iter()
        .zip(im)
        .map(|(&r, &i)| Complex64::new(r, i))
        .collect()
}
fn main() {
    let path = std::env::args().nth(1).expect("input JSON path");
    let mut a: Input = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    if let Some(limit) = std::env::args().nth(2) {
        a.max_iter = limit.parse().expect("iteration limit");
    }
    let options: LmOptions = std::env::args()
        .nth(3)
        .map(|path| {
            serde_json::from_slice(&std::fs::read(path).expect("LM options file"))
                .expect("LM options JSON")
        })
        .unwrap_or_default();
    options.validate().expect("valid LM options");
    let y =
        CscMatrix::try_from_csc_data(a.nb, a.nb, a.cp, a.ri, complex(&a.y_re, &a.y_im)).unwrap();
    let sb = complex(&a.s_re, &a.s_im);
    let v0 = complex(&a.v_re, &a.v_im);
    let mut records = vec![];
    let selected_method = std::env::args().nth(4);
    for method in [
        "newton_klu",
        "lm_full_klu",
        "lm_upper_qdldl",
        "lm_operator_full_klu",
        "lm_operator_upper_qdldl",
        "lm_exact_layout_gn_klu",
        "lm_exact_hessian_klu",
        "lm_exact_hessian_audit_klu",
        "lm_exact_hessian_fresh_klu",
        "lm_exact_layout_gn_fresh_klu",
    ] {
        if let Some(selected) = &selected_method {
            if selected != method {
                continue;
            }
        } else if method == "lm_exact_hessian_audit_klu" || method.ends_with("fresh_klu") {
            // Expensive linear-solve diagnostics are explicitly selected only.
            continue;
        }
        let start = Instant::now();
        let mut v = v0.clone();
        let mut linear_audit = None;
        let (ok, it, fill, solve, nsolve) = match method {
            "newton_klu" => {
                let r = newton_pf(
                    &y,
                    &DVector::from_vec(sb.clone()),
                    &DVector::from_vec(v),
                    a.npv,
                    a.npq,
                    Some(a.tolerance_pu),
                    Some(a.max_iter),
                    &mut KLUSolver::default(),
                    None,
                );
                let (ok, x, it) = match r {
                    Ok((v, it)) => (true, v, it),
                    Err((_, v, it)) => (false, v, it),
                };
                v = x.as_slice().to_vec();
                (ok, it, None, None, None)
            }
            "lm_full_klu" | "lm_operator_full_klu" => {
                let mut d = if method == "lm_operator_full_klu" {
                    GnDriver::build_operator(&y, a.npv, a.npq, sb.clone())
                } else {
                    GnDriver::build(&y, a.npv, a.npq, sb.clone())
                };
                let r = d.solve_gn_with_options(
                    &y,
                    &mut KLUSolver::default(),
                    &mut v,
                    a.tolerance_pu,
                    a.max_iter,
                    &options,
                );
                (
                    r.converged,
                    r.iterations,
                    Some(d.prof_fill_ns),
                    Some(d.prof_solve_ns),
                    Some(d.n_solves),
                )
            }
            "lm_exact_layout_gn_klu" | "lm_exact_hessian_klu" => {
                // Same driver, layout, solver and damping policy; only H(r) changes.
                let mut d = LmDriver::build(&y, a.npv, a.npq, sb.clone());
                let r = d.solve_lm_with_options(
                    &y,
                    &mut KLUSolver::default(),
                    &mut v,
                    method == "lm_exact_hessian_klu",
                    a.tolerance_pu,
                    a.max_iter,
                    &options,
                );
                (r.converged, r.iterations, None, None, None)
            }
            "lm_exact_hessian_audit_klu"
            | "lm_exact_hessian_fresh_klu"
            | "lm_exact_layout_gn_fresh_klu" => {
                let mut solver = AuditKlu {
                    fresh: method.ends_with("fresh_klu"),
                    ..Default::default()
                };
                let mut d = LmDriver::build(&y, a.npv, a.npq, sb.clone());
                let r = d.solve_lm_with_options(
                    &y,
                    &mut solver,
                    &mut v,
                    method != "lm_exact_layout_gn_fresh_klu",
                    a.tolerance_pu,
                    a.max_iter,
                    &options,
                );
                linear_audit = Some(json!({"fresh_factorization":solver.fresh,
                    "max_relative_residual":solver.max_relative_residual,
                    "max_backward_error":solver.max_backward_error}));
                (r.converged, r.iterations, None, None, Some(solver.calls))
            }
            _ => {
                let mut d = if method == "lm_operator_upper_qdldl" {
                    GnTriuDriver::build_operator(&y, a.npv, a.npq, sb.clone())
                } else {
                    GnTriuDriver::build(&y, a.npv, a.npq, sb.clone())
                };
                let r = d.solve_gn_with_options(
                    &y,
                    &mut QDLDLSolver::default(),
                    &mut v,
                    a.tolerance_pu,
                    a.max_iter,
                    &options,
                );
                (
                    r.converged,
                    r.iterations,
                    Some(d.prof_fill_ns),
                    Some(d.prof_solve_ns),
                    Some(d.n_solves),
                )
            }
        };
        let ms = start.elapsed().as_secs_f64() * 1000.;
        // Independent residual reconstruction through the sparse matrix API.
        let current = &y * &DVector::from_vec(v.clone());
        let mut inf = 0_f64;
        for i in 0..a.npq + a.npv {
            let mis = v[i] * current[i].conj() - sb[i];
            if !mis.re.is_finite() || !mis.im.is_finite() {
                inf = f64::INFINITY;
                break;
            }
            inf = inf.max(mis.re.abs());
            if i < a.npq {
                inf = inf.max(mis.im.abs());
            }
        }
        eprintln!(
            "{} {}: success={} it={} residual={:.6e} total={:.3}ms",
            a.init, method, ok, it, inf, ms
        );
        records.push(json!({"method":method,"converged":ok,"iterations":it,"residual_inf":inf,
            "verified_converged":inf.is_finite() && inf<a.tolerance_pu,
            "total_ms":ms,"fill_ns":fill,"solve_ns":solve,"linear_solves":nsolve,"linear_audit":linear_audit,
            "v_re":v.iter().map(|v|v.re).collect::<Vec<_>>(),"v_im":v.iter().map(|v|v.im).collect::<Vec<_>>()}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"case":a.case,"init":a.init,
        "tolerance_pu":a.tolerance_pu,"max_iter":a.max_iter,"lm_options":options,"records":records}))
        .unwrap()
    );
}
