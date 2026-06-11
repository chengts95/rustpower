use std::f64::consts::PI;

use nalgebra::{DVector, ComplexField, SimdComplexField};
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use num_traits::Zero;

use super::new_dsdvbus2::{fill_jacobian_v2, JacobianPattern2};
use super::newtonpf::assemble_f_v2;
use super::solver::Solve;

/// Newton-Raphson power flow with Iwamoto optimal multiplier step size control.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn newton_pf_iwamoto<Solver: Solve>(
    Ybus: &CscMatrix<Complex64>,
    Sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let mut v = v_init.clone();
    let max_iter = max_iter.unwrap_or(100);
    let tol = tolerance.unwrap_or(1e-6);

    let j_pattern = JacobianPattern2::build_from_permuted(
        Ybus.col_offsets(),
        Ybus.row_indices(),
        npv,
        npq,
    );
    let n_state = npv + 2 * npq;
    let mut j_values = vec![0.0; j_pattern.nnz_j];

    let n_bus = npv + npq;
    let mut mis = &v.component_mul(&(Ybus * &v).conjugate()) - Sbus;
    let mut F = DVector::zeros(n_state);
    assemble_f_v2(&mut F, n_bus, &mis, n_state, npq);
    if F.norm() < tol {
        return Ok((v, 0));
    }

    let mut v_m = v.map(|e| e.simd_modulus());
    let mut v_a = v.map(|e| e.simd_argument());
    let mut v_norm = v.map(|e| e.simd_signum());

    let Ap = unsafe {
        std::slice::from_raw_parts_mut(
            j_pattern.j_col_ptrs.as_ptr() as *mut usize,
            j_pattern.j_col_ptrs.len(),
        )
    };
    let Ai = unsafe {
        std::slice::from_raw_parts_mut(
            j_pattern.j_row_indices.as_ptr() as *mut usize,
            j_pattern.j_row_indices.len(),
        )
    };

    for it in 0..max_iter {
        let ibus = Ybus * &v;

        fill_jacobian_v2(
            Ybus,
            v.as_slice(),
            v_norm.as_slice(),
            ibus.as_slice(),
            &j_pattern,
            npv,
            npq,
            &mut j_values,
        );

        // Save original mismatch vector a before solver.solve overwrites it
        let a = F.clone();

        let _ = solver.solve(
            Ap,
            Ai,
            j_values.as_mut_slice(),
            F.data.as_mut_slice(),
            n_state,
        );

        let dx = &F;

        // Reconstruct the complex step vector dv for each bus
        let mut dv = DVector::zeros(v.len());
        for i in 0..v.len() {
            if i < npq {
                let dx_theta = dx[i];
                let dx_v = dx[n_bus + i];
                let angle = v_a[i];
                let mag = v_m[i];
                dv[i] = Complex64::from_polar(1.0, angle) * Complex64::new(-dx_v, -mag * dx_theta);
            } else if i < n_bus {
                let dx_theta = dx[i];
                let angle = v_a[i];
                let mag = v_m[i];
                dv[i] = Complex64::from_polar(1.0, angle) * Complex64::new(0.0, -mag * dx_theta);
            } else {
                dv[i] = Complex64::zero();
            }
        }

        // Calculate the quadratic term c = dv * (Ybus * dv)*
        let ybus_dv = Ybus * &dv;
        let c_complex = dv.component_mul(&ybus_dv.map(|e| e.conjugate()));
        let mut c = DVector::zeros(n_state);
        assemble_f_v2(&mut c, n_bus, &c_complex, n_state, npq);

        // Find the optimal multiplier mu
        let mu = solve_iwamoto_multiplier(&a, &c);

        // Angle update: all non-slack buses.
        v_a.rows_range_mut(0..n_bus)
            .zip_apply(&dx.rows_range(0..n_bus), |a, b| {
                *a -= mu * b;
                *a = a.rem_euclid(2.0 * PI);
            });
        // Magnitude update: PQ buses only (at 0..npq in PQ-first ordering).
        let mut vm_pq = v_m.rows_range_mut(0..npq);
        vm_pq.zip_apply(&dx.rows_range(n_bus..n_state), |a, b| *a -= mu * b);

        v_norm.zip_apply(&v_a, |a, va| *a = Complex64::from_polar(1.0, va));
        v.zip_zip_apply(&v_norm, &v_m, |a, e, vm| *a = vm * e);

        v.component_mul(&(Ybus * &v).conjugate())
            .sub_to(Sbus, &mut mis);
        assemble_f_v2(&mut F, n_bus, &mis, n_state, npq);

        if F.norm() < tol {
            return Ok((v, it + 1));
        }
    }

    Err((String::from("Did not converge!"), v, max_iter))
}

#[allow(non_snake_case)]
fn solve_iwamoto_multiplier(a: &DVector<f64>, c: &DVector<f64>) -> f64 {
    let a_dot_a = a.dot(a);
    let a_dot_c = a.dot(c);
    let c_dot_c = c.dot(c);

    if c_dot_c < 1e-12 {
        return 1.0;
    }

    let g0 = -a_dot_a;
    let g1 = a_dot_a + 2.0 * a_dot_c;
    let g2 = -3.0 * a_dot_c;
    let g3 = 2.0 * c_dot_c;

    // Solve g3*mu^3 + g2*mu^2 + g1*mu + g0 = 0
    let A = g2 / g3;
    let B = g1 / g3;
    let C = g0 / g3;

    let p = B - A * A / 3.0;
    let q = C - A * B / 3.0 + 2.0 * A * A * A / 27.0;

    let D = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);

    let mut roots = Vec::new();
    // Always consider 1.0 as a candidate
    roots.push(1.0);

    if D > 0.0 {
        let sqrt_d = D.sqrt();
        let u = -q / 2.0 + sqrt_d;
        let v = -q / 2.0 - sqrt_d;

        let u_cbrt = u.signum() * u.abs().powf(1.0 / 3.0);
        let v_cbrt = v.signum() * v.abs().powf(1.0 / 3.0);

        let x = u_cbrt + v_cbrt;
        let mu = x - A / 3.0;
        roots.push(mu);
    } else {
        let r = (-p * p * p / 27.0).sqrt();
        if r > 1e-12 {
            let cos_val = (-q / (2.0 * r)).clamp(-1.0, 1.0);
            let phi = cos_val.acos();
            let term = 2.0 * (-p / 3.0).sqrt();

            for k in 0..3 {
                let x = term * ((phi + 2.0 * PI * (k as f64)) / 3.0).cos();
                let mu = x - A / 3.0;
                roots.push(mu);
            }
        }
    }

    // Evaluate V(mu) = 0.5 * ||(1-mu)*a + mu^2 * c||^2 for each candidate
    // Clamp candidate to [0.05, 1.0] for physical stability
    let mut best_mu = 1.0;
    let mut min_val = f64::MAX;

    for &r in &roots {
        let mu = r.clamp(0.05, 1.0);
        let mut f_mu = DVector::zeros(a.len());
        for i in 0..a.len() {
            f_mu[i] = (1.0 - mu) * a[i] + mu * mu * c[i];
        }
        let val = f_mu.dot(&f_mu);
        if val < min_val {
            min_val = val;
            best_mu = mu;
        }
    }

    best_mu
}
