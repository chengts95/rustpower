//! Shared residual evaluation and test fixtures for the LM driver family
//! (exact-LM under `exact/` and classical GN-LM in `gn_flat.rs`).

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Reduced residual `out = [P mis (n_act); Q mis (n_pq)]` at `v`;
/// `ibus` is scratch. Returns `(‖r‖∞, ½‖r‖²)`.
pub(crate) fn residual(
    ybus: &CscMatrix<Complex64>,
    sbus: &[Complex64],
    ibus: &mut [Complex64],
    n_act: usize,
    npq: usize,
    v: &[Complex64],
    out: &mut [f64],
) -> (f64, f64) {
    use crate::basic::newtonpf::{csc_matvec_complex, fill_f_from_power};

    csc_matvec_complex(
        ybus.col_offsets(),
        ybus.row_indices(),
        ybus.values(),
        v,
        ibus,
    );
    let res_inf = fill_f_from_power::<false>(|i| v[i] * ibus[i].conj(), sbus, npq, n_act, out);
    // LM retains its least-squares merit for step acceptance; convergence
    // uses the same infinity norm as Newton and Iwamoto.
    let merit = 0.5 * out[..n_act + npq].iter().map(|r| r * r).sum::<f64>();
    (res_inf, merit)
}

/// Retain the power injections for the following operator fill.
/// Trial-point evaluations can continue using `residual` without storing powers.
pub(crate) fn residual_with_power(
    ybus: &CscMatrix<Complex64>, sbus: &[Complex64], ibus: &mut [Complex64],
    scalc: &mut [Complex64], n_act: usize, npq: usize,
    v: &[Complex64], out: &mut [f64],
) -> (f64, f64) {
    use crate::basic::newtonpf::{csc_matvec_and_scalc, fill_f_from_scalc};
    csc_matvec_and_scalc(ybus.col_offsets(), ybus.row_indices(), ybus.values(), v, ibus, scalc);
    let norm = fill_f_from_scalc::<false>(scalc, sbus, npq, n_act, out);
    let merit = 0.5 * out[..n_act + npq].iter().map(|r| r * r).sum::<f64>();
    (norm, merit)
}

/// Test networks shared by the exact and GN drivers: the ill-conditioned
/// 14-bus case (ext_ref case2, renumbering-invariant) and the IEEE39
/// `PowerFlowMat` loader.
#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
pub(crate) mod fixtures {
    use nalgebra::DVector;
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    use num_complex::Complex64;

    /// ext_ref case2 network in OLD numbering (slack = 0, PV = {3,6,9,12}).
    pub(crate) const NB: usize = 14;

    fn old_edges() -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = (0..NB).map(|i| (i, (i + 1) % NB)).collect();
        for i in (0..NB).step_by(2) {
            e.push((i, (i + 3) % NB));
        }
        e
    }

    fn old_v_star() -> Vec<Complex64> {
        (0..NB)
            .map(|k| {
                let ang = 0.32 * (1.3 * k as f64).sin() - 0.22 * k as f64 / NB as f64;
                let mag = 0.97 + 0.02 * (2.1 * k as f64).sin();
                Complex64::from_polar(mag, ang)
            })
            .collect()
    }

    /// Ybus for an arbitrary bus ordering: `order[new] = old`.
    fn build_ybus(order: &[usize]) -> CscMatrix<Complex64> {
        // y = 1/(0.2 + j0.6) = 0.5 − j1.5; shunt j0.05 on every diagonal.
        let y = Complex64::new(0.5, -1.5);
        let mut coo = CooMatrix::new(NB, NB);
        let mut diag = vec![Complex64::new(0.0, 0.05); NB];
        for &(oi, oj) in &old_edges() {
            let (i, j) = (
                order.iter().position(|&b| b == oi).unwrap(),
                order.iter().position(|&b| b == oj).unwrap(),
            );
            diag[i] += y;
            diag[j] += y;
            coo.push(i, j, -y);
            coo.push(j, i, -y);
        }
        for k in 0..NB {
            coo.push(k, k, diag[k]);
        }
        CscMatrix::from(&coo)
    }

    /// The case in `[PQ | PV | slack]` order: PQ {1,2,4,5,7,8,10,11,13},
    /// PV {3,6,9,12}, slack {0}. Returns (ybus, n_pv, n_pq, v_star, s_spec).
    pub(crate) fn ill_conditioned_case() -> (
        CscMatrix<Complex64>,
        usize,
        usize,
        Vec<Complex64>,
        Vec<Complex64>,
    ) {
        let order: Vec<usize> = [1, 2, 4, 5, 7, 8, 10, 11, 13, 3, 6, 9, 12, 0].into();
        let ybus = build_ybus(&order);
        let v_star_old = old_v_star();

        // Specified injections from the exact solution (old numbering, but
        // the network is renumbering-invariant — compute with the new one).
        let yv = &ybus * &DVector::from_vec(order.iter().map(|&b| v_star_old[b]).collect());
        let s_spec: Vec<Complex64> = (0..NB)
            .map(|k| {
                let v = v_star_old[order[k]];
                v * yv[k].conj()
            })
            .collect();
        let v_star: Vec<Complex64> = order.iter().map(|&b| v_star_old[b]).collect();
        (ybus, 4, 9, v_star, s_spec)
    }

    /// Flat start in the new order: slack pinned at v*, PV magnitudes at
    /// spec (angle 0), PQ at 1∠0.
    pub(crate) fn flat_start(v_star: &[Complex64], n_act: usize, npq: usize) -> Vec<Complex64> {
        let nb = v_star.len();
        let mut v = vec![Complex64::new(1.0, 0.0); nb];
        for k in npq..n_act {
            v[k] = Complex64::from_polar(v_star[k].norm(), 0.0);
        }
        v[nb - 1] = v_star[nb - 1]; // slack
        v
    }

    pub(crate) fn load_ieee39_mat() -> crate::basic::ecs::powerflow::systems::PowerFlowMat {
        use crate::basic::ecs::elements::PPNetwork;
        use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
        let net: crate::io::pandapower::Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut pf = PowerGrid::default();
        pf.world_mut().insert_resource(PPNetwork(net));
        pf.init_pf_net();
        pf.world()
            .get_resource::<crate::basic::ecs::powerflow::systems::PowerFlowMat>()
            .expect("init_pf_net did not produce a PowerFlowMat resource")
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::newtonpf::fill_f_from_scalc;

    #[test]
    fn lm_and_newton_share_reduced_inf_norm() {
        let ybus = CscMatrix::identity(3);
        let v = vec![Complex64::new(1.0, 0.0); 3];
        let sbus = vec![
            Complex64::new(0.25, -0.75),
            Complex64::new(0.5, 100.0),
            Complex64::new(100.0, 100.0),
        ];
        let mut ibus = vec![Complex64::new(0.0, 0.0); 3];
        let mut lm_r = vec![0.0; 3];
        let (norm_inf, merit) = residual(&ybus, &sbus, &mut ibus, 2, 1, &v, &mut lm_r);
        let mut nr_r = vec![0.0; 3];
        let nr_norm = fill_f_from_scalc::<false>(&v, &sbus, 1, 2, &mut nr_r);
        assert_eq!(lm_r, vec![0.75, 0.5, 0.75]);
        assert_eq!(lm_r, nr_r);
        assert_eq!(norm_inf, 0.75);
        assert_eq!(norm_inf, nr_norm);
        assert_eq!(merit, 0.6875);
    }
}
