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
    let (y_cp, y_ri, y_v) = (ybus.col_offsets(), ybus.row_indices(), ybus.values());
    for x in ibus.iter_mut() {
        *x = Complex64::new(0.0, 0.0);
    }
    for j in 0..ybus.ncols() {
        for p in y_cp[j]..y_cp[j + 1] {
            ibus[y_ri[p]] += y_v[p] * v[j];
        }
    }
    let mut res_inf = 0.0f64;
    let mut f = 0.0;
    for i in 0..n_act {
        let s = v[i] * ibus[i].conj() - sbus[i];
        out[i] = s.re;
        res_inf = res_inf.max(s.re.abs());
        f += s.re * s.re;
        if i < npq {
            out[n_act + i] = s.im;
            res_inf = res_inf.max(s.im.abs());
            f += s.im * s.im;
        }
    }
    (res_inf, 0.5 * f)
}

/// Test networks shared by the exact and GN drivers: the ill-conditioned
/// 14-bus case (ext_ref case2, renumbering-invariant) and the IEEE39
/// `PowerFlowMat` loader.
#[cfg(all(test, feature = "klu"))]
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
    pub(crate) fn ill_conditioned_case() -> (CscMatrix<Complex64>, usize, usize, Vec<Complex64>, Vec<Complex64>) {
        let order: Vec<usize> = [1, 2, 4, 5, 7, 8, 10, 11, 13, 3, 6, 9, 12, 0].into();
        let ybus = build_ybus(&order);
        let v_star_old = old_v_star();

        // Specified injections from the exact solution (old numbering, but
        // the network is renumbering-invariant — compute with the new one).
        let yv = &ybus
            * &DVector::from_vec(order.iter().map(|&b| v_star_old[b]).collect());
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
