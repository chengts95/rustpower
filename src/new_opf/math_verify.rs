use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// BRUTE FORCE Mathematical Verification of the OPF Hessian.
/// This implementation prioritizes mathematical purity and transparency over performance.
pub fn verify_hessian(
    data: &OPFData,
    x: &[f64],
    lam_eq: &[f64],
    mu_ineq: &[f64],
    cost_mult: f64,
) -> CscMatrix<f64> {
    let nb = data.nb;
    let nl = data.nl;
    let ng = data.ng;
    let nx = data.nx();
    let v = data.v_from_x(x);
    let v_s = v.as_slice();
    let ybus = &data.ybus;
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];

    // 1. Rectangular Hessian (H_rect) for Power Balance
    // Define M = diag(Lambda*) * Y_conj
    let mut m_vals = Vec::with_capacity(ybus.nnz());
    for idx in 0..ybus.nnz() {
        let i = y_ri[idx];
        m_vals.push(Complex64::new(lp[i], -lq[i]) * y_vals[idx].conj());
    }
    let m_mat = CscMatrix::try_from_csc_data(nb, nb, y_cp.to_vec(), y_ri.to_vec(), m_vals).unwrap();
    let m_t = m_mat.transpose();

    // H_ee = H_ff = Re( M + M^T )
    // H_ef = -H_fe = Im( M - M^T )
    let h_sum = &m_mat + &m_t;
    let h_diff = &m_mat - &m_t;
    
    let mut h_ee_vals = Vec::with_capacity(h_sum.nnz());
    for v in h_sum.values() { h_ee_vals.push(v.re); }
    let h_ee = CscMatrix::try_from_csc_data(nb, nb, h_sum.col_offsets().to_vec(), h_sum.row_indices().to_vec(), h_ee_vals).unwrap();

    let mut h_ef_vals = Vec::with_capacity(h_diff.nnz());
    for v in h_diff.values() { h_ef_vals.push(v.im); }
    let h_ef = CscMatrix::try_from_csc_data(nb, nb, h_diff.col_offsets().to_vec(), h_diff.row_indices().to_vec(), h_ef_vals).unwrap();

    // 3. Transformation Matrix J_trans (Full nx x nx)
    let mut j_trans_coo = CooMatrix::<f64>::new(nx, nx);
    let mut vmag = vec![0.0f64; nb];
    for i in 0..nb {
        let m = v_s[i].norm().max(1e-9);
        vmag[i] = m;
        let sin = v_s[i].im / m;
        let cos = v_s[i].re / m;
        
        j_trans_coo.push(i, i, -v_s[i].im);    // d_vre / d_th
        j_trans_coo.push(i, nb + i, cos);      // d_vre / d_Vm
        j_trans_coo.push(nb + i, i, v_s[i].re); // d_vim / d_th
        j_trans_coo.push(nb + i, nb + i, sin); // d_vim / d_Vm
    }
    for g in 0..2 * ng {
        j_trans_coo.push(2 * nb + g, 2 * nb + g, 1.0);
    }
    let j_trans = CscMatrix::from(&j_trans_coo);

    // 4. Delta_polar Correction (Curvature of Node Power Balance ONLY)
    let ibus = ybus * &v;
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb { lam_v_conj[i] = (Complex64::new(lp[i], -lq[i]) * v[i]).conj(); }
    let term2 = ybus * &lam_v_conj;
    let mut g_rect_eq = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb { g_rect_eq[i] = Complex64::new(lp[i], -lq[i]) * ibus[i].conj() + term2[i]; }

    let mut h_full_coo = CooMatrix::new(nx, nx);

    // A. Add H_eq transformed
    let mut h_rect_full_coo = CooMatrix::new(2 * nb, 2 * nb);
    let add_blk = |coo: &mut CooMatrix<f64>, blk: &CscMatrix<f64>, r_off: usize, c_off: usize| {
        for col in 0..blk.ncols() {
            for idx in blk.col_offsets()[col]..blk.col_offsets()[col+1] {
                coo.push(r_off + blk.row_indices()[idx], c_off + col, blk.values()[idx]);
            }
        }
    };
    add_blk(&mut h_rect_full_coo, &h_ee, 0, 0);
    add_blk(&mut h_rect_full_coo, &h_ee, nb, nb);
    add_blk(&mut h_rect_full_coo, &h_ef, 0, nb);
    let h_fe = h_ef.transpose();
    for j in 0..nb {
        for idx in h_fe.col_offsets()[j]..h_fe.col_offsets()[j+1] {
            h_rect_full_coo.push(nb + h_fe.row_indices()[idx], j, -h_fe.values()[idx]);
        }
    }
    let h_rect_full = CscMatrix::from(&h_rect_full_coo);
    
    // Manual transformation for verification without view()
    let mut j_trans_eq_coo = CooMatrix::new(2 * nb, 2 * nb);
    for j in 0..2 * nb {
        for idx in j_trans.col_offsets()[j]..j_trans.col_offsets()[j + 1] {
            j_trans_eq_coo.push(j_trans.row_indices()[idx], j, j_trans.values()[idx]);
        }
    }
    let j_trans_eq = CscMatrix::from(&j_trans_eq_coo);
    let h_polar_eq = &j_trans_eq.transpose() * &(&h_rect_full * &j_trans_eq);
    
    for col in 0..2*nb {
        for idx in h_polar_eq.col_offsets()[col]..h_polar_eq.col_offsets()[col+1] {
            h_full_coo.push(h_polar_eq.row_indices()[idx], col, h_polar_eq.values()[idx]);
        }
    }

    // B. Add Delta_polar for EQ only
    for i in 0..nb {
        let m = vmag[i];
        let vre = v_s[i].re;
        let vim = v_s[i].im;
        let gre = g_rect_eq[i].re;
        let gim = -g_rect_eq[i].im;
        
        let d_aa = -(gre * vre + gim * vim);
        let d_av = (gim * vre - gre * vim) / m;
        h_full_coo.push(i, i, d_aa);
        h_full_coo.push(i, nb + i, d_av);
        h_full_coo.push(nb + i, i, d_av);
    }

    // C. Add Branch Hessian (using traditional for verification)
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..2*nl];
    let v_norm: DVector<Complex64> = v.map(|vi| vi / vi.norm());
    let (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St) =
        crate::basic::dsbr_dv::dSbr_dV(&data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm);
    let hf = crate::basic::d2sbr_dv2::d2ASbr_dV2(&dSf_dVa, &dSf_dVm, &Sf, &data.cf, &data.yf, &v, &DVector::from_column_slice(mu_f));
    let ht = crate::basic::d2sbr_dv2::d2ASbr_dV2(&dSt_dVa, &dSt_dVm, &St, &data.ct, &data.yt, &v, &DVector::from_column_slice(mu_t));

    let add_blocks = |coo: &mut CooMatrix<f64>, blocks: (CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>)| {
        let (haa, hav, hva, hvv) = blocks;
        let blks = [haa, hav, hva, hvv];
        for (b_idx, block) in blks.iter().enumerate() {
            let r_off = (b_idx / 2) * nb;
            let c_off = (b_idx % 2) * nb;
            for j in 0..nb {
                for idx in block.col_offsets()[j]..block.col_offsets()[j+1] {
                    coo.push(r_off + block.row_indices()[idx], c_off + j, block.values()[idx]);
                }
            }
        }
    };
    add_blocks(&mut h_full_coo, hf);
    add_blocks(&mut h_full_coo, ht);

    // D. Add Cost Hessian
    let base = data.base_mva;
    for g in 0..ng {
        h_full_coo.push(2 * nb + g, 2 * nb + g, cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base);
    }

    CscMatrix::from(&h_full_coo)
}
