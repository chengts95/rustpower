use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// Verify the Rectangular-to-Polar Hessian transformation math.
/// This uses standard sparse matrix construction (COO) for maximum transparency.
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

    // 1. Rectangular Hessian (H_rect) for Equality Constraints (Power Balance)
    // Lagrangian: L_eq = Re(sum_i Lambda_i* * S_i)
    // H_ee = H_ff = M_re + M_re^T
    // H_ef = M_im - M_im^T
    // H_fe = (H_ef)^T
    // where M = diag(Lambda*) * conj(Y)
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];

    let mut m_re_coo = CooMatrix::<f64>::new(nb, nb);
    let mut m_im_coo = CooMatrix::<f64>::new(nb, nb);
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    for j in 0..nb {
        for idx in y_cp[j]..y_cp[j+1] {
            let i = y_ri[idx];
            let y_conj = y_vals[idx].conj();
            let lam_conj = Complex64::new(lp[i], -lq[i]);
            let mij = lam_conj * y_conj;
            m_re_coo.push(i, j, mij.re);
            m_im_coo.push(i, j, mij.im);
        }
    }
    let m_re = CscMatrix::from(&m_re_coo);
    let m_im = CscMatrix::from(&m_im_coo);

    let h_ee = &m_re + &m_re.transpose();
    let h_ff = h_ee.clone();
    let h_ef = &m_im - &m_im.transpose();
    let h_fe = h_ef.transpose();

    // 2. Transformation Matrix J_trans (2nb x 2nb)
    let mut j_trans_coo = CooMatrix::<f64>::new(2 * nb, 2 * nb);
    for i in 0..nb {
        let vre = v_s[i].re;
        let vim = v_s[i].im;
        let vmag = v_s[i].norm().max(1e-9);
        j_trans_coo.push(i, i, -vim);           // d_vre / d_th
        j_trans_coo.push(i, nb + i, vre / vmag); // d_vre / d_Vm
        j_trans_coo.push(nb + i, i, vre);       // d_vim / d_th
        j_trans_coo.push(nb + i, nb + i, vim / vmag); // d_vim / d_Vm
    }
    let j_trans = CscMatrix::from(&j_trans_coo);

    // 3. Raw Polar Hessian: H_polar_raw = J_trans^T * H_rect * J_trans
    // Construct H_rect from blocks
    let mut h_rect_full_coo = CooMatrix::new(2 * nb, 2 * nb);
    let add_blk = |coo: &mut CooMatrix<f64>, blk: &CscMatrix<f64>, r_off: usize, c_off: usize| {
        for j in 0..nb {
            for idx in blk.col_offsets()[j]..blk.col_offsets()[j+1] {
                coo.push(r_off + blk.row_indices()[idx], c_off + j, blk.values()[idx]);
            }
        }
    };
    add_blk(&mut h_rect_full_coo, &h_ee, 0, 0);
    add_blk(&mut h_rect_full_coo, &h_ff, nb, nb);
    add_blk(&mut h_rect_full_coo, &h_ef, 0, nb);
    add_blk(&mut h_rect_full_coo, &h_fe, nb, 0);
    let h_rect = CscMatrix::from(&h_rect_full_coo);

    let h_polar_raw = &j_trans.transpose() * &(&h_rect * &j_trans);

    // 4. Delta_polar Correction (Diagonal blocks)
    let ibus = ybus * &v;
    let mut lam_vec = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb { lam_vec[i] = Complex64::new(lp[i], -lq[i]); }
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb { lam_v_conj[i] = (lam_vec[i] * v[i]).conj(); }
    let term2 = ybus * &lam_v_conj;

    let mut h_full_coo = CooMatrix::new(nx, nx);

    // Add transformed power balance Hessian to top-left
    for j in 0..2*nb {
        for idx in h_polar_raw.col_offsets()[j]..h_polar_raw.col_offsets()[j+1] {
            h_full_coo.push(h_polar_raw.row_indices()[idx], j, h_polar_raw.values()[idx]);
        }
    }

    for i in 0..nb {
        let vre = v_s[i].re;
        let vim = v_s[i].im;
        let vmag = v_s[i].norm().max(1e-9);
        let zi = lam_vec[i] * ibus[i].conj() + term2[i];
        let gre = zi.re;
        let gim = -zi.im;
        let d_aa = -(gre * vre + gim * vim);
        let d_av = (gim * vre - gre * vim) / vmag;
        h_full_coo.push(i, i, d_aa);
        h_full_coo.push(i, nb + i, d_av);
        h_full_coo.push(nb + i, i, d_av);
    }

    // 5. Add Cost Hessian
    let base = data.base_mva;
    for g in 0..ng {
        let val = cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base;
        h_full_coo.push(2 * nb + g, 2 * nb + g, val);
    }

    // 6. Add Branch Flow Hessian
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
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

    CscMatrix::from(&h_full_coo)
}


