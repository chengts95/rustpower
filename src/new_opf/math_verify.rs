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

    // 1. Rectangular Hessian (H_rect) for Power Balance
    // Lagrangian: L_eq = sum lamP*P + sum lamQ*Q
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    
    // M = diag(Lambda*) * Y_conj
    let mut m_re_coo = CooMatrix::<f64>::new(nb, nb);
    let mut m_im_coo = CooMatrix::<f64>::new(nb, nb);
    for j in 0..nb {
        for idx in ybus.col_offsets()[j]..ybus.col_offsets()[j+1] {
            let i = ybus.row_indices()[idx];
            let mij = Complex64::new(lp[i], -lq[i]) * ybus.values()[idx].conj();
            m_re_coo.push(i, j, mij.re);
            m_im_coo.push(i, j, mij.im);
        }
    }
    let m_re = CscMatrix::from(&m_re_coo);
    let m_im = CscMatrix::from(&m_im_coo);
    
    // H_eq in rectangular: [ H_ee  H_ef ]
    //                     [ H_fe  H_ff ]
    let h_ee = &m_re + &m_re.transpose();
    let h_ff = h_ee.clone();
    let h_ef = &m_im - &m_im.transpose();
    let h_fe = h_ef.transpose();

    // 2. Rectangular Hessian for Branch Limits (mu * |S|^2)
    // We'll use the traditional path to compute the polar Hessian for branches,
    // then "pull it back" to rectangular space if needed for verification,
    // OR just add it to the final polar Hessian. 
    // To prove the theory, let's keep branches in polar for now but 
    // ENSURE the Delta_polar correction includes their gradient.

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
    // Pg, Qg are identity mappings
    for g in 0..2 * ng {
        j_trans_coo.push(2 * nb + g, 2 * nb + g, 1.0);
    }
    let j_trans = CscMatrix::from(&j_trans_coo);

    // 4. Delta_polar Correction (Curvature of all constraints)
    // Needs Total Gradient in Rectangular coordinates.
    
    // A. Polar Gradient (Jacobian rows * multipliers)
    // Power Balance part: Already derived as Z = Lambda* \circ I* + term2
    let ibus = ybus * &v;
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb { lam_v_conj[i] = (Complex64::new(lp[i], -lq[i]) * v[i]).conj(); }
    let term2 = ybus * &lam_v_conj;
    let mut g_rect_eq = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb { g_rect_eq[i] = Complex64::new(lp[i], -lq[i]) * ibus[i].conj() + term2[i]; }

    // B. Branch Limit Gradient (Polar -> Rectangular)
    let (_, _, _, dh) = crate::opf::constraints::opf_consfcn(data, x);
    let mut g_polar_ineq = vec![0.0f64; 2 * nb];
    for l in 0..2 * nl {
        if mu_ineq[l] == 0.0 { continue; }
        for idx in dh.col_offsets()[l]..dh.col_offsets()[l+1] {
            let var = dh.row_indices()[idx];
            if var < 2 * nb { g_polar_ineq[var] += mu_ineq[l] * dh.values()[idx]; }
        }
    }
    
    // Map polar ineq gradient back to rectangular
    let mut g_rect_total_re = vec![0.0f64; nb];
    let mut g_rect_total_im = vec![0.0f64; nb];
    for i in 0..nb {
        let m = vmag[i];
        let sin = v_s[i].im / m;
        let cos = v_s[i].re / m;
        
        let gre_br = (cos * m * g_polar_ineq[nb + i] - sin * g_polar_ineq[i]) / m;
        let gim_br = (sin * m * g_polar_ineq[nb + i] + cos * g_polar_ineq[i]) / m;
        
        g_rect_total_re[i] = g_rect_eq[i].re + gre_br;
        g_rect_total_im[i] = -g_rect_eq[i].im + gim_br; // Z = gre - j*gim
    }

    // 5. Build Final Hessian
    let mut h_full_coo = CooMatrix::new(nx, nx);

    // A. Add H_eq transformed
    let mut h_rect_full_coo = CooMatrix::new(nx, nx);
    let add_blk = |coo: &mut CooMatrix<f64>, blk: &CscMatrix<f64>, r_off: usize, c_off: usize| {
        for col in 0..blk.ncols() {
            for idx in blk.col_offsets()[col]..blk.col_offsets()[col+1] {
                coo.push(r_off + blk.row_indices()[idx], c_off + col, blk.values()[idx]);
            }
        }
    };
    add_blk(&mut h_rect_full_coo, &h_ee, 0, 0);
    add_blk(&mut h_rect_full_coo, &h_ff, nb, nb);
    add_blk(&mut h_rect_full_coo, &h_ef, 0, nb);
    add_blk(&mut h_rect_full_coo, &h_fe, nb, 0);
    let h_rect_full = CscMatrix::from(&h_rect_full_coo);
    let h_polar_eq = &j_trans.transpose() * &(&h_rect_full * &j_trans);
    
    for col in 0..nx {
        for idx in h_polar_eq.col_offsets()[col]..h_polar_eq.col_offsets()[col+1] {
            h_full_coo.push(h_polar_eq.row_indices()[idx], col, h_polar_eq.values()[idx]);
        }
    }

    // B. Add Delta_polar
    for i in 0..nb {
        let m = vmag[i];
        let vre = v_s[i].re;
        let vim = v_s[i].im;
        let gre = g_rect_total_re[i];
        let gim = g_rect_total_im[i];
        
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
            let r_off = (b_idx % 2) * nb;
            let c_off = (b_idx / 2) * nb;
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
