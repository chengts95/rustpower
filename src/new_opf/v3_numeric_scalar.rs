use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::v3_symbolic::V3SymbolicCache;
use crate::opf::problem::OPFData;

use crate::basic::d2sbr_dv2::d2ASbr_dV2;
use crate::basic::dsbr_dv::dSbr_dV;

/// V3 Scalar FMA Numeric Fill.
/// 
/// Implements the "Imaginary Annihilation" logic from infer.md.
/// Each Ybus NNZ is processed with complex scalar FMA instead of 2x2 matrix ops.
pub fn v3_scalar_numeric_fill(
    data: &OPFData,
    cache: &V3SymbolicCache,
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

    // --- 1. Precompute State and Scaled Multipliers ---
    let mut vre = vec![0.0f64; nb];
    let mut vim = vec![0.0f64; nb];
    let mut vmag = vec![0.0f64; nb];
    let mut cos_th = vec![0.0f64; nb];
    let mut sin_th = vec![0.0f64; nb];
    for i in 0..nb {
        vre[i] = v_s[i].re;
        vim[i] = v_s[i].im;
        let m = v_s[i].norm().max(1e-9);
        vmag[i] = m;
        cos_th[i] = vre[i] / m;
        sin_th[i] = vim[i] / m;
    }
    
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    
    // Auxiliary vectors for scalar FMA:
    // term1[i] = Lambda_i* * V_i
    let mut term1 = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        term1[i] = Complex64::new(lp[i], -lq[i]) * v[i];
    }
    
    let ibus = ybus * &v;
    // term2 = Y * (Lambda* * V)*
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb { lam_v_conj[i] = term1[i].conj(); }
    let term2 = ybus * &lam_v_conj;

    let mut lxx_vals = vec![0.0f64; cache.lxx_cp[nx]];

    // --- 2. Pass 1: Power Balance Hessian (The Scalar FMA Revolution) ---
    let y_vals = ybus.values();
    let yf_vals = data.yf.values();
    let yt_vals = data.yt.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();
    let y_trans = &cache.y_transpose_idx;

    for j in 0..nb {
        for idx in y_cp[j]..y_cp[j+1] {
            let i = y_ri[idx];
            let y_ik_conj = y_vals[idx].conj();
            let y_ki_conj = y_vals[y_trans[idx]].conj();

            // M_ik = (Lambda_i* * V_i) * (Y_ik* * V_k*)
            // M_ki = (Lambda_k* * V_k) * (Y_ki* * V_i*)
            let mik = term1[i] * y_ik_conj * v[j].conj();
            let mki = term1[j] * y_ki_conj * v[i].conj();

            // Core Theorem 4: Pull-back to Polar via Scalar combination
            // These formulas represent the Re{ M^H H_C M } annihilation.
            let haa = -(mik + mki.conj()).re;
            let hvv = (mik + mki.conj()).re / (vmag[i] * vmag[j]);
            let hav = (mik - mki.conj()).im / vmag[j];
            let hva = -(mik - mki.conj()).im / vmag[i];

            let ptrs = cache.y_to_lxx[idx];
            lxx_vals[ptrs[0]] = haa;
            lxx_vals[ptrs[1]] = hav;
            lxx_vals[ptrs[2]] = hva;
            lxx_vals[ptrs[3]] = hvv;
        }
    }

    // --- 3. Pass 2: Branch Flow Limits (Revolutionary Fused Assembly) ---
    // For |Sf|^2 and |St|^2 limits, we build the 4x4 polar Hessian directly.
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];

    let apply_mc_real = |i: usize, k: usize, h_rect: nalgebra::Matrix4<f64>| -> nalgebra::Matrix4<f64> {
        let mi = [ -vim[i], cos_th[i], vre[i], sin_th[i] ];
        let mk = [ -vim[k], cos_th[k], vre[k], sin_th[k] ];
        
        let mut m_mat = nalgebra::Matrix4::zeros();
        m_mat[(0, 0)] = mi[0]; m_mat[(0, 2)] = mi[1];
        m_mat[(1, 1)] = mk[0]; m_mat[(1, 3)] = mk[1];
        m_mat[(2, 0)] = mi[2]; m_mat[(2, 2)] = mi[3];
        m_mat[(3, 1)] = mk[2]; m_mat[(3, 3)] = mk[3];
        
        m_mat.transpose() * h_rect * m_mat
    };

    for l in 0..nl {
        let mu_f_val = mu_f[l];
        let mu_t_val = mu_t[l];
        if mu_f_val == 0.0 && mu_t_val == 0.0 { continue; }
        
        let f = data.f_buses[l];
        let t = data.t_buses[l];
        
        let y_ff = yf_vals[cache.br_to_yf_idx[l][0]];
        let y_ft = yf_vals[cache.br_to_yf_idx[l][1]];
        let y_tf = yt_vals[cache.br_to_yt_idx[l][0]];
        let y_tt = yt_vals[cache.br_to_yt_idx[l][1]];

        let compute_br_h = |i: usize, k: usize, y_ii: Complex64, y_ik: Complex64, mu: f64| -> (nalgebra::Matrix4<f64>, [f64; 2], [f64; 2]) {
            if mu == 0.0 { return (nalgebra::Matrix4::zeros(), [0.0, 0.0], [0.0, 0.0]); }
            let vi = v[i];
            let vk = v[k];
            let i_flow = y_ii * vi + y_ik * vk;
            let p = (vi * i_flow.conj()).re;
            let q = (vi * i_flow.conj()).im;
            
            // J_P = dP/d[ei, fi, ek, fk], J_Q = dQ/d[ei, fi, ek, fk]
            // S = (ei+jfi) * ( (Gii-jBii)(ei-jfi) + (Gik-jBik)(ek-jfk) )
            let g_ii = y_ii.re; let b_ii = y_ii.im;
            let g_ik = y_ik.re; let b_ik = y_ik.im;
            
            let i_re = i_flow.re; let i_im = i_flow.im;
            let ei = vi.re; let fi = vi.im;
            let ek = vk.re; let fk = vk.im;

            let jp = [ i_re + ei*g_ii + fi*b_ii, i_im - ei*b_ii + fi*g_ii, ei*g_ik + fi*b_ik, -ei*b_ik + fi*g_ik ];
            let jq = [ i_im + ei*b_ii - fi*g_ii, -i_re + ei*g_ii + fi*b_ii, ei*b_ik - fi*g_ik, ei*g_ik + fi*b_ik ];
            
            let jp_vec = nalgebra::RowVector4::from_column_slice(&jp);
            let jq_vec = nalgebra::RowVector4::from_column_slice(&jq);
            
            // H_rect = 2*mu*(Jp^T Jp + Jq^T Jq) + 2*mu*P*Hg + 2*mu*Q*Hb
            let mut h_rect = (jp_vec.transpose() * jp_vec + jq_vec.transpose() * jq_vec) * (2.0 * mu);
            
            // Hg: d2P/dX^2.  P = ei*ire + fi*iim.
            // Non-zeros: (ei,ei)=2*Gii, (fi,fi)=2*Gii, (ei,ek)=Gik, (fi,fk)=Gik, (ei,fk)=-Bik, (fi,ek)=Bik
            let scale_g = 2.0 * mu * p;
            h_rect[(0, 0)] += scale_g * 2.0 * g_ii;
            h_rect[(1, 1)] += scale_g * 2.0 * g_ii;
            h_rect[(0, 2)] += scale_g * g_ik; h_rect[(2, 0)] += scale_g * g_ik;
            h_rect[(1, 3)] += scale_g * g_ik; h_rect[(3, 1)] += scale_g * g_ik;
            h_rect[(0, 3)] -= scale_g * b_ik; h_rect[(3, 0)] -= scale_g * b_ik;
            h_rect[(1, 2)] += scale_g * b_ik; h_rect[(2, 1)] += scale_g * b_ik;
            
            // Hb: d2Q/dX^2.  Q = fi*ire - ei*iim.
            let scale_b = 2.0 * mu * q;
            h_rect[(0, 0)] += scale_b * 2.0 * b_ii;
            h_rect[(1, 1)] += scale_b * 2.0 * b_ii;
            h_rect[(0, 2)] += scale_b * b_ik; h_rect[(2, 0)] += scale_b * b_ik;
            h_rect[(1, 3)] += scale_b * b_ik; h_rect[(3, 1)] += scale_b * b_ik;
            h_rect[(0, 3)] += scale_b * g_ik; h_rect[(3, 0)] += scale_b * g_ik;
            h_rect[(1, 2)] -= scale_b * g_ik; h_rect[(2, 1)] -= scale_b * g_ik;
            
            let g_rect = [ 2.0*mu*(p*jp[0] + q*jq[0]), 2.0*mu*(p*jp[1] + q*jq[1]) ];
            let g_rect_k = [ 2.0*mu*(p*jp[2] + q*jq[2]), 2.0*mu*(p*jp[3] + q*jq[3]) ];
            (h_rect, g_rect, g_rect_k)
        };

        let (hr_f, gr_f, gr_f_k) = compute_br_h(f, t, y_ff, y_ft, mu_f_val);
        let (hr_t_swapped, gr_t, gr_t_f) = compute_br_h(t, f, y_tt, y_tf, mu_t_val);
        
        // Combine T end back to F-T order
        let mut hr_t = nalgebra::Matrix4::zeros();
        let s = [1, 0, 3, 2];
        for r in 0..4 { for c in 0..4 { hr_t[(r, c)] = hr_t_swapped[(s[r], s[c])]; } }
        
        let h_polar = apply_mc_real(f, t, hr_f + hr_t);
        
        // Part B: Curvature correction
        let g_total_f = [ gr_f[0] + gr_t_f[0], gr_f[1] + gr_t_f[1] ];
        let g_total_t = [ gr_t[0] + gr_f_k[0], gr_t[1] + gr_f_k[1] ];
        
        let mut add_part_b = |node: usize, g_rect: [f64; 2], h_p: &mut nalgebra::Matrix4<f64>, local_off: usize| {
            let m = vmag[node];
            let cos = vre[node] / m;
            let sin = vim[node] / m;
            let d_aa = -(g_rect[0] * vre[node] + g_rect[1] * vim[node]);
            let d_av = (-sin * g_rect[0] + cos * g_rect[1]);
            h_p[(local_off, local_off)] += d_aa;
            h_p[(local_off, local_off+2)] += d_av;
            h_p[(local_off+2, local_off)] += d_av;
        };
        
        let mut h_polar_total = h_polar;
        add_part_b(f, g_total_f, &mut h_polar_total, 0);
        add_part_b(t, g_total_t, &mut h_polar_total, 1);

        let ptrs = cache.br_to_lxx[l];
        for r in 0..4 { for c in 0..4 { lxx_vals[ptrs[r * 4 + c]] += h_polar_total[(r, c)]; } }
    }

    // --- 4. Delta_polar Corrections ---
    for i in 0..nb {
        let zi = term1[i] / v[i] * ibus[i].conj() + term2[i]; // (Lambda_i* * I_i*) + term2
        let zv = zi * v[i];
        lxx_vals[cache.lxx_diag_ptrs[i]] += -zv.re;
        lxx_vals[cache.lxx_av_diag_ptrs[i]] += -zv.im / vmag[i];
        lxx_vals[cache.lxx_va_diag_ptrs[i]] += -zv.im / vmag[i];
    }

    // --- 5. Cost Hessian ---
    let base = data.base_mva;
    for g in 0..ng {
        lxx_vals[cache.lxx_diag_ptrs[2 * nb + g]] = cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base;
    }

    CscMatrix::try_from_csc_data(nx, nx, cache.lxx_cp.clone(), cache.lxx_ri.clone(), lxx_vals).unwrap()
}
