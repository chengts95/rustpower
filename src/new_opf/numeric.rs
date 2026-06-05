use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::symbolic::SymbolicCache;
use crate::opf::problem::OPFData;

use crate::basic::d2sbr_dv2::d2ASbr_dV2;
use crate::basic::dsbr_dv::dSbr_dV;

/// Optimized single-pass numeric fill for OPF Jacobian and Hessian.
/// Implements Algorithm 2 from the paper with Rectangular-to-Polar transformation.
pub fn numeric_fill(
    data: &OPFData,
    cache: &SymbolicCache,
    x: &[f64],
    lam_eq: &[f64],
    mu_ineq: &[f64],
    cost_mult: f64,
) -> (CscMatrix<f64>, CscMatrix<f64>) {
    let nb = data.nb;
    let nl = data.nl;
    let ng = data.ng;
    let nx = data.nx();
    let v = data.v_from_x(x);
    let v_s = v.as_slice();
    let ybus = &data.ybus;

    // 1. Precompute per-bus transformation data (O(n))
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

    // J_trans per bus (2x2):
    // [ d_vre/d_th  d_vre/d_Vm ] = [ -vim  cos ]
    // [ d_vim/d_th  d_vim/d_Vm ]   [  vre  sin ]
    let jt = |i: usize| [
        -vim[i], cos_th[i],
         vre[i], sin_th[i]
    ];

    // 2. Evaluate equality multipliers and auxiliary vectors (O(nnz))
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    let mut lam_vec = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb {
        lam_vec[i] = Complex64::new(lp[i], -lq[i]); // Lambda* = lamP - j*lamQ
        lam_v_conj[i] = (lam_vec[i] * v[i]).conj();
    }

    let ibus = ybus * &v;
    let term2 = ybus * &lam_v_conj;

    let mut lxx_vals = vec![0.0f64; cache.lxx_template.nnz()];
    let mut dg_vals = vec![0.0f64; cache.dg_template.nnz()];

    // 3. Power Balance Jacobian and Hessian (Revolutionary Single-Pass Fill)
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    for j in 0..nb {
        for idx in y_cp[j]..y_cp[j+1] {
            let i = y_ri[idx];
            let y_conj = y_vals[idx].conj();
            
            // --- A. Jacobian (dg) ---
            // Rectangular: dPj/dVi, dQj/dVi
            let (jr_re, jr_im, ji_re, ji_im) = if i == j {
                let ire = ibus[j].re;
                let iim = ibus[j].im;
                let g = y_vals[idx].re;
                let b = y_vals[idx].im;
                (g*vre[j] + b*vim[j] + ire, -b*vre[j] + g*vim[j] + iim,
                 g*vim[j] - b*vre[j] - iim, -(b*vim[j] + g*vre[j]) + ire)
            } else {
                let g = y_vals[idx].re;
                let b = y_vals[idx].im;
                (g*vre[j] + b*vim[j], -b*vre[j] + g*vim[j],
                 g*vim[j] - b*vre[j], -(b*vim[j] + g*vre[j]))
            };

            let m_i = jt(i);
            // Transform row i of Jacobian to polar
            let dp_dth = jr_re * m_i[0] + jr_im * m_i[2];
            let dp_dvm = jr_re * m_i[1] + jr_im * m_i[3];
            let dq_dth = ji_re * m_i[0] + ji_im * m_i[2];
            let dq_dvm = ji_re * m_i[1] + ji_im * m_i[3];

            let g_ptrs = cache.y_to_dg[idx];
            dg_vals[g_ptrs[0]] = dp_dth;
            dg_vals[g_ptrs[1]] = dp_dvm;
            dg_vals[g_ptrs[2]] = dq_dth;
            dg_vals[g_ptrs[3]] = dq_dvm;

            // --- B. Hessian (Lxx) ---
            // Rectangular: M_ij = Lambda_i* * Y_ij*
            let mij = lam_vec[i] * y_conj;
            let ji_idx = cache.y_transpose_idx[idx];
            let m_ji = lam_vec[j] * y_vals[ji_idx].conj();

            // H_rect blocks
            let hrr = mij.re + m_ji.re;
            let hii = hrr;
            let h_ef = mij.im - m_ji.im;
            let h_fe = -h_ef;

            // Transform to polar: H_p = Ji^T * H_r * Jj
            let m_j = jt(j);
            let haa = m_i[0]*(hrr*m_j[0] + h_ef*m_j[2]) + m_i[2]*(h_fe*m_j[0] + hii*m_j[2]);
            let hav = m_i[0]*(hrr*m_j[1] + h_ef*m_j[3]) + m_i[2]*(h_fe*m_j[1] + hii*m_j[3]);
            let hva = m_i[1]*(hrr*m_j[0] + h_ef*m_j[2]) + m_i[3]*(h_fe*m_j[0] + hii*m_j[2]);
            let hvv = m_i[1]*(hrr*m_j[1] + h_ef*m_j[3]) + m_i[3]*(h_fe*m_j[1] + hii*m_j[3]);

            let h_ptrs = cache.y_to_lxx[idx];
            lxx_vals[h_ptrs[0]] = haa;
            lxx_vals[h_ptrs[1]] = hav;
            lxx_vals[h_ptrs[2]] = hva;
            lxx_vals[h_ptrs[3]] = hvv;
        }
    }

    // 4. Delta_polar correction (diagonal O(n))
    for i in 0..nb {
        let zi = lam_vec[i] * ibus[i].conj() + term2[i];
        let gre = zi.re;
        let gim = -zi.im;
        let d_aa = -(gre * vre[i] + gim * vim[i]);
        let d_av = (gim * vre[i] - gre * vim[i]) / vmag[i];
        
        let ptrs = cache.y_to_lxx[cache.y_diag_ptrs[i]];
        lxx_vals[ptrs[0]] += d_aa;
        lxx_vals[ptrs[1]] += d_av;
        lxx_vals[ptrs[2]] += d_av;
    }

    // 5. Cost Hessian and Gen terms in dg
    for g in 0..ng {
        dg_vals[cache.pg_dg_ptrs[g]] = -1.0;
        dg_vals[cache.qg_dg_ptrs[g]] = -1.0;
        let val = cost_mult * 2.0 * data.cost_coeffs[g][0] * data.base_mva * data.base_mva;
        lxx_vals[cache.h_diag_ptrs[2 * nb + g]] = val;
    }

    // 6. Add Branch Flow Hessian (Revolutionary Optimized Fill)
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
    let v_norm: DVector<Complex64> = v.map(|vi| vi / vi.norm());
    let (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St) =
        dSbr_dV(&data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm);

    let hf = d2ASbr_dV2(&dSf_dVa, &dSf_dVm, &Sf, &data.cf, &data.yf, &v, &DVector::from_column_slice(mu_f));
    let ht = d2ASbr_dV2(&dSt_dVa, &dSt_dVm, &St, &data.ct, &data.yt, &v, &DVector::from_column_slice(mu_t));

    let add_br_h = |lxx: &mut [f64], h_blocks: (CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>), _is_to: bool| {
        let (haa, hav, hva, hvv) = h_blocks;
        // Each branch l contributes to its nodes f, t.
        // d2Sbr/dV2 is a 2x2 block matrix (for nodes f, t).
        // MATPOWER's d2ASbr_dV2 returns the sum of these contributions for all branches.
        // To do this truly single-pass, we'd need to modify d2ASbr_dV2.
        // For now, since haa etc are nb x nb, we'll iterate their NNZ.
        // But wait! haa NNZ is already mapped to Ybus NNZ.
        
        let blocks = [haa, hav, hva, hvv];
        for (b_idx, block) in blocks.iter().enumerate() {
            let cp = block.col_offsets();
            let ri = block.row_indices();
            let vals = block.values();
            for col in 0..nb {
                for idx in cp[col]..cp[col+1] {
                    let row = ri[idx];
                    let val = vals[idx];
                    // Find position in Lxx. We use a faster way if we know the Ybus structure.
                    // Actually, since haa sparsity is a subset of Ybus, we can find the Ybus idx.
                    if let Some(y_idx) = find_nnz_cx(ybus, row, col) {
                        let ptr = cache.y_to_lxx[y_idx][b_idx];
                        lxx[ptr] += val;
                    }
                }
            }
        }
    };
    add_br_h(&mut lxx_vals, hf, false);
    add_br_h(&mut lxx_vals, ht, true);

    (
        CscMatrix::try_from_csc_data(nx, nx, cache.lxx_template.col_offsets().to_vec(), cache.lxx_template.row_indices().to_vec(), lxx_vals).unwrap(),
        CscMatrix::try_from_csc_data(nx, 2*nb, cache.dg_template.col_offsets().to_vec(), cache.dg_template.row_indices().to_vec(), dg_vals).unwrap()
    )
}

fn find_nnz_cx(mat: &CscMatrix<Complex64>, r: usize, c: usize) -> Option<usize> {
    let range = mat.col_offsets()[c]..mat.col_offsets()[c + 1];
    mat.row_indices()[range.clone()].binary_search(&r).ok().map(|pos| range.start + pos)
}
