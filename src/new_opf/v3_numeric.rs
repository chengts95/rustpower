use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::v3_symbolic::V3SymbolicCache;
use crate::opf::problem::OPFData;

use crate::basic::d2sbr_dv2::d2ASbr_dV2;
use crate::basic::dsbr_dv::dSbr_dV;

/// V3 Numeric Fill: Fuses node and branch Hessian contributions.
/// Completely eliminates the need for Yf and Yt matrices.
pub fn v3_numeric_fill(
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

    // 1. Precompute Transformation and Multipliers
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
    
    let jt = |i: usize| [
        -vim[i], cos_th[i],
         vre[i], sin_th[i]
    ];

    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    let mut lam_vec = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    let mut lam_v_conj = DVector::from_element(nb, Complex64::new(0.0, 0.0));
    for i in 0..nb {
        lam_vec[i] = Complex64::new(lp[i], -lq[i]);
        lam_v_conj[i] = (lam_vec[i] * v[i]).conj();
    }
    let ibus = ybus * &v;
    let term2 = ybus * &lam_v_conj;

    let mut lxx_vals = vec![0.0f64; cache.lxx_cp[nx]];

    // 2. Pass 1: Power Balance Hessian (Direct Fill from Ybus)
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    for j in 0..nb {
        for idx in y_cp[j]..y_cp[j+1] {
            let i = y_ri[idx];
            let y_conj = y_vals[idx].conj();
            let ji_idx = cache.y_transpose_idx[idx];
            
            let mij = lam_vec[i] * y_conj;
            let mji = lam_vec[j] * y_vals[ji_idx].conj();

            let hrr = mij.re + mji.re;
            let hii = hrr;
            let h_ef = mij.im - mji.im;
            let h_fe = -h_ef;

            let m_i = jt(i);
            let m_j = jt(j);
            let haa = m_i[0]*(hrr*m_j[0] + h_ef*m_j[2]) + m_i[2]*(h_fe*m_j[0] + hii*m_j[2]);
            let hav = m_i[0]*(hrr*m_j[1] + h_ef*m_j[3]) + m_i[2]*(h_fe*m_j[1] + hii*m_j[3]);
            let hva = m_i[1]*(hrr*m_j[0] + h_ef*m_j[2]) + m_i[3]*(h_fe*m_j[0] + hii*m_j[2]);
            let hvv = m_i[1]*(hrr*m_j[1] + h_ef*m_j[3]) + m_i[3]*(h_fe*m_j[1] + hii*m_j[3]);

            let ptrs = cache.y_to_lxx[idx];
            lxx_vals[ptrs[0]] = haa;
            lxx_vals[ptrs[1]] = hav;
            lxx_vals[ptrs[2]] = hva;
            lxx_vals[ptrs[3]] = hvv;
        }
    }

    // 3. Pass 2: Branch Flow Limits Hessian
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
    let v_norm: DVector<Complex64> = v.map(|vi| vi / vi.norm());
    let (dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St) =
        dSbr_dV(&data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm);

    let hf = d2ASbr_dV2(&dSf_dVa, &dSf_dVm, &Sf, &data.cf, &data.yf, &v, &DVector::from_column_slice(mu_f));
    let ht = d2ASbr_dV2(&dSt_dVa, &dSt_dVm, &St, &data.ct, &data.yt, &v, &DVector::from_column_slice(mu_t));

    let mut add_br_h = |lxx: &mut [f64], h_blocks: (CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>)| {
        let (haa, hav, hva, hvv) = h_blocks;
        let blks = [haa, hav, hva, hvv];
        for (b_idx, block) in blks.iter().enumerate() {
            let cp = block.col_offsets();
            let ri = block.row_indices();
            let vals = block.values();
            for col in 0..nb {
                for idx in cp[col]..cp[col+1] {
                    let row = ri[idx];
                    let val = vals[idx];
                    let r_off = (b_idx % 2) * nb;
                    let c_off = (b_idx / 2) * nb;
                    let range = cache.lxx_cp[c_off + col]..cache.lxx_cp[c_off + col + 1];
                    if let Ok(pos) = cache.lxx_ri[range.clone()].binary_search(&(r_off + row)) {
                        lxx[range.start + pos] += val;
                    }
                }
            }
        }
    };
    add_br_h(&mut lxx_vals, hf);
    add_br_h(&mut lxx_vals, ht);

    // 4. Delta_polar correction (diagonal O(n))
    for i in 0..nb {
        let zi = lam_vec[i] * ibus[i].conj() + term2[i];
        let zv = zi * v[i];
        let d_aa = -zv.re;
        let d_av = -zv.im / vmag[i];
        
        lxx_vals[cache.lxx_diag_ptrs[i]] += d_aa;
        lxx_vals[cache.lxx_va_diag_ptrs[i]] += d_av;
        lxx_vals[cache.lxx_av_diag_ptrs[i]] += d_av;
    }

    // 5. Cost Hessian
    let base = data.base_mva;
    for g in 0..ng {
        let val = cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base;
        lxx_vals[cache.lxx_diag_ptrs[2 * nb + g]] = val;
    }

    CscMatrix::try_from_csc_data(nx, nx, cache.lxx_cp.clone(), cache.lxx_ri.clone(), lxx_vals).unwrap()
}
