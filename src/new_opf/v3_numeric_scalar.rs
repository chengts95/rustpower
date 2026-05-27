use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::v3_symbolic::V3SymbolicCache;
use crate::opf::problem::OPFData;

/// V3 Scalar FMA Numeric Fill.
/// 
/// Implements the "Imaginary Annihilation" logic from TN2.
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

    // --- 1. Precompute per-bus state and multipliers ---
    let mut vmag = vec![0.0f64; nb];
    let mut inv_vmag = vec![0.0f64; nb];
    for i in 0..nb {
        vmag[i] = v_s[i].norm().max(1e-9);
        inv_vmag[i] = 1.0 / vmag[i];
    }
    
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    let mut lam_c = vec![Complex64::new(0.0, 0.0); nb];
    let mut lam_v = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        lam_c[i] = Complex64::new(lp[i], -lq[i]); // Lagrangian uses Lambda*
        lam_v[i] = lam_c[i] * v_s[i];
    }

    let ibus: DVector<Complex64> = ybus * &v;
    let ibus_s = ibus.as_slice();

    // d_lam[i] = (Ybus^H * lam_v)[i]
    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();
    for i in 0..nb {
        for idx in y_cp[i]..y_cp[i+1] {
            d_lam[i] += y_vals[idx].conj() * lam_v[y_ri[idx]];
        }
    }

    let mut lxx_vals = vec![0.0f64; cache.lxx_cp[nx]];

    // --- 2. Pass 1: Node Power Balance (Full TN2 Alignment) ---
    let y_trans = &cache.y_transpose_idx;
    let j_unit = Complex64::i();

    for j in 0..nb {
        for idx in y_cp[j]..y_cp[j+1] {
            let i = y_ri[idx];
            let y_ij = y_vals[idx];
            
            // TN2 Core Components for entry (i,j)
            let c_ij = lam_v[i] * (y_ij * v_s[j]).conj();
            let e_ij = if i == j {
                v_s[i].conj() * (y_ij.conj() * lam_v[i] - d_lam[i])
            } else {
                let ji_idx = y_trans[idx];
                v_s[i].conj() * y_vals[ji_idx].conj() * lam_v[j]
            };
            let f_ij = if i == j {
                c_ij - lam_v[i] * ibus_s[i].conj()
            } else {
                c_ij
            };

            // Polar Hessian block entries for (i,j)
            let gaa = (e_ij + f_ij).re;
            let gva = (j_unit * inv_vmag[i] * (e_ij - f_ij)).re;
            
            // Gvv requires C[j,i]
            let ji_idx = y_trans[idx];
            let c_ji = lam_v[j] * (y_vals[ji_idx] * v_s[i]).conj();
            let gvv = (inv_vmag[i] * (c_ij + c_ji) * inv_vmag[j]).re;
            
            // Gav[i,j] component: Gva is symmetric to Gav, so Gav[i,j] = Gva[j,i]
            // We calculate Gav[i,j] using the transposed formula components.
            let gav = (j_unit * inv_vmag[j] * ( (v_s[j].conj() * y_vals[idx].conj() * lam_v[i]) - c_ji )).re;
            
            let ptrs = cache.y_to_lxx[idx];
            // Assign values to Lxx slots (one-to-one mapping)
            lxx_vals[ptrs[0]] = gaa;
            lxx_vals[ptrs[1]] = gav;
            lxx_vals[ptrs[2]] = gva;
            lxx_vals[ptrs[3]] = gvv;
        }
    }

    // --- 3. Pass 2: Branch Flow Limits ---
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
    let v_norm_vec: DVector<Complex64> = v.map(|vi| vi / vi.norm());
    let (d_sf_d_va, d_sf_d_vm, d_st_d_va, d_st_d_vm, sf, st) =
        crate::basic::dsbr_dv::dSbr_dV(&data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm_vec);

    let hf = crate::basic::d2sbr_dv2::d2ASbr_dV2(&d_sf_d_va, &d_sf_d_vm, &sf, &data.cf, &data.yf, &v, &DVector::from_column_slice(mu_f));
    let ht = crate::basic::d2sbr_dv2::d2ASbr_dV2(&d_st_d_va, &d_st_d_vm, &st, &data.ct, &data.yt, &v, &DVector::from_column_slice(mu_t));

    let mut add_br_h = |lxx: &mut [f64], h_blocks: (CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>, CscMatrix<f64>)| {
        let (haa, hav, hva, hvv) = h_blocks;
        let blks = [haa, hav, hva, hvv];
        for (b_idx, block) in blks.iter().enumerate() {
            let cp = block.col_offsets();
            let ri = block.row_indices();
            let vals = block.values();
            let r_off = (b_idx % 2) * nb;
            let c_off = (b_idx / 2) * nb;
            for col in 0..nb {
                for idx in cp[col]..cp[col+1] {
                    let row = ri[idx];
                    let val = vals[idx];
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

    // --- 4. Total Polar Gradient and Curvature Correction ---
    let mut g_polar = vec![0.0f64; nx];
    let (_, _, dg, dh) = crate::opf::constraints::opf_consfcn(data, x);
    
    for j in 0..2 * nb {
        let lam = lam_eq[j];
        for idx in dg.col_offsets()[j]..dg.col_offsets()[j+1] {
            g_polar[dg.row_indices()[idx]] += lam * dg.values()[idx];
        }
    }
    for j in 0..2 * nl {
        let mu = mu_ineq[j];
        for idx in dh.col_offsets()[j]..dh.col_offsets()[j+1] {
            g_polar[dh.row_indices()[idx]] += mu * dh.values()[idx];
        }
    }

    for i in 0..nb {
        let g_th = g_polar[i];
        let g_vm = g_polar[nb + i];
        let m = vmag[i];
        lxx_vals[cache.lxx_diag_ptrs[i]] += -m * g_vm;
        lxx_vals[cache.lxx_va_diag_ptrs[i]] += g_th / m;
        lxx_vals[cache.lxx_av_diag_ptrs[i]] += g_th / m;
    }

    // --- 5. Cost Hessian ---
    let base = data.base_mva;
    for g in 0..ng {
        lxx_vals[cache.lxx_diag_ptrs[2 * nb + g]] = cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base;
    }

    CscMatrix::try_from_csc_data(nx, nx, cache.lxx_cp.clone(), cache.lxx_ri.clone(), lxx_vals).unwrap()
}
