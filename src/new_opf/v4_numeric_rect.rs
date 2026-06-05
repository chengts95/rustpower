use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::v3_symbolic::V3SymbolicCache;
use crate::opf::problem::OPFData;

/// V4/V5 Revolutionary Assembly.
/// Now includes the Slacks Penalty Term (mu/z * dh * dh^T) merged into the branch loop.
pub fn v4_rect_numeric_fill(
    data: &OPFData,
    cache: &V3SymbolicCache,
    x: &[f64],
    lam_eq: &[f64],
    mu_ineq: &[f64],
    z_ineq: Option<&[f64]>, // New: slacks for V5 merged fill
    cost_mult: f64,
) -> CscMatrix<f64> {
    let nb = data.nb;
    let nl = data.nl;
    let ng = data.ng;
    let nx = data.nx();
    let v = data.v_from_x(x);
    let v_s = v.as_slice();
    let ybus = &data.ybus;

    // --- 1. Precompute per-bus state ---
    let mut inv_vmag = vec![0.0f64; nb];
    for i in 0..nb {
        inv_vmag[i] = 1.0 / v_s[i].norm().max(1e-9);
    }
    
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2*nb];
    let mut lam_v = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        lam_v[i] = Complex64::new(lp[i], -lq[i]) * v_s[i];
    }

    let ibus = ybus * &v;
    let ibus_s = ibus.as_slice();

    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    let y_vals = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();
    let y_trans = &cache.y_transpose_idx;
    
    for i in 0..nb {
        for idx in y_cp[i]..y_cp[i+1] {
            d_lam[i] += y_vals[idx].conj() * lam_v[y_ri[idx]];
        }
    }

    let mut lxx_vals = vec![0.0f64; cache.lxx_cp[nx]];
    let lxx_ptr = lxx_vals.as_mut_ptr();

    let j_unit = Complex64::i();

    // --- 2. Node Power Balance ---
    for j in 0..nb {
        let nnz_j = y_cp[j+1] - y_cp[j];
        let out_aa = unsafe { std::slice::from_raw_parts_mut(lxx_ptr.add(cache.lxx_cp[j]), nnz_j) };
        let out_va = unsafe { std::slice::from_raw_parts_mut(lxx_ptr.add(cache.lxx_cp[j] + nnz_j), nnz_j) };
        let out_av = unsafe { std::slice::from_raw_parts_mut(lxx_ptr.add(cache.lxx_cp[nb + j]), nnz_j) };
        let out_vv = unsafe { std::slice::from_raw_parts_mut(lxx_ptr.add(cache.lxx_cp[nb + j] + nnz_j), nnz_j) };

        for offset in 0..nnz_j {
            let idx = y_cp[j] + offset;
            let i = y_ri[idx];
            let y_ij = y_vals[idx];
            let ji_idx = y_trans[idx];
            
            let c_ij = lam_v[i] * (y_ij * v_s[j]).conj();
            
            let (e_ij, f_ij, gav) = if i == j {
                let e = v_s[i].conj() * (y_ij.conj() * lam_v[i] - d_lam[i]);
                let f = c_ij - lam_v[i] * ibus_s[i].conj();
                (e, f, 0.0) 
            } else {
                let y_ji = y_vals[ji_idx];
                let c_ji = lam_v[j] * (y_ji * v_s[i]).conj();
                let e = v_s[i].conj() * y_ji.conj() * lam_v[j];
                let f = c_ij;
                let gav_val = (j_unit * inv_vmag[j] * (v_s[j].conj() * y_ij.conj() * lam_v[i] - c_ji)).re;
                (e, f, gav_val)
            };

            let gaa = (e_ij + f_ij).re;
            let gva = (j_unit * inv_vmag[i] * (e_ij - f_ij)).re;
            
            let c_ji = lam_v[j] * (y_vals[ji_idx] * v_s[i]).conj();
            let gvv = (inv_vmag[i] * (c_ij + c_ji) * inv_vmag[j]).re;
            
            out_aa[offset] = gaa;
            out_va[offset] = gva;
            out_av[offset] = if i == j { gva } else { gav };
            out_vv[offset] = gvv;
        }
    }

    // --- 3. Branch Flow Limits (Now with Slacks Merge!) ---
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
    let yf_vals = data.yf.values();
    let yt_vals = data.yt.values();

    for l in 0..nl {
        let f = data.f_buses[l];
        let t = data.t_buses[l];

        // Slacks penalty weights
        let wf = if let Some(z) = z_ineq { mu_f[l] / z[l] } else { 0.0 };
        let wt = if let Some(z) = z_ineq { mu_t[l] / z[nl + l] } else { 0.0 };

        let hf = branch_end_hess_v4(
            yf_vals[cache.br_to_yf_idx[l][0]].conj(),
            yf_vals[cache.br_to_yf_idx[l][1]].conj(),
            v_s[f], v_s[t], mu_f[l], wf,
        );
        let ht = branch_end_hess_v4(
            yt_vals[cache.br_to_yt_idx[l][1]].conj(),
            yt_vals[cache.br_to_yt_idx[l][0]].conj(),
            v_s[t], v_s[f], mu_t[l], wt,
        );

        let mut hq = [[0.0f64; 4]; 4];
        const P: [usize; 4] = [1, 0, 3, 2];
        for i in 0..4 {
            for k in 0..4 {
                hq[i][k] += hf[i][k];
                hq[P[i]][P[k]] += ht[i][k];
            }
        }

        let ptrs = &cache.br_to_lxx[l];
        for ni in 0..2 {
            for nj in 0..2 {
                let base = (ni * 2 + nj) * 4;
                lxx_vals[ptrs[base + 0]] += hq[ni][nj];
                lxx_vals[ptrs[base + 1]] += hq[ni][2 + nj];
                lxx_vals[ptrs[base + 2]] += hq[2 + ni][nj];
                lxx_vals[ptrs[base + 3]] += hq[2 + ni][2 + nj];
            }
        }
    }

    // --- 4. Cost Hessian ---
    let base = data.base_mva;
    for g in 0..ng {
        lxx_vals[cache.lxx_diag_ptrs[2 * nb + g]] = cost_mult * 2.0 * data.cost_coeffs[g][0] * base * base;
    }

    CscMatrix::try_from_csc_data(nx, nx, cache.lxx_cp.clone(), cache.lxx_ri.clone(), lxx_vals).unwrap()
}

#[inline]
pub fn branch_end_hess_v4(a: Complex64, b: Complex64, v_self: Complex64, v_other: Complex64, mu: f64, w: f64) -> [[f64; 4]; 4] {
    let j = Complex64::i();
    let vms = v_self.norm();
    let vmo = v_other.norm();
    let t = b * v_self * v_other.conj();
    let s = a * (vms * vms) + t;

    // First derivative dS/du
    let d = [j * t, -j * t, 2.0 * a * vms + t / vms, t / vmo];

    // Slack gradient dh/du = 2 * Re(conj(S) * dS/du)
    let mut dh = [0.0f64; 4];
    for i in 0..4 {
        dh[i] = 2.0 * (s.conj() * d[i]).re;
    }

    let mut ss = [[Complex64::new(0.0, 0.0); 4]; 4];
    ss[0][0] = -t;
    ss[0][1] = t;          ss[1][0] = t;
    ss[1][1] = -t;
    ss[0][2] = j * t / vms; ss[2][0] = ss[0][2];
    ss[0][3] = j * t / vmo; ss[3][0] = ss[0][3];
    ss[1][2] = -j * t / vms; ss[2][1] = ss[1][2];
    ss[1][3] = -j * t / vmo; ss[3][1] = ss[1][3];
    ss[2][2] = 2.0 * a;
    ss[2][3] = t / (vms * vmo); ss[3][2] = ss[2][3];

    let mut h = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            // H_lagrangian + Penalty Term (w * dh_i * dh_k)
            h[i][k] = 2.0 * mu * (d[i].conj() * d[k] + s.conj() * ss[i][k]).re + w * dh[i] * dh[k];
        }
    }
    h
}
