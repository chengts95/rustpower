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
            
            // Gav[i,j] = Gva[j,i]. For i==j the transpose is itself, so reuse gva
            // (the off-diagonal formula below omits the diagonal d_lam/Ibus terms).
            let gav = if i == j {
                gva
            } else {
                (j_unit * inv_vmag[j] * ( (v_s[j].conj() * y_vals[idx].conj() * lam_v[i]) - c_ji )).re
            };
            
            let ptrs = cache.y_to_lxx[idx];
            // Assign values to Lxx slots (one-to-one mapping)
            lxx_vals[ptrs[0]] = gaa;
            lxx_vals[ptrs[1]] = gav;
            lxx_vals[ptrs[2]] = gva;
            lxx_vals[ptrs[3]] = gvv;
        }
    }

    // --- 3. Pass 2: Branch Flow Limits (direct per-branch closed form) ---
    // For each branch, the from/to apparent-power-squared Hessian is a local 4x4
    // block in [theta_f, theta_t, vm_f, vm_t]. Derived from first principles:
    //   S_self = a*vm_self^2 + b*V_self*conj(V_other),  a=conj(Y_self), b=conj(Y_other)
    // No intermediate matrices: each entry is scattered straight into br_to_lxx slots.
    let mu_f = &mu_ineq[..nl];
    let mu_t = &mu_ineq[nl..];
    let yf_vals = data.yf.values();
    let yt_vals = data.yt.values();

    for l in 0..nl {
        let f = data.f_buses[l];
        let t = data.t_buses[l];
        let vf = v_s[f];
        let vt = v_s[t];

        // from-end: self=f, other=t → natural order [theta_f, theta_t, vm_f, vm_t]
        let hf = branch_end_hess(
            yf_vals[cache.br_to_yf_idx[l][0]].conj(), // a = conj(Yff)
            yf_vals[cache.br_to_yf_idx[l][1]].conj(), // b = conj(Yft)
            vf, vt, mu_f[l],
        );
        // to-end: self=t, other=f → order [theta_t, theta_f, vm_t, vm_f]
        let ht = branch_end_hess(
            yt_vals[cache.br_to_yt_idx[l][1]].conj(), // a = conj(Ytt)
            yt_vals[cache.br_to_yt_idx[l][0]].conj(), // b = conj(Ytf)
            vt, vf, mu_t[l],
        );

        // Accumulate into the quad ordered [theta_f, theta_t, vm_f, vm_t].
        let mut hq = [[0.0f64; 4]; 4];
        const P: [usize; 4] = [1, 0, 3, 2]; // to-end variable permutation → quad order
        for i in 0..4 {
            for k in 0..4 {
                hq[i][k] += hf[i][k];
                hq[P[i]][P[k]] += ht[i][k];
            }
        }

        // Scatter: node 0=f, 1=t; quad theta idx = node, vm idx = 2+node.
        let ptrs = &cache.br_to_lxx[l];
        for ni in 0..2 {
            for nj in 0..2 {
                let base = (ni * 2 + nj) * 4;
                lxx_vals[ptrs[base + 0]] += hq[ni][nj];           // aa
                lxx_vals[ptrs[base + 1]] += hq[ni][2 + nj];       // av
                lxx_vals[ptrs[base + 2]] += hq[2 + ni][nj];       // va
                lxx_vals[ptrs[base + 3]] += hq[2 + ni][2 + nj];   // vv
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

/// Per-branch-end Hessian of mu*|S_self|^2, a 4x4 real block in variable order
/// [theta_self, theta_other, vm_self, vm_other].
///
/// S_self = a*vm_self^2 + T,  T = b * V_self * conj(V_other),
/// with a = conj(Y_self-self), b = conj(Y_self-other).
/// H_xy = 2*mu*Re( conj(S_x)*S_y + conj(S)*S_xy ).
#[inline]
fn branch_end_hess(a: Complex64, b: Complex64, v_self: Complex64, v_other: Complex64, mu: f64) -> [[f64; 4]; 4] {
    let j = Complex64::i();
    let vms = v_self.norm();
    let vmo = v_other.norm();
    let t = b * v_self * v_other.conj();
    let s = a * (vms * vms) + t;

    // First derivatives of S_self w.r.t. [theta_self, theta_other, vm_self, vm_other].
    let d = [
        j * t,
        -j * t,
        2.0 * a * vms + t / vms,
        t / vmo,
    ];

    // Second derivatives S_xy (symmetric 4x4).
    let z = Complex64::new(0.0, 0.0);
    let mut ss = [[z; 4]; 4];
    ss[0][0] = -t;
    ss[0][1] = t;          ss[1][0] = t;
    ss[1][1] = -t;
    ss[0][2] = j * t / vms; ss[2][0] = ss[0][2];
    ss[0][3] = j * t / vmo; ss[3][0] = ss[0][3];
    ss[1][2] = -j * t / vms; ss[2][1] = ss[1][2];
    ss[1][3] = -j * t / vmo; ss[3][1] = ss[1][3];
    ss[2][2] = 2.0 * a;
    ss[2][3] = t / (vms * vmo); ss[3][2] = ss[2][3];
    // ss[3][3] = 0

    let mut h = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            h[i][k] = 2.0 * mu * (d[i].conj() * d[k] + s.conj() * ss[i][k]).re;
        }
    }
    h
}
