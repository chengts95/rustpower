//! V5.3: Partitioned Isomorphic KKT Assembly.
//!
//! This module implements a top-down, partitioned assembly architecture for the KKT matrix.
//! It leverages the topological isomorphism between the network structure (Ybus/Ybr) 
//! and the KKT Hessian to achieve zero-scatter, cache-local assembly.
//!
//! Branch limit Hessians are pre-projected into a contiguous array shaped exactly like Ybus.
//! The main assembly loop then performs a purely sequential, single-pass streaming read/write.

use crate::opf::problem::OPFData;
use crate::new_opf::v5_kkt::KKTSymbolicV5;
use num_complex::Complex64;

/// Column-centric symbolic mapping for V5.3.
pub struct KKTSymbolicV5_3 {
    pub base: KKTSymbolicV5,
    /// For each branch l, stores the exact indices into Ybus values for the 4 corners:
    /// [ (f,f), (f,t), (t,f), (t,t) ] where pairs are (col, row).
    pub br_to_ybus_idx: Vec<[usize; 4]>,
}

impl KKTSymbolicV5_3 {
    pub fn build(data: &OPFData) -> Self {
        let base = KKTSymbolicV5::build(data);
        let nl = data.nl;
        let ybus = &data.ybus;
        let y_cp = ybus.col_offsets();
        let y_ri = ybus.row_indices();
        
        let mut br_to_ybus_idx = vec![[0usize; 4]; nl];
        
        let find_idx = |col: usize, row: usize| -> usize {
            let range = (y_cp[col] as usize)..(y_cp[col + 1] as usize);
            y_ri[range.clone()].iter().position(|&r| r as usize == row).unwrap() + range.start
        };

        for l in 0..nl {
            let f = data.f_buses[l];
            let t = data.t_buses[l];
            br_to_ybus_idx[l] = [
                find_idx(f, f), // col f, row f
                find_idx(f, t), // col f, row t
                find_idx(t, f), // col t, row f
                find_idx(t, t), // col t, row t
            ];
        }
        
        Self { base, br_to_ybus_idx }
    }
}

pub fn assemble_kkt_v5_3(
    v53: &KKTSymbolicV5_3,
    data: &OPFData,
    y_trans: &[usize],
    x: &[f64],
    lam_eq: &[f64],
    mu_ineq: &[f64],
    z_ineq: &[f64],
    cost_mult: f64,
    kkt_vals: &mut [f64],
) {
    let nb = data.nb;
    let nx = v53.base.nx;
    let nl = data.nl;
    let ybus = &data.ybus;

    // 1. Prepare global state (vector maps)
    let v = data.v_from_x(x);
    let vs = v.as_slice();
    let mut inv_vmag = vec![0.0f64; nb];
    let mut vnorm = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        let m = vs[i].norm().max(1e-9);
        inv_vmag[i] = 1.0 / m;
        vnorm[i] = vs[i] / m;
    }
    let lp = &lam_eq[..nb];
    let lq = &lam_eq[nb..2 * nb];
    let mut lam_v = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb { lam_v[i] = Complex64::new(lp[i], -lq[i]) * vs[i]; }
    let ibus = ybus * &v;
    let ibus_s = ibus.as_slice();
    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        for idx in (ybus.col_offsets()[i] as usize)..(ybus.col_offsets()[i+1] as usize) {
            d_lam[i] += ybus.values()[idx].conj() * lam_v[ybus.row_indices()[idx] as usize];
        }
    }
    let mut is_fixed = vec![false; nx];
    for &vix in &v53.base.ieq { is_fixed[vix] = true; }

    // 2. Pre-Project Branch Hessians into Ybus Operator Space
    // We allocate a buffer matching Ybus nnz exactly.
    let mut ybus_br_hess = vec![[0.0f64; 4]; ybus.values().len()];
    pre_project_branch_hessians(data, vs, mu_ineq, z_ineq, nl, &v53.br_to_ybus_idx, &mut ybus_br_hess);

    // 3. Clear KKT
    kkt_vals.fill(0.0);

    // 4. Fill Columns via strictly separated functions
    fill_theta_columns(nb, v53, ybus, vs, &vnorm, &inv_vmag, &lam_v, ibus_s, &d_lam, y_trans, &ybus_br_hess, &is_fixed, kkt_vals);
    fill_vm_columns(nb, v53, ybus, vs, &vnorm, &inv_vmag, &lam_v, ibus_s, &d_lam, y_trans, &ybus_br_hess, &is_fixed, kkt_vals);
    fill_generator_columns(data, nx, cost_mult, &is_fixed, &v53.base.col_ptrs, kkt_vals);

    // 5. Constraint & Linear Eq Columns (Reuse V5.2 sequential fill)
    let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for g in 0..data.ng { gens_at_bus[data.gen_bus[g]].push(g); }
    let mut v52_temp = vec![0.0; v53.base.row_idx.len()];
    super::v5_2_kernel::fill_constraint_columns(&v53.base, data, y_trans, &gens_at_bus, x, &mut v52_temp);
    let c_start = v53.base.col_ptrs[nx];
    kkt_vals[c_start..].copy_from_slice(&v52_temp[c_start..]);
}

#[inline(always)]
fn pre_project_branch_hessians(
    data: &OPFData,
    vs: &[Complex64],
    mu_ineq: &[f64],
    z_ineq: &[f64],
    nl: usize,
    br_to_ybus_idx: &[[usize; 4]],
    ybus_br_hess: &mut [[f64; 4]],
) {
    use super::v4_numeric_rect::branch_end_hess_v4;
    let find_br_entry = |mat: &nalgebra_sparse::CscMatrix<Complex64>, c: usize, r: usize| -> usize {
        let range = (mat.col_offsets()[c] as usize)..(mat.col_offsets()[c + 1] as usize);
        mat.row_indices()[range.clone()].iter().position(|&val| val as usize == r).unwrap() + range.start
    };

    for l in 0..nl {
        let f = data.f_buses[l];
        let t = data.t_buses[l];
        let mu_f = mu_ineq[l];
        let wf = mu_f / z_ineq[l];
        let mu_t = mu_ineq[nl + l];
        let wt = mu_t / z_ineq[nl + l];

        let hf_all = branch_end_hess_v4(
            data.yf.values()[find_br_entry(&data.yf, f, l)].conj(),
            data.yf.values()[find_br_entry(&data.yf, t, l)].conj(),
            vs[f], vs[t], mu_f, wf,
        );
        let ht_all = branch_end_hess_v4(
            data.yt.values()[find_br_entry(&data.yt, t, l)].conj(),
            data.yt.values()[find_br_entry(&data.yt, f, l)].conj(),
            vs[t], vs[f], mu_t, wt,
        );

        let mut hq = [[0.0f64; 4]; 4];
        const P: [usize; 4] = [1, 0, 3, 2];
        for r in 0..4 {
            for c in 0..4 { hq[r][c] = hf_all[r][c] + ht_all[P[r]][P[c]]; }
        }

        let idxs = &br_to_ybus_idx[l];
        ybus_br_hess[idxs[0]][0] += hq[0][0]; ybus_br_hess[idxs[0]][1] += hq[2][0]; ybus_br_hess[idxs[0]][2] += hq[0][2]; ybus_br_hess[idxs[0]][3] += hq[2][2];
        ybus_br_hess[idxs[1]][0] += hq[1][0]; ybus_br_hess[idxs[1]][1] += hq[3][0]; ybus_br_hess[idxs[1]][2] += hq[1][2]; ybus_br_hess[idxs[1]][3] += hq[3][2];
        ybus_br_hess[idxs[2]][0] += hq[0][1]; ybus_br_hess[idxs[2]][1] += hq[2][1]; ybus_br_hess[idxs[2]][2] += hq[0][3]; ybus_br_hess[idxs[2]][3] += hq[2][3];
        ybus_br_hess[idxs[3]][0] += hq[1][1]; ybus_br_hess[idxs[3]][1] += hq[3][1]; ybus_br_hess[idxs[3]][2] += hq[1][3]; ybus_br_hess[idxs[3]][3] += hq[3][3];
    }
}

#[inline(always)]
fn fill_theta_columns(
    nb: usize, v53: &KKTSymbolicV5_3, ybus: &nalgebra_sparse::CscMatrix<Complex64>,
    vs: &[Complex64], vnorm: &[Complex64], inv_vmag: &[f64], lam_v: &[Complex64],
    ibus_s: &[Complex64], d_lam: &[Complex64], y_trans: &[usize], ybus_br_hess: &[[f64; 4]],
    is_fixed: &[bool], kkt_vals: &mut [f64]
) {
    use super::v5_2_kernel::*;
    let y_v = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    for j in 0..nb {
        let start = v53.base.col_ptrs[j];
        let col_slice = &mut kkt_vals[start..];
        let deg = (y_cp[j + 1] - y_cp[j]) as usize;
        
        let col_ctx = ColCtx {
            vk: vs[j], vnorm_k: vnorm[j], inv_vmag_k: inv_vmag[j],
            lam_v_k: lam_v[j], ibus_k: ibus_s[j], d_lam_k: d_lam[j],
        };
        
        for off in 0..deg {
            let idx = y_cp[j] as usize + off;
            let i = y_ri[idx] as usize;
            let nbr = NbrCtx {
                vi: vs[i], y_ik: y_v[idx], y_ki: y_v[y_trans[idx]],
                lam_v_i: lam_v[i], inv_vmag_i: inv_vmag[i], is_diag: i == j,
            };
            
            let br_h = &ybus_br_hess[idx];
            col_slice[off]       = haa(&col_ctx, &nbr) + br_h[0];
            col_slice[deg + off] = hva(&col_ctx, &nbr) + br_h[1];
            col_slice[2*deg + off] = dp_dth(&col_ctx, &nbr);
            col_slice[3*deg + off] = dq_dth(&col_ctx, &nbr);
        }
        if is_fixed[j] { col_slice[4 * deg] = 1.0; }
    }
}

#[inline(always)]
fn fill_vm_columns(
    nb: usize, v53: &KKTSymbolicV5_3, ybus: &nalgebra_sparse::CscMatrix<Complex64>,
    vs: &[Complex64], vnorm: &[Complex64], inv_vmag: &[f64], lam_v: &[Complex64],
    ibus_s: &[Complex64], d_lam: &[Complex64], y_trans: &[usize], ybus_br_hess: &[[f64; 4]],
    is_fixed: &[bool], kkt_vals: &mut [f64]
) {
    use super::v5_2_kernel::*;
    let y_v = ybus.values();
    let y_ri = ybus.row_indices();
    let y_cp = ybus.col_offsets();

    for j in 0..nb {
        let col = nb + j;
        let start = v53.base.col_ptrs[col];
        let col_slice = &mut kkt_vals[start..];
        let deg = (y_cp[j + 1] - y_cp[j]) as usize; // Vm_j has same degree as theta_j
        
        let col_ctx = ColCtx {
            vk: vs[j], vnorm_k: vnorm[j], inv_vmag_k: inv_vmag[j],
            lam_v_k: lam_v[j], ibus_k: ibus_s[j], d_lam_k: d_lam[j],
        };
        
        for off in 0..deg {
            let idx = y_cp[j] as usize + off;
            let i = y_ri[idx] as usize;
            let nbr = NbrCtx {
                vi: vs[i], y_ik: y_v[idx], y_ki: y_v[y_trans[idx]],
                lam_v_i: lam_v[i], inv_vmag_i: inv_vmag[i], is_diag: i == j,
            };
            
            let br_h = &ybus_br_hess[idx];
            col_slice[off]       = hav(&col_ctx, &nbr) + br_h[2];
            col_slice[deg + off] = hvv(&col_ctx, &nbr) + br_h[3];
            col_slice[2*deg + off] = dp_dvm(&col_ctx, &nbr);
            col_slice[3*deg + off] = dq_dvm(&col_ctx, &nbr);
        }
        if is_fixed[col] { col_slice[4 * deg] = 1.0; }
    }
}

#[inline(always)]
fn fill_generator_columns(
    data: &OPFData, nx: usize, cost_mult: f64, is_fixed: &[bool], col_ptrs: &[usize], kkt_vals: &mut [f64]
) {
    let nb = data.nb;
    for j in (2 * nb)..nx {
        let start = col_ptrs[j];
        let col_slice = &mut kkt_vals[start..];
        
        let g = (j - 2 * nb) % data.ng;
        let is_qg = (j - 2 * nb) >= data.ng;
        if !is_qg {
            col_slice[0] = cost_mult * 2.0 * data.cost_coeffs[g][0] * data.base_mva * data.base_mva;
            col_slice[1] = -1.0;
        } else {
            col_slice[0] = 0.0;
            col_slice[1] = -1.0;
        }
        if is_fixed[j] { col_slice[2] = 1.0; }
    }
}
