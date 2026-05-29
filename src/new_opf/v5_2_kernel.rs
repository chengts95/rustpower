//! V5.2: block-operator fused KKT kernel.
//!
//! The sparse block-cache works like a block-matrix operation: every KKT block is a
//! small operator with a *uniform* signature `fn(&ColCtx, &NbrCtx) -> f64`. For one Ybus
//! column we grab the contiguous output runs (Haa/Hva/dgP/dgQ …), sweep the neighbors
//! once, and stream each block operator's scalar into its run — no intermediate matrices
//! (no `lxx`, no `dS_dVa/dVm`), no scatter, single forward write per run.
//!
//! Node Hessian formulas come from the validated V4 rectangular kernel; Jacobian formulas
//! from the validated `dSbus_dV` columns (both reproduced byte-for-byte in V5.0/V5.1-b).
//!
//! Scope of this first cut: variable columns (θ, Vm, gen) of the KKT, natural order,
//! node power-balance only (branch limits / merged slacks added later). Constraint
//! columns (CSR) + permutation + PIPS wiring come next.

use num_complex::Complex64;
use crate::opf::problem::OPFData;
use super::v5_kkt::KKTSymbolicV5;

/// Per-Ybus-column context (column k = the variable's bus). Precomputed once per column.
#[derive(Clone, Copy)]
pub struct ColCtx {
    pub vk: Complex64,
    pub vnorm_k: Complex64,
    pub inv_vmag_k: f64,
    pub lam_v_k: Complex64, // (λP_k − jλQ_k)·V_k
    pub ibus_k: Complex64,  // (Ybus·V)_k   (diagonal terms)
    pub d_lam_k: Complex64, // (Ybusᴴ·lam_v)_k (diagonal Hessian term)
}

/// Per-neighbor context (row i within column k).
#[derive(Clone, Copy)]
pub struct NbrCtx {
    pub vi: Complex64,
    pub y_ik: Complex64,   // Ybus[i,k]
    pub y_ki: Complex64,   // Ybus[k,i] (transpose entry, for the Hessian)
    pub lam_v_i: Complex64,
    pub inv_vmag_i: f64,
    pub is_diag: bool,     // i == k
}

// ── shared subexpressions (kept local so each block op stays a pure (Col,Nbr)→f64) ──
#[inline(always)]
fn c_ik(col: &ColCtx, nbr: &NbrCtx) -> Complex64 {
    nbr.lam_v_i * (nbr.y_ik * col.vk).conj()
}
#[inline(always)]
fn c_ki(col: &ColCtx, nbr: &NbrCtx) -> Complex64 {
    col.lam_v_k * (nbr.y_ki * nbr.vi).conj()
}
#[inline(always)]
fn ef(col: &ColCtx, nbr: &NbrCtx) -> (Complex64, Complex64) {
    let cik = c_ik(col, nbr);
    if nbr.is_diag {
        let e = col.vk.conj() * (nbr.y_ik.conj() * col.lam_v_k - col.d_lam_k);
        let f = cik - col.lam_v_k * col.ibus_k.conj();
        (e, f)
    } else {
        let e = nbr.vi.conj() * nbr.y_ki.conj() * col.lam_v_k;
        (e, cik)
    }
}
#[inline(always)]
fn ds_dva(col: &ColCtx, nbr: &NbrCtx) -> Complex64 {
    let j = Complex64::i();
    if nbr.is_diag {
        j * col.vk * (col.ibus_k - nbr.y_ik * col.vk).conj()
    } else {
        j * nbr.vi * (-nbr.y_ik * col.vk).conj()
    }
}
#[inline(always)]
fn ds_dvm(col: &ColCtx, nbr: &NbrCtx) -> Complex64 {
    if nbr.is_diag {
        col.vk * (nbr.y_ik * col.vnorm_k).conj() + col.ibus_k.conj() * col.vnorm_k
    } else {
        nbr.vi * (nbr.y_ik * col.vnorm_k).conj()
    }
}

// ── the eight uniform block operators: (&ColCtx, &NbrCtx) -> f64 ──

/// θ-column block ∂²L/∂θ_i∂θ_k (Haa).
#[inline(always)]
pub fn haa(col: &ColCtx, nbr: &NbrCtx) -> f64 {
    let (e, f) = ef(col, nbr);
    (e + f).re
}
/// θ-column block ∂²L/∂Vm_i∂θ_k (Hva).
#[inline(always)]
pub fn hva(col: &ColCtx, nbr: &NbrCtx) -> f64 {
    let (e, f) = ef(col, nbr);
    (Complex64::i() * nbr.inv_vmag_i * (e - f)).re
}
/// Vm-column block ∂²L/∂θ_i∂Vm_k (Hav).
#[inline(always)]
pub fn hav(col: &ColCtx, nbr: &NbrCtx) -> f64 {
    if nbr.is_diag {
        return hva(col, nbr);
    }
    let cki = c_ki(col, nbr);
    (Complex64::i() * col.inv_vmag_k * (col.vk.conj() * nbr.y_ik.conj() * nbr.lam_v_i - cki)).re
}
/// Vm-column block ∂²L/∂Vm_i∂Vm_k (Hvv).
#[inline(always)]
pub fn hvv(col: &ColCtx, nbr: &NbrCtx) -> f64 {
    (nbr.inv_vmag_i * (c_ik(col, nbr) + c_ki(col, nbr)) * col.inv_vmag_k).re
}
/// θ-column coupling ∂P_i/∂θ_k (dgᵀ, P row).
#[inline(always)]
pub fn dp_dth(col: &ColCtx, nbr: &NbrCtx) -> f64 { ds_dva(col, nbr).re }
/// θ-column coupling ∂Q_i/∂θ_k (dgᵀ, Q row).
#[inline(always)]
pub fn dq_dth(col: &ColCtx, nbr: &NbrCtx) -> f64 { ds_dva(col, nbr).im }
/// Vm-column coupling ∂P_i/∂Vm_k.
#[inline(always)]
pub fn dp_dvm(col: &ColCtx, nbr: &NbrCtx) -> f64 { ds_dvm(col, nbr).re }
/// Vm-column coupling ∂Q_i/∂Vm_k.
#[inline(always)]
pub fn dq_dvm(col: &ColCtx, nbr: &NbrCtx) -> f64 { ds_dvm(col, nbr).im }

/// Fill the KKT **variable columns** (θ, Vm, Pg, Qg) of `kkt_vals` in place, streaming.
/// `y_trans[idx]` maps Ybus nnz `idx`=(i,k) to its transpose nnz (k,i).
/// Node power-balance only (no branch / merged-slack here).
pub fn fill_variable_columns(
    v5: &KKTSymbolicV5,
    data: &OPFData,
    y_trans: &[usize],
    x: &[f64],
    lam_eq: &[f64],
    cost_mult: f64,
    kkt_vals: &mut [f64],
) {
    let nb = data.nb;
    let ng = data.ng;
    let ybus = &data.ybus;
    let y_cp = ybus.col_offsets();
    let y_ri = ybus.row_indices();
    let y_v = ybus.values();

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
    for i in 0..nb {
        lam_v[i] = Complex64::new(lp[i], -lq[i]) * vs[i];
    }
    let ibus = ybus * &v;
    let ibus_s = ibus.as_slice();
    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        for idx in y_cp[i]..y_cp[i + 1] {
            d_lam[i] += y_v[idx].conj() * lam_v[y_ri[idx]];
        }
    }

    let mut is_fixed = vec![false; v5.nx];
    for &vix in &v5.ieq { is_fixed[vix] = true; }

    let cp = &v5.col_ptrs;

    let col_ctx = |k: usize| ColCtx {
        vk: vs[k], vnorm_k: vnorm[k], inv_vmag_k: inv_vmag[k],
        lam_v_k: lam_v[k], ibus_k: ibus_s[k], d_lam_k: d_lam[k],
    };

    // θ columns and Vm columns (both driven by Ybus column k)
    for k in 0..nb {
        let deg = y_cp[k + 1] - y_cp[k];
        let col = col_ctx(k);

        let th0 = cp[k];          // θ_k column base: [Haa | Hva | dgP | dgQ]
        let vm0 = cp[nb + k];     // Vm_k column base: [Hav | Hvv | dgP | dgQ]
        for off in 0..deg {
            let idx = y_cp[k] + off;
            let i = y_ri[idx];
            let nbr = NbrCtx {
                vi: vs[i], y_ik: y_v[idx], y_ki: y_v[y_trans[idx]],
                lam_v_i: lam_v[i], inv_vmag_i: inv_vmag[i], is_diag: i == k,
            };
            // θ_k column
            kkt_vals[th0 + off]           = haa(&col, &nbr);
            kkt_vals[th0 + deg + off]     = hva(&col, &nbr);
            kkt_vals[th0 + 2 * deg + off] = dp_dth(&col, &nbr);
            kkt_vals[th0 + 3 * deg + off] = dq_dth(&col, &nbr);
            // Vm_k column
            kkt_vals[vm0 + off]           = hav(&col, &nbr);
            kkt_vals[vm0 + deg + off]     = hvv(&col, &nbr);
            kkt_vals[vm0 + 2 * deg + off] = dp_dvm(&col, &nbr);
            kkt_vals[vm0 + 3 * deg + off] = dq_dvm(&col, &nbr);
        }
        if is_fixed[k]      { kkt_vals[th0 + 4 * deg] = 1.0; }
        if is_fixed[nb + k] { kkt_vals[vm0 + 4 * deg] = 1.0; }
    }

    // generator columns: [diag, −1 coupling, optional lineq]
    let base_mva = data.base_mva;
    for g in 0..ng {
        let pg = 2 * nb + g;
        let qg = 2 * nb + ng + g;
        let pg0 = cp[pg];
        kkt_vals[pg0]     = cost_mult * 2.0 * data.cost_coeffs[g][0] * base_mva * base_mva; // cost diag
        kkt_vals[pg0 + 1] = -1.0; // ∂P_eq/∂Pg
        if is_fixed[pg] { kkt_vals[pg0 + 2] = 1.0; }
        let qg0 = cp[qg];
        kkt_vals[qg0]     = 0.0;  // structural Qg diagonal
        kkt_vals[qg0 + 1] = -1.0; // ∂Q_eq/∂Qg
        if is_fixed[qg] { kkt_vals[qg0 + 2] = 1.0; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opf::builder::opf_data_from_network;
    use crate::io::pandapower::load_csv_zip;

    /// V5.2 block-operator variable-column fill must match V5.0 on the variable-column
    /// portion of the KKT (node power-balance only ⇒ mu=0 so no branch Hessian).
    #[test]
    fn test_v5_2_variable_columns_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net = load_csv_zip(&format!("{}/cases/IEEE118/data.zip", dir)).unwrap();
        let data = opf_data_from_network(&net);
        let nb = data.nb;
        let nx = data.nx();

        let v5 = KKTSymbolicV5::build(&data);
        let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);

        let x = data.warm_x0();
        let lam = vec![0.1; 2 * nb];
        let mu = vec![0.0; 2 * data.nl]; // no branch Hessian
        let cm = 1e-4;

        // Reference: V5.0 fill (lxx with mu=0, no z) → variable-column portion
        let lxx = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
            &data, &v3c, x.as_slice(), &lam, &mu, None, cm,
        );
        let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
        let dg_t = dg.transpose();
        let mut ref_vals = vec![0.0f64; v5.row_idx.len()];
        v5.fill(
            lxx.col_offsets(), lxx.values(),
            dg.col_offsets(), dg.values(),
            dg_t.col_offsets(), dg_t.values(),
            &mut ref_vals,
        );

        // V5.2 block-operator fill of variable columns
        let mut v52_vals = vec![0.0f64; v5.row_idx.len()];
        fill_variable_columns(&v5, &data, &v3c.y_transpose_idx, x.as_slice(), &lam, cm, &mut v52_vals);

        // Compare only the variable-column region [0, col_ptrs[nx])
        let end = v5.col_ptrs[nx];
        let mut max_diff = 0.0f64;
        let mut worst = 0usize;
        for p in 0..end {
            let d = (v52_vals[p] - ref_vals[p]).abs();
            if d > max_diff { max_diff = d; worst = p; }
        }
        println!(
            "V5.2 variable-column fill vs V5.0: compared {} vals, max_diff={:.3e} at pos {}",
            end, max_diff, worst
        );
        assert!(max_diff < 1e-12, "V5.2 variable columns differ (max_diff={:.3e})", max_diff);
    }
}
