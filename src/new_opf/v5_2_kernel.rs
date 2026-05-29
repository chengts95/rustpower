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

// ── constraint-column block operators (CSR row access via y_trans) ──
// For constraint bus i, swept over its variable neighbors k. Uses Y_ik = Ybus[i,k],
// fetched from Ybus column i via the transpose index (no materialized Ybusᵀ).

/// Constraint-column context (the constraint's bus i).
#[derive(Clone, Copy)]
pub struct ConColCtx {
    pub vi: Complex64,
    pub vnorm_i: Complex64,
    pub ibus_i: Complex64,
}
/// Constraint-column neighbor context (the variable's bus k).
#[derive(Clone, Copy)]
pub struct ConNbrCtx {
    pub vk: Complex64,
    pub vnorm_k: Complex64,
    pub y_ik: Complex64, // Ybus[i,k]
    pub is_diag: bool,
}

#[inline(always)]
fn con_ds_dva(c: &ConColCtx, n: &ConNbrCtx) -> Complex64 {
    let j = Complex64::i();
    if n.is_diag {
        j * c.vi * (c.ibus_i - n.y_ik * c.vi).conj()
    } else {
        j * c.vi * (-n.y_ik * n.vk).conj()
    }
}
#[inline(always)]
fn con_ds_dvm(c: &ConColCtx, n: &ConNbrCtx) -> Complex64 {
    if n.is_diag {
        c.vi * (n.y_ik * c.vnorm_i).conj() + c.ibus_i.conj() * c.vnorm_i
    } else {
        c.vi * (n.y_ik * n.vnorm_k).conj()
    }
}
/// ∂P_i/∂θ_k, ∂P_i/∂Vm_k, ∂Q_i/∂θ_k, ∂Q_i/∂Vm_k — uniform (&ConColCtx,&ConNbrCtx)->f64.
#[inline(always)] pub fn cp_dth(c: &ConColCtx, n: &ConNbrCtx) -> f64 { con_ds_dva(c, n).re }
#[inline(always)] pub fn cp_dvm(c: &ConColCtx, n: &ConNbrCtx) -> f64 { con_ds_dvm(c, n).re }
#[inline(always)] pub fn cq_dth(c: &ConColCtx, n: &ConNbrCtx) -> f64 { con_ds_dva(c, n).im }
#[inline(always)] pub fn cq_dvm(c: &ConColCtx, n: &ConNbrCtx) -> f64 { con_ds_dvm(c, n).im }

/// Fill the KKT **constraint columns** (P_eq, Q_eq) of `kkt_vals` in place, streaming.
/// Row access into Ybus is done through `y_trans` (transpose index): the offset-th
/// neighbor of constraint bus i is `k = y_ri[y_cp[i]+offset]` and `Y_ik` is
/// `y_vals[y_trans[y_cp[i]+offset]]` — CSR-via-index, no materialized transpose.
/// `gens_at_bus[i]` lists generators on bus i (ascending) for the gen coupling run.
pub fn fill_constraint_columns(
    v5: &KKTSymbolicV5,
    data: &OPFData,
    y_trans: &[usize],
    gens_at_bus: &[Vec<usize>],
    x: &[f64],
    kkt_vals: &mut [f64],
) {
    let nb = data.nb;
    let ng = data.ng;
    let nx = v5.nx;
    let ybus = &data.ybus;
    let y_cp = ybus.col_offsets();
    let y_ri = ybus.row_indices();
    let y_v = ybus.values();

    let v = data.v_from_x(x);
    let vs = v.as_slice();
    let mut vnorm = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        vnorm[i] = vs[i] / vs[i].norm().max(1e-9);
    }
    let ibus = ybus * &v;
    let ibus_s = ibus.as_slice();
    let cp = &v5.col_ptrs;

    for i in 0..nb {
        let deg = y_cp[i + 1] - y_cp[i];
        let con = ConColCtx { vi: vs[i], vnorm_i: vnorm[i], ibus_i: ibus_s[i] };

        let p0 = cp[nx + i];        // P_eq_i column: [∂P/∂θ_k | ∂P/∂Vm_k | gen]
        let q0 = cp[nx + nb + i];   // Q_eq_i column: [∂Q/∂θ_k | ∂Q/∂Vm_k | gen]
        for off in 0..deg {
            let pos = y_cp[i] + off;
            let k = y_ri[pos];
            let nbr = ConNbrCtx {
                vk: vs[k], vnorm_k: vnorm[k],
                y_ik: y_v[y_trans[pos]], // Ybus[i,k] via transpose index
                is_diag: k == i,
            };
            kkt_vals[p0 + off]         = cp_dth(&con, &nbr);
            kkt_vals[p0 + deg + off]   = cp_dvm(&con, &nbr);
            kkt_vals[q0 + off]         = cq_dth(&con, &nbr);
            kkt_vals[q0 + deg + off]   = cq_dvm(&con, &nbr);
        }
        // generator coupling run: −1 per gen on bus i
        let pg_run = p0 + 2 * deg;
        let qg_run = q0 + 2 * deg;
        for (gi, _g) in gens_at_bus[i].iter().enumerate() {
            kkt_vals[pg_run + gi] = -1.0;
            kkt_vals[qg_run + gi] = -1.0;
        }
    }
    // linear-equality columns (nx + 2nb + r): single unit entry (the ae row)
    for r in 0..v5.ieq.len() {
        kkt_vals[cp[nx + 2 * nb + r]] = 1.0;
    }
    let _ = ng;
}

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

        // V5.2 block-operator fill: variable columns + constraint columns (full KKT)
        let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for g in 0..data.ng { gens_at_bus[data.gen_bus[g]].push(g); }

        let mut v52_vals = vec![0.0f64; v5.row_idx.len()];
        fill_variable_columns(&v5, &data, &v3c.y_transpose_idx, x.as_slice(), &lam, cm, &mut v52_vals);
        fill_constraint_columns(&v5, &data, &v3c.y_transpose_idx, &gens_at_bus, x.as_slice(), &mut v52_vals);

        // Compare the FULL KKT (all columns) vs V5.0
        let mut max_diff = 0.0f64;
        let mut worst = 0usize;
        for p in 0..v5.row_idx.len() {
            let d = (v52_vals[p] - ref_vals[p]).abs();
            if d > max_diff { max_diff = d; worst = p; }
        }
        let nvar = v5.col_ptrs[nx];
        println!(
            "V5.2 FULL KKT fill vs V5.0: compared {} vals, max_diff={:.3e} at pos {} ({})",
            v5.row_idx.len(), max_diff, worst,
            if worst < nvar { "variable col" } else { "constraint col" }
        );
        assert!(max_diff < 1e-12, "V5.2 full KKT differs (max_diff={:.3e})", max_diff);
    }

    /// Per-iteration KKT-matrix production speed: V4 (lxx+dg+build_saddle_point) vs
    /// V5.0 (lxx+dg+transpose+fill) vs V5.2 (block operators, fully inline — no lxx, no
    /// dg matrix, no transpose). Node power-balance KKT (mu=0).
    /// cargo test --release bench_v5_2_kkt_prep -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_v5_2_kkt_prep() {
        use crate::opf::pips::build_saddle_point;
        use nalgebra_sparse::{CooMatrix, CscMatrix};

        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        for case in ["IEEE118", "pegase9241"] {
            let path = format!("{}/cases/{}/data.zip", dir, case);
            if !std::path::Path::new(&path).exists() { continue; }
            let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
            let data = opf_data_from_network(&net);
            let nb = data.nb; let nx = data.nx();
            let v5 = KKTSymbolicV5::build(&data);
            let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);
            let x = data.warm_x0();
            let lam = vec![0.1; 2 * nb];
            let mu = vec![0.0; 2 * data.nl];
            let cm = 1e-4;
            let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
            for g in 0..data.ng { gens_at_bus[data.gen_bus[g]].push(g); }
            let neqlin = v5.ieq.len();
            let iters = if case == "IEEE118" { 300 } else { 30 };

            // build merged dg once (structure invariant); included per-iter for V4/V5.0 fairness via opf_consfcn
            let mut kkt = vec![0.0f64; v5.row_idx.len()];

            // V4: v4 node fill + opf_consfcn dg + build_saddle_point
            let t = std::time::Instant::now();
            let mut sink = 0.0;
            for _ in 0..iters {
                let lxx = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(&data, &v3c, x.as_slice(), &lam, &mu, None, cm);
                let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
                let mut coo = CooMatrix::<f64>::new(nx, 2*nb+neqlin);
                for j in 0..dg.ncols() { for idx in dg.col_offsets()[j]..dg.col_offsets()[j+1] { coo.push(dg.row_indices()[idx], j, dg.values()[idx]); } }
                for (r,&vv) in v5.ieq.iter().enumerate() { coo.push(vv, 2*nb+r, 1.0); }
                let dgf = CscMatrix::from(&coo);
                let k = build_saddle_point(&lxx, &Some(dgf), nx, v5.neq);
                sink += k.values()[0];
            }
            let d_v4 = t.elapsed() / iters;

            // V5.0: v4 node fill + opf_consfcn dg + transpose + fill
            let t = std::time::Instant::now();
            for _ in 0..iters {
                let lxx = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(&data, &v3c, x.as_slice(), &lam, &mu, None, cm);
                let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
                let mut coo = CooMatrix::<f64>::new(nx, 2*nb+neqlin);
                for j in 0..dg.ncols() { for idx in dg.col_offsets()[j]..dg.col_offsets()[j+1] { coo.push(dg.row_indices()[idx], j, dg.values()[idx]); } }
                for (r,&vv) in v5.ieq.iter().enumerate() { coo.push(vv, 2*nb+r, 1.0); }
                let dgf = CscMatrix::from(&coo);
                let dgt = dgf.transpose();
                v5.fill_from_merged(lxx.col_offsets(), lxx.values(), dgf.col_offsets(), dgf.values(), dgt.col_offsets(), dgt.values(), &mut kkt);
                sink += kkt[0];
            }
            let d_v50 = t.elapsed() / iters;

            // V5.2: block operators, fully inline
            let t = std::time::Instant::now();
            for _ in 0..iters {
                fill_variable_columns(&v5, &data, &v3c.y_transpose_idx, x.as_slice(), &lam, cm, &mut kkt);
                fill_constraint_columns(&v5, &data, &v3c.y_transpose_idx, &gens_at_bus, x.as_slice(), &mut kkt);
                sink += kkt[0];
            }
            let d_v52 = t.elapsed() / iters;

            println!(
                "[{}] KKT-prep/iter — V4: {:?} | V5.0: {:?} ({:.1}x) | V5.2 inline: {:?} ({:.1}x)  (sink={:.2e})",
                case, d_v4,
                d_v50, d_v4.as_secs_f64()/d_v50.as_secs_f64(),
                d_v52, d_v4.as_secs_f64()/d_v52.as_secs_f64(),
                sink
            );
        }
    }
}
