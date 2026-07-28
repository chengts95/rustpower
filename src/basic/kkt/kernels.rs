//! Phase 1 — numeric fill kernels for the LM augmented system (doc §4).
//!
//! All three kernels honor the same discipline as `fill_jacobian_v3`:
//! region-split sweeps with no `i == j` branch in the hot loop, raw-pointer
//! writes, and positions derived by start-plus-offset arithmetic only.
//!
//! * [`fill_h`] — polar H(r) quadrants (§1.4): one off-diagonal sweep over
//!   the Ybus edges, one per-bus diagonal sweep.
//! * [`fill_jt`] — Jᵀ as a pure transpose-copy of the J block; every target
//!   position is derived from `y_trans` and the segment cuts (§4.3).
//! * [`apply_mu_delta`] — LM inner-loop μ update; touches only the `aa`/`vv`
//!   diagonal slots (§4.4).

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::pattern::KktPattern;

/// Off-diagonal edge `(row i, col k)` of the H(r) quadrants.
///
/// `FULL`  — the row is PQ: also write the `va` (and `vv`) quadrant.
/// `VCOL`  — the column bus is PQ: also write the `av`/`vv` (|V| column).
///
/// Formulas are MATPOWER TN2 evaluated with rectangular-complex arithmetic
/// (v4 Node Power Balance, `d2sbus_dv2.rs`): with `f_ij = c_ij` off-diagonal,
///   gaa = Re{e + f}        gva = Re{ j·(e − f)/|V_i| }
///   gvv = Re{(c_ik + c_ki)/(|V_i||V_k|)}   gav = Re{ j·(conj(V_k)·conj(Y_ik)·λV_i − c_ki)/|V_k| }
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn h_offdiag_edge<const FULL: bool, const VCOL: bool>(
    y_cp: &[usize],
    y_ri: &[usize],
    y_v: &[Complex64],
    y_trans: &[usize],
    col_starts: &[usize],
    v: &[Complex64],
    lam_v: &[Complex64],
    inv_vmag: &[f64],
    n_act: usize,
    active_end: usize,
    k: usize,
    t: usize,
    h: *mut f64,
) {
    let j_unit = Complex64::new(0.0, 1.0);
    let p = y_cp[k] + t;
    let i = y_ri[p];
    let y_ik = y_v[p];
    let y_ki = y_v[y_trans[p]];

    let c_ik = lam_v[i] * (y_ik * v[k]).conj();
    let e_ik = v[i].conj() * y_ki.conj() * lam_v[k];
    let c_ki = lam_v[k] * (y_ki * v[i]).conj();

    let gaa = (e_ik + c_ik).re;
    unsafe {
        *h.add(col_starts[k] + t) = gaa;
        if FULL {
            let gva = (j_unit * inv_vmag[i] * (e_ik - c_ik)).re;
            *h.add(col_starts[k] + active_end + t) = gva;
        }
        if VCOL {
            let vc = col_starts[n_act + k];
            let gav = (j_unit * inv_vmag[k] * (v[k].conj() * y_ik.conj() * lam_v[i] - c_ki)).re;
            *h.add(vc + t) = gav;
            if FULL {
                let gvv = inv_vmag[i] * inv_vmag[k] * (c_ik + c_ki).re;
                *h.add(vc + active_end + t) = gvv;
            }
        }
    }
}

/// Fill the H(r) block: `H(r) = Σ_k r_k·∇²F_k` with polar quadrants.
///
/// `v` covers **all** buses (slack included — its physics enters through
/// `ibus`); `r` is the reduced residual `[P (n_act); Q (n_pq)]`. The
/// multipliers are `λ_k = rP_k − i·rQ_k` (PQ), `rP_k` (PV), `0` (slack).
pub fn fill_h(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    v: &[Complex64],
    r: &[f64],
    h_vals: &mut [f64],
) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let npq = cache.n_pq();
    let nb = ybus.ncols();
    let (y_cp, y_ri, y_v) = (ybus.col_offsets(), ybus.row_indices(), ybus.values());
    let (pq_ends, active_ends, diag_off) =
        (cache.pq_ends(), cache.active_ends(), cache.diag_off());
    let col_starts = &pat.graph.col_starts[..];
    let y_trans = cache.y_trans();
    debug_assert_eq!(h_vals.len(), pat.graph.nnz);

    // ── Per-bus precomputes ───────────────────────────────────────────
    let mut lam_v = vec![Complex64::new(0.0, 0.0); nb];
    for k in 0..n_act {
        let lam = if k < npq {
            Complex64::new(r[k], -r[n_act + k])
        } else {
            Complex64::new(r[k], 0.0)
        };
        lam_v[k] = lam * v[k];
    }
    let mut ibus = vec![Complex64::new(0.0, 0.0); nb];
    for j in 0..nb {
        for p in y_cp[j]..y_cp[j + 1] {
            ibus[y_ri[p]] += y_v[p] * v[j];
        }
    }
    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    for i in 0..nb {
        for p in y_cp[i]..y_cp[i + 1] {
            d_lam[i] += y_v[p].conj() * lam_v[y_ri[p]];
        }
    }
    let inv_vmag: Vec<f64> = (0..nb).map(|i| 1.0 / v[i].norm().max(1e-12)).collect();

    let h = h_vals.as_mut_ptr();

    // ── Off-diagonal pass: every coupling entry, diagonal skipped ─────
    // Regions split like fill_jacobian_v3: the diagonal of a PQ column sits
    // in the PQ region, of a PV column in the PV region; both boundaries are
    // cache constants, so each region is one branch-free sweep.
    for k in 0..npq {
        let d = diag_off[k];
        for t in 0..d {
            h_offdiag_edge::<true, true>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
        for t in d + 1..pq_ends[k] {
            h_offdiag_edge::<true, true>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
        for t in pq_ends[k]..active_ends[k] {
            h_offdiag_edge::<false, true>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
    }
    for k in npq..n_act {
        let d = diag_off[k];
        for t in 0..pq_ends[k] {
            h_offdiag_edge::<true, false>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
        for t in pq_ends[k]..d {
            h_offdiag_edge::<false, false>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
        for t in d + 1..active_ends[k] {
            h_offdiag_edge::<false, false>(y_cp, y_ri, y_v, y_trans, col_starts, v, &lam_v, &inv_vmag, n_act, active_ends[k], k, t, h);
        }
    }

    // ── Diagonal pass: one sweep, every diagonal entry ────────────────
    let j_unit = Complex64::new(0.0, 1.0);
    for k in 0..n_act {
        let d = diag_off[k];
        let y_kk = y_v[y_cp[k] + d];
        let c_kk = lam_v[k] * (y_kk * v[k]).conj();
        let e_kk = v[k].conj() * (y_kk.conj() * lam_v[k] - d_lam[k]);
        let f_kk = c_kk - lam_v[k] * ibus[k].conj();

        let gaa = (e_kk + f_kk).re;
        unsafe {
            *h.add(col_starts[k] + d) = gaa;
            if k < npq {
                let ae = active_ends[k];
                let gva = (j_unit * inv_vmag[k] * (e_kk - f_kk)).re;
                *h.add(col_starts[k] + ae + d) = gva; // va
                let vc = col_starts[n_act + k];
                *h.add(vc + d) = gva; // av = va on the diagonal
                *h.add(vc + ae + d) = 2.0 * inv_vmag[k] * inv_vmag[k] * c_kk.re; // vv
            }
        }
    }
}

/// Fill Jᵀ as a pure transpose-copy of the J block (§4.3).
///
/// `j_vals` holds the J block in graph layout (produced by
/// `fill_jacobian_v3`, whose layout *is* the graph pattern). For each Ybus
/// edge `p = (row k, col i)` with mirror offset `tm = y_trans[p] − y_cp[k]`:
///
/// ```text
/// J11 (θ col, P seg) → Jᵀ[θ_k, P_i]   (P col, θ seg)
/// J21 (θ col, Q seg) → Jᵀ[θ_k, Q_i]   (Q col, θ seg)
/// J12 (|V| col, P seg) → Jᵀ[|V|_k, P_i] (P col, |V| seg)
/// J22 (|V| col, Q seg) → Jᵀ[|V|_k, Q_i] (Q col, |V| seg)
/// ```
///
/// Every Jᵀ slot is written exactly once; the mapping is a bijection.
pub fn fill_jt(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    j_vals: &[f64],
    jt_vals: &mut [f64],
) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let npq = cache.n_pq();
    let (y_cp, y_ri) = (ybus.col_offsets(), ybus.row_indices());
    let (pq_ends, active_ends) = (cache.pq_ends(), cache.active_ends());
    let cs = &pat.graph.col_starts[..];
    let y_trans = cache.y_trans();
    debug_assert_eq!(j_vals.len(), pat.graph.nnz);
    debug_assert_eq!(jt_vals.len(), pat.graph.nnz);

    let j = j_vals.as_ptr();
    let jt = jt_vals.as_mut_ptr();

    for i in 0..n_act {
        let (pq_end, active_end) = (pq_ends[i], active_ends[i]);
        for t in 0..active_end {
            let p = y_cp[i] + t;
            let k = y_ri[p];
            let tm = y_trans[p] - y_cp[k];
            unsafe {
                *jt.add(cs[k] + tm) = *j.add(cs[i] + t);
                if t < pq_end {
                    *jt.add(cs[n_act + k] + tm) = *j.add(cs[i] + active_end + t);
                }
                if i < npq {
                    // Target is column k's |V| segment: its start is column
                    // k's own θ-segment length, not the loop column's.
                    *jt.add(cs[k] + active_ends[k] + tm) = *j.add(cs[n_act + i] + t);
                    if t < pq_end {
                        *jt.add(cs[n_act + k] + active_ends[k] + tm) =
                            *j.add(cs[n_act + i] + active_end + t);
                    }
                }
            }
        }
    }
}

/// LM inner-loop μ update (§4.4): `dμ` added to the `aa` diagonal of every
/// θ column and the `vv` diagonal of every |V| column. The main fill is
/// never re-run for a μ change.
pub fn apply_mu_delta(pat: &KktPattern, h_vals: &mut [f64], dmu: f64) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let cs = &pat.graph.col_starts[..];
    let (active_ends, diag_off) = (cache.active_ends(), cache.diag_off());

    for k in 0..n_act {
        unsafe {
            *h_vals.get_unchecked_mut(cs[k] + diag_off[k]) += dmu;
            if k < cache.n_pq() {
                *h_vals.get_unchecked_mut(cs[n_act + k] + active_ends[k] + diag_off[k]) += dmu;
            }
        }
    }
}
