//! Phase 1/2 — numeric fill kernels for the LM augmented system (doc §4).
//!
//! All kernels honor the same discipline as `fill_jacobian_v3`:
//! region-split sweeps with no `i == j` branch in the hot loop, raw-pointer
//! writes, and every position derived at the point of use from the column's
//! own base plus the Ybus structure — no per-quadrant tables.
//!
//! Every kernel takes a `FLAT` const generic selecting the storage view
//! (doc §3.3); the traversal, the segment offsets and the write-once
//! coverage are identical in both views, only the per-column start rule
//! changes (`cs = pat.graph.col_starts`, `L_c = active_ends + pq_ends` of
//! the column's bus, `n = n_state`, `nnz = graph.nnz`):
//!
//! | block | block view (own slice) | flat view (one global CSC) |
//! |---|---|---|
//! | H col c | `cs[c]` | `2·cs[c]` |
//! | J col c | `cs[c]` | `2·cs[c] + L_c` |
//! | Jᵀ col c | `cs[c]` | `2·nnz + cs[c] + c` |
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
/// `h_col` / `h_vcol` are the column's own starts in the active view,
/// computed by the caller right before the sweep (|V| column only exists
/// for PQ buses, so `h_vcol` is only meaningful when `VCOL`).
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
    v: &[Complex64],
    lam_v: &[Complex64],
    inv_vmag: &[f64],
    active_end: usize,
    k: usize,
    t: usize,
    h_col: usize,
    h_vcol: usize,
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
        *h.add(h_col + t) = gaa;
        if FULL {
            let gva = (j_unit * inv_vmag[i] * (e_ik - c_ik)).re;
            *h.add(h_col + active_end + t) = gva;
        }
        if VCOL {
            let gav = (j_unit * inv_vmag[k] * (v[k].conj() * y_ik.conj() * lam_v[i] - c_ki)).re;
            *h.add(h_vcol + t) = gav;
            if FULL {
                let gvv = inv_vmag[i] * inv_vmag[k] * (c_ik + c_ki).re;
                *h.add(h_vcol + active_end + t) = gvv;
            }
        }
    }
}

/// Fill the H(r) block: `H(r) = Σ_k r_k·∇²F_k` with polar quadrants.
///
/// `v` covers **all** buses (slack included — its physics enters through
/// `ibus`); `r` is the reduced residual `[P (n_act); Q (n_pq)]`. The
/// multipliers are `λ_k = rP_k − i·rQ_k` (PQ), `rP_k` (PV), `0` (slack).
///
/// `FLAT = false`: `h_vals` is the H block's own slice, column c at `cs[c]`.
/// `FLAT = true`: `h_vals` is the global values array, column c at `2·cs[c]`.
pub fn fill_h<const FLAT: bool>(
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
    let cs = &pat.graph.col_starts[..];
    let y_trans = cache.y_trans();
    debug_assert!(h_vals.len() >= if FLAT { 3 * pat.graph.nnz + n_act + npq } else { pat.graph.nnz });

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
        // 本列自己的 θ 列与 |V| 列起始（PQ 母线两个列都存在），循环开头现算。
        let (h_col, h_vcol) = if FLAT {
            (2 * cs[k], 2 * cs[n_act + k])
        } else {
            (cs[k], cs[n_act + k])
        };
        let d = diag_off[k];
        for t in 0..d {
            h_offdiag_edge::<true, true>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, h_vcol, h);
        }
        for t in d + 1..pq_ends[k] {
            h_offdiag_edge::<true, true>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, h_vcol, h);
        }
        for t in pq_ends[k]..active_ends[k] {
            h_offdiag_edge::<false, true>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, h_vcol, h);
        }
    }
    for k in npq..n_act {
        // PV 母线只有 θ 列：|V| 列 n_act+k 不存在，不取它的起始。
        let h_col = if FLAT { 2 * cs[k] } else { cs[k] };
        let d = diag_off[k];
        for t in 0..pq_ends[k] {
            h_offdiag_edge::<true, false>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, 0, h);
        }
        for t in pq_ends[k]..d {
            h_offdiag_edge::<false, false>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, 0, h);
        }
        for t in d + 1..active_ends[k] {
            h_offdiag_edge::<false, false>(y_cp, y_ri, y_v, y_trans, v, &lam_v, &inv_vmag, active_ends[k], k, t, h_col, 0, h);
        }
    }

    // ── Diagonal pass: one sweep, every diagonal entry ────────────────
    let j_unit = Complex64::new(0.0, 1.0);
    for k in 0..n_act {
        let h_col = if FLAT { 2 * cs[k] } else { cs[k] };
        let d = diag_off[k];
        let y_kk = y_v[y_cp[k] + d];
        let c_kk = lam_v[k] * (y_kk * v[k]).conj();
        let e_kk = v[k].conj() * (y_kk.conj() * lam_v[k] - d_lam[k]);
        let f_kk = c_kk - lam_v[k] * ibus[k].conj();

        let gaa = (e_kk + f_kk).re;
        unsafe {
            *h.add(h_col + d) = gaa;
            if k < npq {
                // |V| 列只为 PQ 母线存在，在这里才算它自己的起始。
                let h_vcol = if FLAT { 2 * cs[n_act + k] } else { cs[n_act + k] };
                let ae = active_ends[k];
                let gva = (j_unit * inv_vmag[k] * (e_kk - f_kk)).re;
                *h.add(h_col + ae + d) = gva; // va
                *h.add(h_vcol + d) = gva; // av = va on the diagonal
                *h.add(h_vcol + ae + d) = 2.0 * inv_vmag[k] * inv_vmag[k] * c_kk.re; // vv
            }
        }
    }
}

/// Fill Jᵀ as a pure transpose-copy of the J block (§4.3).
///
/// The J values are read through `j_src` and the transposed copies written
/// through `jt_dst`. In the block view the two point at the J and Jᵀ block
/// slices; in the flat view both point at the one global values array (the
/// J and Jᵀ regions are disjoint, see §7 write-once coverage).
///
/// For each Ybus edge `p = (row k, col i)` with mirror offset
/// `tm = y_trans[p] − y_cp[k]`:
///
/// ```text
/// J11 (θ col, P seg) → Jᵀ[θ_k, P_i]   (P col, θ seg)
/// J21 (θ col, Q seg) → Jᵀ[θ_k, Q_i]   (Q col, θ seg)
/// J12 (|V| col, P seg) → Jᵀ[|V|_k, P_i] (P col, |V| seg)
/// J22 (|V| col, Q seg) → Jᵀ[|V|_k, Q_i] (Q col, |V| seg)
/// ```
///
/// Every Jᵀ slot is written exactly once; the mapping is a bijection.
///
/// # Safety
/// `j_src` must be readable for the whole J region and `jt_dst` writable
/// for the whole Jᵀ region in the active view (the regions may not overlap
/// except by exact pointer equality of distinct positions).
pub fn fill_jt<const FLAT: bool>(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    j_src: *const f64,
    jt_dst: *mut f64,
) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let npq = cache.n_pq();
    let nnz = pat.graph.nnz;
    let (y_cp, y_ri) = (ybus.col_offsets(), ybus.row_indices());
    let (pq_ends, active_ends) = (cache.pq_ends(), cache.active_ends());
    let cs = &pat.graph.col_starts[..];
    let y_trans = cache.y_trans();

    for i in 0..n_act {
        // 源 θ 列 i 自己的起始，循环开头现算（L_i = 列 i 的整列长度）。
        let (pq_end, active_end) = (pq_ends[i], active_ends[i]);
        let seg_len = active_end + pq_end;
        let src_theta = if FLAT { 2 * cs[i] + seg_len } else { cs[i] };
        for t in 0..active_end {
            let p = y_cp[i] + t;
            let k = y_ri[p];
            let tm = y_trans[p] - y_cp[k];
            // 目标 P 列 k 自己的起始（graph 的行都是 active，k < n_act 恒成立）。
            let dst_theta = if FLAT { 2 * nnz + cs[k] + k } else { cs[k] };
            unsafe {
                *jt_dst.add(dst_theta + tm) = *j_src.add(src_theta + t);
                if t < pq_end {
                    // 行 k 是 PQ：目标 Q 列 n_act+k 才存在，在这里算它自己的起始。
                    let dst_q = if FLAT {
                        2 * nnz + cs[n_act + k] + (n_act + k)
                    } else {
                        cs[n_act + k]
                    };
                    *jt_dst.add(dst_q + tm) = *j_src.add(src_theta + active_end + t);
                }
                if i < npq {
                    // 源 |V| 列 n_act+i 只为 PQ 母线存在，在这里算它自己的起始；
                    // 目标 |V| 段起点是列 k 自己的 θ 段长。
                    let src_vmag = if FLAT { 2 * cs[n_act + i] + seg_len } else { cs[n_act + i] };
                    let dst_vmag = active_ends[k];
                    *jt_dst.add(dst_theta + dst_vmag + tm) = *j_src.add(src_vmag + t);
                    if t < pq_end {
                        let dst_q = if FLAT {
                            2 * nnz + cs[n_act + k] + (n_act + k)
                        } else {
                            cs[n_act + k]
                        };
                        *jt_dst.add(dst_q + dst_vmag + tm) =
                            *j_src.add(src_vmag + active_end + t);
                    }
                }
            }
        }
    }
}

/// LM inner-loop μ update (§4.4): `dμ` added to the `aa` diagonal of every
/// θ column and the `vv` diagonal of every |V| column. The main fill is
/// never re-run for a μ change.
///
/// `FLAT = false`: `h_vals` is the H block's own slice.
/// `FLAT = true`: `h_vals` is the global values array (H col c at `2·cs[c]`).
pub fn apply_mu_delta<const FLAT: bool>(pat: &KktPattern, h_vals: &mut [f64], dmu: f64) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let cs = &pat.graph.col_starts[..];
    let (active_ends, diag_off) = (cache.active_ends(), cache.diag_off());

    for k in 0..n_act {
        // 本列自己的起始与对角偏移，现算。
        let h_col = if FLAT { 2 * cs[k] } else { cs[k] };
        unsafe {
            *h_vals.get_unchecked_mut(h_col + diag_off[k]) += dmu;
            if k < cache.n_pq() {
                // |V| 列只为 PQ 母线存在，在这里才算它自己的起始。
                let h_vcol = if FLAT { 2 * cs[n_act + k] } else { cs[n_act + k] };
                *h_vals.get_unchecked_mut(h_vcol + active_ends[k] + diag_off[k]) += dmu;
            }
        }
    }
}
