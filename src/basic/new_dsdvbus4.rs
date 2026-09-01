//! Fourth-generation numeric fill — the no-stored-column-starts design.
//!
//! Numerically identical to [`super::new_dsdvbus3::fill_jacobian_v3`]; the
//! only difference is where the four quadrant starts come from. V3 reads
//! `JacobianPattern2::{j11,j21,j12,j22}_starts` (four precomputed tables);
//! V4 derives them inline, per column, from the graph column starts and the
//! segment lengths — all already present for the pattern itself:
//!
//! ```text
//! J11(k) = cs[k]              J21(k) = J11(k) + active_ends[k]
//! J12(k) = cs[n_active + k]   J22(k) = J12(k) + active_ends[k]
//! ```
//!
//! Rationale: one source of truth. Stored quadrant tables are affine copies
//! of `cs` + segment cuts; any future change to the layout convention would
//! have to update them in lockstep, and a missed one is a silent misaligned
//! write. Derivation cannot drift. It also costs nothing at runtime: the
//! starts are one add on values already hot in registers/L1, versus an extra
//! dependent load per table entry.
//!
//! `FLAT = false`: block view — `j_values` is the J block's own array,
//! column c starts at `cs[c]` (identical layout to V3).
//!
//! `FLAT = true`: flat LM view (`src::lm`, architecture doc §3.3) —
//! `j_values` is the global KKT values array `[μI+H Jᵀ; J −I]`; δ-column c
//! is `[H col c | J col c]` with the H segment at `2·cs[c]`, so the J
//! segment starts at `2·cs[c] + L_c`, `L_c = active_ends + pq_ends`.
//!
//! V3 stays untouched on the PF path; V4 is benchmarked against it in the
//! tests below (`v4_vs_v3_*`).

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

macro_rules! slot {
    ($a:expr, $i:expr) => {
        *$a.get_unchecked_mut($i)
    };
}
macro_rules! jslice {
    ($ptr:expr, $start:expr, $len:expr) => {{ unsafe { std::slice::from_raw_parts_mut($ptr.add($start), $len) } }};
}

/// V4 numeric fill. Same sweeps, same formulas, same write positions as V3;
/// quadrant starts computed inline. See module docs for the two views.
#[allow(non_snake_case, clippy::too_many_arguments)]
#[inline(always)]
pub fn fill_jacobian_v4<const FLAT: bool>(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    scalc: &[Complex64], // V * conj(I)
    j_col_ptrs: &[usize],
    pq_ends: &[usize],
    active_ends: &[usize],
    diag_ptrs: &[usize],
    npv: usize,
    npq: usize,
    j_values: &mut [f64],
) {
    let y_col_offsets = Ybus.col_offsets();
    let y_row_indices = Ybus.row_indices();
    let y_vals = Ybus.values();
    let n_active = npv + npq;

    for k in 0..npq {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];

        let ek = v[k].re;
        let fk = v[k].im;
        let enk = Vnorm[k].re;
        let fnk = Vnorm[k].im;

        // scalc[k] = (ek + jfk) * (Irek - jIimk) = (ek*Irek + fk*Iimk) + j(fk*Irek - ek*Iimk)
        // P_calc = ek*Irek + fk*Iimk
        // Q_calc = fk*Irek - ek*Iimk
        let pk = scalc[k].re;
        let qk = scalc[k].im;

        let vmag = ek * enk + fk * fnk;
        let inv_vmag = 1.0 / vmag;
        let diag_offset = diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();

        // 本列自己的四个象限起始，从 cs 和段长现算（PQ 母线四个象限都存在）。
        let seg_len = active_end + pq_end; // L_k：bus k 的整列长度
        let (j11_col, j12_col) = if FLAT {
            // δ-列 = [H 段 | J 段]，H 段起于 2·cs[c]
            (2 * j_col_ptrs[k] + seg_len, 2 * j_col_ptrs[n_active + k] + seg_len)
        } else {
            (j_col_ptrs[k], j_col_ptrs[n_active + k])
        };
        let j21_col = j11_col + active_end;
        let j22_col = j12_col + active_end;

        let out_j11 = jslice!(j_ptr, j11_col, active_end);
        let out_j21 = jslice!(j_ptr, j21_col, pq_end);
        let out_j12 = jslice!(j_ptr, j12_col, active_end);
        let out_j22 = jslice!(j_ptr, j22_col, pq_end);
        // 第一部分：处理 offset 在 [0, pq_end) 范围内的情况
        // 所有四个输出数组都需要写入
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            // 第一组复数乘法: Va
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            // 第二组复数乘法: Vm
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            // 写入所有四个输出
            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j21[offset] = -(ei * Va_re + fi * Va_im);
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
            out_j22[offset] = fi * Vm_re - ei * Vm_im;
        }

        // 第二部分：处理 offset 在 [pq_end, active_end) 范围内的情况
        // 只写入 out_j11 和 out_j12，out_j21 和 out_j22 不写入
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            // 第一组复数乘法: Va
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            // 第二组复数乘法: Vm
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            // 只写入两个输出
            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
        }

        // Diagonal corrections
        unsafe {
            slot!(j_values, j11_col + diag_offset) += -qk;
            slot!(j_values, j21_col + diag_offset) += pk;
            slot!(j_values, j12_col + diag_offset) += pk * inv_vmag;
            slot!(j_values, j22_col + diag_offset) += qk * inv_vmag;
        }
    }

    for k in npq..n_active {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];
        let ek = v[k].re;
        let fk = v[k].im;
        let qk = scalc[k].im;
        let diag_offset = diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();

        // PV 母线只有 θ 列：只算 j11/j21 两个起始（|V| 列 n_active+k 不存在）。
        let seg_len = active_end + pq_end;
        let j11_col = if FLAT { 2 * j_col_ptrs[k] + seg_len } else { j_col_ptrs[k] };
        let j21_col = j11_col + active_end;

        let out_j11 = jslice!(j_ptr, j11_col, active_end);
        let out_j21 = jslice!(j_ptr, j21_col, pq_end);
        // 第一部分：处理 offset 在 [0, pq_end) 范围内的情况
        // 这里两个数组都需要写入
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            // 复数乘法部分
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            // 两个输出都需要计算
            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j21[offset] = -(ei * Va_re + fi * Va_im);
        }

        // 第二部分：处理 offset 在 [pq_end, active_end) 范围内的情况
        // 这里只需要写入 out_j11，out_j21 不需要写入（或保持原值）
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            // 只写入 out_j11
            out_j11[offset] = fi * Va_re - ei * Va_im;
        }
        unsafe {
            slot!(j_values, j11_col + diag_offset) += -qk;
        }
    }
}

/// EXPERIMENT — fused single-pass J + Jᵀ fill (block view).
///
/// Same sweeps and formulas as [`fill_jacobian_v4`]; every computed value is
/// written twice: once at its J position (contiguous, per-column) and once
/// at its transposed Jᵀ position (scattered into the mirror column via
/// `y_trans`). This replaces the separate `fill_jt` copy pass — J elements
/// are still computed exactly once, but the Ybus is traversed once instead
/// of twice.
///
/// Transposed targets for entry `(row i, col k, mirror offset
/// `tm = y_trans[p] − y_cp[k→i]`)` — note the fused loop walks column `k`,
/// so the mirror offset is taken against column `i`:
///
/// ```text
/// J11 (θ col, P seg) → Jᵀ P-col i, θ seg:  cs[i] + tm
/// J21 (θ col, Q seg) → Jᵀ Q-col n_a+i, θ:  cs[n_a+i] + tm        (i PQ)
/// J12 (|V| col, P seg) → Jᵀ P-col i, |V|:  cs[i] + ae[i] + tm
/// J22 (|V| col, Q seg) → Jᵀ Q-col n_a+i:   cs[n_a+i] + ae[i] + tm (i PQ)
/// ```
///
/// Safety invariants (same bijection as `fill_jt`, opposite traversal
/// direction): every Jᵀ slot is written exactly once by the sweeps; the
/// diagonal corrections below are `+=` applied after the whole column —
/// including its diagonal entry — has been written. `j_values` and
/// `jt_values` must be disjoint slices.
#[allow(non_snake_case, clippy::too_many_arguments, dead_code)]
#[inline(always)]
pub fn fill_j_and_jt_exp(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    scalc: &[Complex64], // V * conj(I)
    j_col_ptrs: &[usize],
    pq_ends: &[usize],
    active_ends: &[usize],
    diag_ptrs: &[usize],
    y_trans: &[usize],
    npv: usize,
    npq: usize,
    j_values: &mut [f64],
    jt_values: &mut [f64],
) {
    let y_col_offsets = Ybus.col_offsets();
    let y_row_indices = Ybus.row_indices();
    let y_vals = Ybus.values();
    let n_active = npv + npq;

    let j_ptr = j_values.as_mut_ptr();
    let jt_ptr = jt_values.as_mut_ptr();

    for k in 0..npq {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];

        let ek = v[k].re;
        let fk = v[k].im;
        let enk = Vnorm[k].re;
        let fnk = Vnorm[k].im;
        let pk = scalc[k].re;
        let qk = scalc[k].im;

        let vmag = ek * enk + fk * fnk;
        let inv_vmag = 1.0 / vmag;
        let diag_offset = diag_ptrs[k] - y_start;

        // 本列自己的 J 两列起始（PQ 母线 θ 列和 |V| 列都存在）。
        let j11_col = j_col_ptrs[k];
        let j12_col = j_col_ptrs[n_active + k];
        let j21_col = j11_col + active_end;
        let j22_col = j12_col + active_end;

        let out_j11 = jslice!(j_ptr, j11_col, active_end);
        let out_j21 = jslice!(j_ptr, j21_col, pq_end);
        let out_j12 = jslice!(j_ptr, j12_col, active_end);
        let out_j22 = jslice!(j_ptr, j22_col, pq_end);

        // 第一部分：[0, pq_end)，行 i 必为 PQ —— 四个 J 输出 + 四个 Jᵀ 散射。
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            let ei = v[i].re;
            let fi = v[i].im;

            let j11 = fi * Va_re - ei * Va_im;
            let j21 = -(ei * Va_re + fi * Va_im);
            let j12 = ei * Vm_re + fi * Vm_im;
            let j22 = fi * Vm_re - ei * Vm_im;
            out_j11[offset] = j11;
            out_j21[offset] = j21;
            out_j12[offset] = j12;
            out_j22[offset] = j22;

            // 目标列 i 自己的起始与镜像偏移，现算（i 是 PQ：两列都存在）。
            let tm = y_trans[y_ptr] - y_col_offsets[i];
            let (jt_p, jt_q) = (j_col_ptrs[i], j_col_ptrs[n_active + i]);
            let ae_i = active_ends[i];
            unsafe {
                *jt_ptr.add(jt_p + tm) = j11;
                *jt_ptr.add(jt_q + tm) = j21;
                *jt_ptr.add(jt_p + ae_i + tm) = j12;
                *jt_ptr.add(jt_q + ae_i + tm) = j22;
            }
        }

        // 第二部分：[pq_end, active_end)，行 i 可能是 PV —— 两个 J 输出 +
        // 两个 Jᵀ 散射（Q 列目标不存在，不碰）。
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            let ei = v[i].re;
            let fi = v[i].im;

            let j11 = fi * Va_re - ei * Va_im;
            let j12 = ei * Vm_re + fi * Vm_im;
            out_j11[offset] = j11;
            out_j12[offset] = j12;

            let tm = y_trans[y_ptr] - y_col_offsets[i];
            let jt_p = j_col_ptrs[i];
            unsafe {
                *jt_ptr.add(jt_p + tm) = j11;
                *jt_ptr.add(jt_p + active_ends[i] + tm) = j12;
            }
        }

        // Diagonal corrections（列 k 的全部槽位 —— 含对角 —— 已写完，+= 安全）。
        unsafe {
            let d = diag_offset;
            let ae = active_end;
            slot!(j_values, j11_col + d) += -qk;
            slot!(j_values, j21_col + d) += pk;
            slot!(j_values, j12_col + d) += pk * inv_vmag;
            slot!(j_values, j22_col + d) += qk * inv_vmag;
            // Jᵀ 的对角镜像槽：P/Q 列 k 自己的起始 + 列内对角偏移。
            let (jt_p, jt_q) = (j_col_ptrs[k], j_col_ptrs[n_active + k]);
            *jt_ptr.add(jt_p + d) += -qk;
            *jt_ptr.add(jt_q + d) += pk;
            *jt_ptr.add(jt_p + ae + d) += pk * inv_vmag;
            *jt_ptr.add(jt_q + ae + d) += qk * inv_vmag;
        }
    }

    for k in npq..n_active {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];
        let ek = v[k].re;
        let fk = v[k].im;
        let qk = scalc[k].im;
        let diag_offset = diag_ptrs[k] - y_start;

        // PV 母线只有 θ 列。
        let j11_col = j_col_ptrs[k];
        let j21_col = j11_col + active_end;

        let out_j11 = jslice!(j_ptr, j11_col, active_end);
        let out_j21 = jslice!(j_ptr, j21_col, pq_end);

        // 第一部分：[0, pq_end)，行 i 必为 PQ —— 两个 J 输出 + 两个 Jᵀ 散射。
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            let j11 = fi * Va_re - ei * Va_im;
            let j21 = -(ei * Va_re + fi * Va_im);
            out_j11[offset] = j11;
            out_j21[offset] = j21;

            let tm = y_trans[y_ptr] - y_col_offsets[i];
            unsafe {
                *jt_ptr.add(j_col_ptrs[i] + tm) = j11;
                *jt_ptr.add(j_col_ptrs[n_active + i] + tm) = j21;
            }
        }

        // 第二部分：[pq_end, active_end) —— 一个 J 输出 + 一个 Jᵀ 散射。
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            let j11 = fi * Va_re - ei * Va_im;
            out_j11[offset] = j11;

            let tm = y_trans[y_ptr] - y_col_offsets[i];
            unsafe {
                *jt_ptr.add(j_col_ptrs[i] + tm) = j11;
            }
        }
        unsafe {
            let d = diag_offset;
            slot!(j_values, j11_col + d) += -qk;
            *jt_ptr.add(j_col_ptrs[k] + d) += -qk;
        }
    }
}


#[cfg(test)]
mod tests {
    //! V4 vs V3: bitwise-identical output on synthetic fixtures and on the
    //! IEEE118 system, plus a fill-only timing comparison (assembly cost
    //! isolated — no solver involved).
    //!
    //! Timing run (release, output visible):
    //! ```text
    //! cargo test --release v4_vs_v3_perf_ieee118 -- --nocapture
    //! ```

    use super::*;
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    use crate::basic::ecs::powerflow::systems::PowerFlowMat;
    use crate::lm::{KktPattern, fill_jt};
    use crate::basic::new_dsdvbus2::JacobianPattern2;
    use crate::basic::new_dsdvbus3::fill_jacobian_v3;
    use crate::io::pandapower::{Network, load_csv_zip};
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    use std::time::{Duration, Instant};

    fn ybus_from_edges(nb: usize, edges: &[(usize, usize)]) -> CscMatrix<Complex64> {
        let mut coo = CooMatrix::new(nb, nb);
        for k in 0..nb {
            coo.push(k, k, Complex64::new(2.0, -7.0));
        }
        for &(i, j) in edges {
            let y = Complex64::new(-1.0, 4.0);
            coo.push(i, j, y);
            coo.push(j, i, y);
        }
        CscMatrix::from(&coo)
    }

    /// Point evaluation inputs: v, Vnorm, scalc at a non-trivial voltage.
    fn eval_inputs(nb: usize, ybus: &CscMatrix<Complex64>) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
        let v: Vec<Complex64> = (0..nb)
            .map(|k| {
                let ang = 0.03 * (1.3 * k as f64).sin() - 0.01 * k as f64;
                let mag = 1.0 + 0.004 * (2.1 * k as f64).cos();
                Complex64::from_polar(mag, ang)
            })
            .collect();
        let mut ibus = vec![Complex64::new(0.0, 0.0); nb];
        for j in 0..nb {
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
            }
        }
        let scalc: Vec<Complex64> = (0..nb).map(|i| v[i] * ibus[i].conj()).collect();
        let vnorm: Vec<Complex64> = (0..nb)
            .map(|i| {
                let m = v[i].norm();
                if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) }
            })
            .collect();
        (v, vnorm, scalc)
    }

    /// V4 block view on the same pattern tables V3 uses.
    fn fill_v4_block(
        ybus: &CscMatrix<Complex64>,
        pat: &JacobianPattern2,
        v: &[Complex64],
        vnorm: &[Complex64],
        scalc: &[Complex64],
        npv: usize,
        npq: usize,
        out: &mut [f64],
    ) {
        fill_jacobian_v4::<false>(
            ybus, v, vnorm, scalc,
            &pat.j_col_ptrs, &pat.pq_ends, &pat.active_ends, &pat.diag_ptrs,
            npv, npq, out,
        );
    }

    fn assert_v4_matches_v3(ybus: &CscMatrix<Complex64>, npv: usize, npq: usize) {
        let nb = ybus.ncols();
        let pat = JacobianPattern2::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);
        let (v, vnorm, scalc) = eval_inputs(nb, ybus);

        let mut j_v3 = vec![0.0; pat.nnz_j];
        fill_jacobian_v3::<false>(ybus, &v, &vnorm, &scalc, &pat.j_col_ptrs, &pat.pq_ends, &pat.active_ends, &pat.diag_ptrs, npv, npq, &mut j_v3);
        let mut j_v4 = vec![0.0; pat.nnz_j];
        fill_v4_block(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j_v4);

        assert_eq!(j_v3, j_v4, "V4 block view disagrees with V3");
    }

    #[test]
    fn v4_matches_v3_synthetic() {
        // 4 buses: 0,1 PQ; 2 PV; 3 slack.
        let y4 = ybus_from_edges(4, &[(0, 1), (0, 2), (1, 2), (2, 3)]);
        assert_v4_matches_v3(&y4, 1, 2);
        // 14 buses: 0..9 PQ, 9..13 PV, 13 slack; ring + chords.
        let nb = 14;
        let mut edges: Vec<(usize, usize)> = (0..nb).map(|i| (i, (i + 1) % nb)).collect();
        for i in (0..nb).step_by(2) {
            edges.push((i, (i + 3) % nb));
        }
        let y14 = ybus_from_edges(nb, &edges);
        assert_v4_matches_v3(&y14, 4, 9);
    }

    fn load_ieee118_mat() -> PowerFlowMat {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net: Network = load_csv_zip(&format!("{dir}/cases/IEEE118/data.zip")).unwrap();
        let mut pf = PowerGrid::default();
        pf.world_mut().insert_resource(PPNetwork(net));
        pf.init_pf_net();
        pf.world()
            .get_resource::<PowerFlowMat>()
            .expect("init_pf_net did not produce a PowerFlowMat resource")
            .clone()
    }

    #[test]
    fn v4_matches_v3_ieee118() {
        let mat = load_ieee118_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let nb = ybus.ncols();
        let pat = JacobianPattern2::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);

        // The PF initial point: ibus/scalc from the flat start.
        let v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let mut ibus = vec![Complex64::new(0.0, 0.0); nb];
        for j in 0..nb {
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
            }
        }
        let scalc: Vec<Complex64> = (0..nb).map(|i| v[i] * ibus[i].conj()).collect();
        let vnorm: Vec<Complex64> = (0..nb)
            .map(|i| {
                let m = v[i].norm();
                if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) }
            })
            .collect();

        let mut j_v3 = vec![0.0; pat.nnz_j];
        fill_jacobian_v3::<false>(ybus, &v, &vnorm, &scalc, &pat.j_col_ptrs, &pat.pq_ends, &pat.active_ends, &pat.diag_ptrs, npv, npq, &mut j_v3);
        let mut j_v4 = vec![0.0; pat.nnz_j];
        fill_v4_block(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j_v4);

        assert_eq!(j_v3, j_v4, "V4 block view disagrees with V3 on IEEE118");
    }

    fn timeit(label: &str, repeats: usize, mut f: impl FnMut()) -> Duration {
        f(); // warm-up
        let mut total = Duration::ZERO;
        let mut min = Duration::MAX;
        for _ in 0..repeats {
            let t = Instant::now();
            f();
            let d = t.elapsed();
            total += d;
            min = min.min(d);
        }
        let avg = total / repeats as u32;
        println!("    {label:<24} avg {avg:?}   min {min:?}");
        avg
    }

    /// Fill-only race, no solver: V3 (stored quadrant tables) vs
    /// V4 (inline derivation). Isolates the table-load vs add difference.
    #[test]
    fn v4_vs_v3_perf_ieee118() {
        let mat = load_ieee118_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let nb = ybus.ncols();
        let pat = JacobianPattern2::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);
        let (v, vnorm, scalc) = eval_inputs(nb, ybus);

        let repeats = 2000;
        let mut j = vec![0.0; pat.nnz_j];
        println!("--- IEEE118 Jacobian fill: V3 (stored tables) vs V4 (inline) ---");
        let avg_v3 = timeit("V3 fill_jacobian_v3", repeats, | |
            fill_jacobian_v3::<false>(ybus, &v, &vnorm, &scalc, &pat.j_col_ptrs, &pat.pq_ends, &pat.active_ends, &pat.diag_ptrs, npv, npq, &mut j)
        );
        let avg_v4 = timeit("V4 fill_jacobian_v4", repeats, | |
            fill_v4_block(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j)
        );
        let ratio = avg_v3.as_secs_f64() / avg_v4.as_secs_f64();
        println!("    V3/V4 = {ratio:.3}x");
    }

    // ─── Fused J+Jᵀ experiment ──────────────────────────────────────────────

    /// Two-pass reference: v4 fill, then the fill_jt transpose-copy.
    fn two_pass_j_jt(
        ybus: &CscMatrix<Complex64>,
        pat: &KktPattern,
        v: &[Complex64],
        vnorm: &[Complex64],
        scalc: &[Complex64],
        npv: usize,
        npq: usize,
        j: &mut [f64],
        jt: &mut [f64],
    ) {
        let cache = &pat.cache;
        fill_jacobian_v4::<false>(
            ybus, v, vnorm, scalc,
            &pat.graph.col_starts, cache.pq_ends(), cache.active_ends(), cache.diag_ptrs(),
            npv, npq, j,
        );
        fill_jt::<false>(ybus, pat, j.as_ptr(), jt.as_mut_ptr());
    }

    fn fused_fill(
        ybus: &CscMatrix<Complex64>,
        pat: &KktPattern,
        v: &[Complex64],
        vnorm: &[Complex64],
        scalc: &[Complex64],
        npv: usize,
        npq: usize,
        j: &mut [f64],
        jt: &mut [f64],
    ) {
        let cache = &pat.cache;
        fill_j_and_jt_exp(
            ybus, v, vnorm, scalc,
            &pat.graph.col_starts, cache.pq_ends(), cache.active_ends(),
            cache.diag_ptrs(), cache.y_trans(),
            npv, npq, j, jt,
        );
    }

    fn assert_fused_matches_two_pass(ybus: &CscMatrix<Complex64>, npv: usize, npq: usize) {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, npv, npq);
        let (v, vnorm, scalc) = eval_inputs(nb, ybus);
        let nnz = pat.graph.nnz;

        let (mut j_ref, mut jt_ref) = (vec![0.0; nnz], vec![0.0; nnz]);
        two_pass_j_jt(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j_ref, &mut jt_ref);
        let (mut j_f, mut jt_f) = (vec![0.0; nnz], vec![0.0; nnz]);
        fused_fill(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j_f, &mut jt_f);

        assert_eq!(j_ref, j_f, "fused J disagrees with two-pass");
        assert_eq!(jt_ref, jt_f, "fused Jᵀ disagrees with two-pass");
    }

    #[test]
    fn fused_matches_two_pass_synthetic() {
        let y4 = ybus_from_edges(4, &[(0, 1), (0, 2), (1, 2), (2, 3)]);
        assert_fused_matches_two_pass(&y4, 1, 2);
        let nb = 14;
        let mut edges: Vec<(usize, usize)> = (0..nb).map(|i| (i, (i + 1) % nb)).collect();
        for i in (0..nb).step_by(2) {
            edges.push((i, (i + 3) % nb));
        }
        let y14 = ybus_from_edges(nb, &edges);
        assert_fused_matches_two_pass(&y14, 4, 9);
    }

    #[test]
    fn fused_matches_two_pass_ieee118() {
        let mat = load_ieee118_mat();
        assert_fused_matches_two_pass(&mat.y_bus, mat.npv, mat.npq);
    }

    /// The real race: (v4 + fill_jt) two passes vs the fused single pass.
    #[test]
    fn fused_vs_two_pass_perf_ieee118() {
        let mat = load_ieee118_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, npv, npq);
        let (v, vnorm, scalc) = eval_inputs(nb, ybus);
        let nnz = pat.graph.nnz;

        let repeats = 2000;
        let (mut j, mut jt) = (vec![0.0; nnz], vec![0.0; nnz]);
        println!("--- IEEE118: (v4 + fill_jt) two-pass vs fused single-pass ---");
        let avg_two = timeit("two-pass v4+fill_jt", repeats, | |
            two_pass_j_jt(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j, &mut jt)
        );
        let avg_fused = timeit("fused fill_j_and_jt", repeats, | |
            fused_fill(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j, &mut jt)
        );
        let ratio = avg_two.as_secs_f64() / avg_fused.as_secs_f64();
        println!("    two-pass/fused = {ratio:.3}x");
    }
}
