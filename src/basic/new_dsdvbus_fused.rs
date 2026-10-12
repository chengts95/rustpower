//! Experimental fusion of V4 Jacobian filling with Ybus times voltage.
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

macro_rules! jslice {
    ($ptr:expr, $start:expr, $len:expr) => {{ unsafe { std::slice::from_raw_parts_mut($ptr.add($start), $len) } }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::jacobian_cache::JacobianCache;
    use crate::basic::new_dsdvbus4::fill_jacobian_v4;
    use crate::basic::newtonpf::csc_matvec_and_scalc;
    use nalgebra_sparse::CooMatrix;

    fn check<const FLAT: bool>(y: &CscMatrix<Complex64>, v: &[Complex64], npv: usize, npq: usize) {
        let cache = JacobianCache::build_from_permuted(y.col_offsets(), y.row_indices(), npv, npq);
        let vn: Vec<_> = v.iter().map(|v| v / v.norm()).collect();
        let mut currents = vec![Complex64::new(0.0, 0.0); v.len()];
        let mut powers = currents.clone();
        csc_matvec_and_scalc(
            y.col_offsets(),
            y.row_indices(),
            y.values(),
            v,
            &mut currents,
            &mut powers,
        );
        let nnz = cache.j_row_indices.len();
        let len = if FLAT { 2 * nnz + 7 } else { nnz };
        let mut reference = vec![12345.0; len];
        fill_jacobian_v4::<FLAT>(
            y,
            v,
            &vn,
            &powers,
            &cache.j_col_ptrs,
            &cache.pq_ends,
            &cache.active_ends,
            &cache.diag_ptrs,
            npv,
            npq,
            &mut reference,
        );
        let mut fused = vec![12345.0; len];
        let mut fused_currents = vec![Complex64::new(f64::NAN, f64::NAN); v.len()];
        let mut fused_powers = fused_currents.clone();
        // Repeated calls must clear accumulated currents and overwrite all J slots.
        for _ in 0..2 {
            fill_jacobian_and_scalc_v4::<FLAT>(
                y,
                v,
                &vn,
                &mut fused_currents,
                &mut fused_powers,
                &cache.j_col_ptrs,
                &cache.pq_ends,
                &cache.active_ends,
                &cache.diag_ptrs,
                npv,
                npq,
                &mut fused,
            );
            assert_eq!(currents, fused_currents);
            assert_eq!(powers, fused_powers);
            assert_eq!(reference, fused);
        }
    }

    #[test]
    fn fused_matches_separate_sweeps_including_slack_contributions() {
        let mut coo = CooMatrix::new(5, 5);
        for k in 0..5 {
            coo.push(k, k, Complex64::new(2.0, -7.0));
        }
        for (i, k) in [(0, 1), (0, 2), (2, 3), (0, 4), (1, 4)] {
            coo.push(i, k, Complex64::new(-0.7, 3.2));
            coo.push(k, i, Complex64::new(-1.1, 4.3));
        }
        let mut y = CscMatrix::from(&coo);
        for scale in [1.0, 1.13] {
            for value in y.values_mut() {
                *value *= scale;
            }
            let v: Vec<_> = (0..5)
                .map(|k| Complex64::from_polar((0.9 + 0.07 * k as f64) * scale, 0.13 * k as f64))
                .collect();
            for (npv, npq) in [(1, 2), (3, 0), (0, 3), (0, 0), (0, 5), (5, 0)] {
                check::<false>(&y, &v, npv, npq);
                check::<true>(&y, &v, npv, npq);
            }
        }
    }
}

/// Assemble the Jacobian and calculated bus powers in one Ybus sweep.
/// Currents are accumulated in the original CSC order, including slack rows
/// and columns. Apply power-dependent diagonal corrections after all currents
/// are complete. Destination layout requirements are the same as V4.
#[allow(non_snake_case, clippy::too_many_arguments)]
#[inline(always)]
pub fn fill_jacobian_and_scalc_v4<const FLAT: bool>(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    ibus: &mut [Complex64],
    scalc: &mut [Complex64], // V * conj(I)
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
    ibus.fill(Complex64::new(0.0, 0.0));

    for k in 0..npq {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];

        let ek = v[k].re;
        let fk = v[k].im;
        let enk = Vnorm[k].re;
        let fnk = Vnorm[k].im;

        let j_ptr = j_values.as_mut_ptr();

        // 本列自己的四个象限起始，从 cs 和段长现算（PQ 母线四个象限都存在）。
        let seg_len = active_end + pq_end; // L_k：bus k 的整列长度
        let (j11_col, j12_col) = if FLAT {
            // δ-列 = [H 段 | J 段]，H 段起于 2·cs[c]
            (
                2 * j_col_ptrs[k] + seg_len,
                2 * j_col_ptrs[n_active + k] + seg_len,
            )
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
            ibus[i].re += Va_re;
            ibus[i].im += Va_im;

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
            ibus[i].re += Va_re;
            ibus[i].im += Va_im;

            let ei = v[i].re;
            let fi = v[i].im;

            // 第二组复数乘法: Vm
            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

            // 只写入两个输出
            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
        }

        // Entries in slack rows contribute to currents but have no J rows.
        for p in y_start + active_end..y_col_offsets[k + 1] {
            ibus[y_row_indices[p]] += y_vals[p] * v[k];
        }
    }

    for k in npq..n_active {
        let y_start = y_col_offsets[k];
        let pq_end = pq_ends[k];
        let active_end = active_ends[k];
        let ek = v[k].re;
        let fk = v[k].im;
        let j_ptr = j_values.as_mut_ptr();

        // PV 母线只有 θ 列：只算 j11/j21 两个起始（|V| 列 n_active+k 不存在）。
        let seg_len = active_end + pq_end;
        let j11_col = if FLAT {
            2 * j_col_ptrs[k] + seg_len
        } else {
            j_col_ptrs[k]
        };
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
            ibus[i].re += Va_re;
            ibus[i].im += Va_im;

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
            ibus[i].re += Va_re;
            ibus[i].im += Va_im;

            let ei = v[i].re;
            let fi = v[i].im;

            // 只写入 out_j11
            out_j11[offset] = fi * Va_re - ei * Va_im;
        }
        for p in y_start + active_end..y_col_offsets[k + 1] {
            ibus[y_row_indices[p]] += y_vals[p] * v[k];
        }
    }

    // Slack columns have no J columns, but still affect every bus current.
    for k in n_active..v.len() {
        for p in y_col_offsets[k]..y_col_offsets[k + 1] {
            ibus[y_row_indices[p]] += y_vals[p] * v[k];
        }
    }
    for i in 0..v.len() {
        scalc[i] = v[i] * ibus[i].conj();
    }

    // These O(n) corrections read no Ybus values.
    for k in 0..n_active {
        let active_end = active_ends[k];
        let seg_len = active_end + pq_ends[k];
        let first = if FLAT {
            2 * j_col_ptrs[k] + seg_len
        } else {
            j_col_ptrs[k]
        };
        let d = diag_ptrs[k] - y_col_offsets[k];
        j_values[first + d] -= scalc[k].im;
        if k < npq {
            let second = if FLAT {
                2 * j_col_ptrs[n_active + k] + seg_len
            } else {
                j_col_ptrs[n_active + k]
            };
            let inv_vmag = 1.0 / (v[k].re * Vnorm[k].re + v[k].im * Vnorm[k].im);
            j_values[first + active_end + d] += scalc[k].re;
            j_values[second + d] += scalc[k].re * inv_vmag;
            j_values[second + active_end + d] += scalc[k].im * inv_vmag;
        }
    }
}
