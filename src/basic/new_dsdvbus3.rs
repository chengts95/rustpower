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
/// Third-generation numeric fill.
///
/// Optimizes by taking S_calc (V * conj(I)) directly to handle diagonal corrections,
/// potentially avoiding passing the full 'ibus' vector if not needed elsewhere.
#[allow(non_snake_case)]
#[inline(always)]
pub fn fill_jacobian_v3<const FLAT: bool>(
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

        // Re-deriving diag terms from ibus:
        // J11_diag: dP/dth = -Q_calc = ek*Iimk - fk*Irek
        // J21_diag: dQ/dth = P_calc = ek*Irek + fk*Iimk
        // J12_diag: dP/dVm = (P_calc + |V|^2*G)/Vm ... wait, simpler to just use ibus elements?
        // Actually, if we have scalc, we can't easily get ibus[k] without dividing by V.
        // But ibus[k] = conj(scalc[k] / v[k])
        // let ik_conj = scalc[k] / v[k];
        // let Ire_k = ik_conj.re;
        // let Iim_k = -ik_conj.im;
        let vmag = ek * enk + fk * fnk;
        let inv_vmag = 1.0 / vmag;
        let diag_offset = diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();
        let seg_len = active_end + pq_end;
        let (j11_col, j12_col) = if FLAT {
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
            let i = *unsafe { y_row_indices.get_unchecked(y_ptr) };
            let Y_ik = *unsafe { y_vals.get_unchecked(y_ptr) };

            // 第一组复数乘法: Va
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let vi = unsafe { *v.get_unchecked(i) };
            let ei = vi.re;
            let fi = vi.im;

            let j11 = fi * Va_re - ei * Va_im;
            let j21 = -(ei * Va_re + fi * Va_im);

            unsafe {
                *out_j11.get_unchecked_mut(offset) = j11;
                *out_j21.get_unchecked_mut(offset) = j21;
                *out_j12.get_unchecked_mut(offset) = -j21 * inv_vmag;
                *out_j22.get_unchecked_mut(offset) = j11 * inv_vmag;
            }
        }

        // 第二部分：处理 offset 在 [pq_end, active_end) 范围内的情况
        // 只写入 out_j11 和 out_j12，out_j21 和 out_j22 不写入
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = *unsafe { y_row_indices.get_unchecked(y_ptr) };
            let Y_ik = *unsafe { y_vals.get_unchecked(y_ptr) };

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let vi = unsafe { *v.get_unchecked(i) };
            let ei = vi.re;
            let fi = vi.im;

            let j11 = fi * Va_re - ei * Va_im;
            let j21 = -(ei * Va_re + fi * Va_im);

            unsafe {
                *out_j11.get_unchecked_mut(offset) = j11;
                *out_j12.get_unchecked_mut(offset) = -j21 * inv_vmag;
            }
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
        let seg_len = active_end + pq_end;
        let j11_col = if FLAT { 2 * j_col_ptrs[k] + seg_len } else { j_col_ptrs[k] };
        let j21_col = j11_col + active_end;


        let out_j11 = jslice!(j_ptr, j11_col, active_end);
        let out_j21 = jslice!(j_ptr, j21_col, pq_end);

        // 第一部分：处理 offset 在 [0, pq_end) 范围内的情况
        // 这里两个数组都需要写入
        for offset in 0..pq_end {
            let y_ptr = y_start + offset;
            let i = *unsafe { y_row_indices.get_unchecked(y_ptr) };
            let Y_ik = *unsafe { y_vals.get_unchecked(y_ptr) };

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let vi = unsafe { *v.get_unchecked(i) };
            let ei = vi.re;
            let fi = vi.im;

            unsafe {
                *out_j11.get_unchecked_mut(offset) = fi * Va_re - ei * Va_im;
                *out_j21.get_unchecked_mut(offset) = -(ei * Va_re + fi * Va_im);
            }
        }

        // 第二部分：处理 offset 在 [pq_end, active_end) 范围内的情况
        // 这里只需要写入 out_j11
        for offset in pq_end..active_end {
            let y_ptr = y_start + offset;
            let i = *unsafe { y_row_indices.get_unchecked(y_ptr) };
            let Y_ik = *unsafe { y_vals.get_unchecked(y_ptr) };

            // 复数乘法部分
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let vi = unsafe { *v.get_unchecked(i) };
            let ei = vi.re;
            let fi = vi.im;

            unsafe {
                *out_j11.get_unchecked_mut(offset) = fi * Va_re - ei * Va_im;
            }
        }
        unsafe {
            slot!(j_values, j11_col + diag_offset) += -qk;
        }
    }
}
