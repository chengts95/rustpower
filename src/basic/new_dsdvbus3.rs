use super::new_dsdvbus2::JacobianPattern2;
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
pub fn fill_jacobian_v3(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    scalc: &[Complex64], // V * conj(I)
    pattern: &JacobianPattern2,
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
        let pq_end = pattern.pq_ends[k];
        let active_end = pattern.active_ends[k];

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
        let diag_offset = pattern.diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();

        let out_j11 = jslice!(j_ptr, pattern.j11_starts[k], active_end);
        let out_j21 = jslice!(j_ptr, pattern.j21_starts[k], pq_end);
        let out_j12 = jslice!(j_ptr, pattern.j12_starts[k], active_end);
        let out_j22 = jslice!(j_ptr, pattern.j22_starts[k], pq_end);
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
            slot!(j_values, pattern.j11_starts[k] + diag_offset) += -qk;
            slot!(j_values, pattern.j21_starts[k] + diag_offset) += pk;
            slot!(j_values, pattern.j12_starts[k] + diag_offset) += pk * inv_vmag;
            slot!(j_values, pattern.j22_starts[k] + diag_offset) += qk * inv_vmag;
        }
    }

    for k in npq..n_active {
        let y_start = y_col_offsets[k];
        let pq_end = pattern.pq_ends[k];
        let active_end = pattern.active_ends[k];
        let ek = v[k].re;
        let fk = v[k].im;
        let qk = scalc[k].im;
        let diag_offset = pattern.diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();

        let out_j11 = jslice!(j_ptr, pattern.j11_starts[k], active_end);
        let out_j21 = jslice!(j_ptr, pattern.j21_starts[k], pq_end);
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

            // 复数乘法部分
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            // 只写入 out_j11
            out_j11[offset] = fi * Va_re - ei * Va_im;
        }
        unsafe {
            slot!(j_values, pattern.j11_starts[k] + diag_offset) += -qk;
        }
    }
}
