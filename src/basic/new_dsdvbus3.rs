use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use super::new_dsdvbus2::JacobianPattern2;

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
        let ik_conj = scalc[k] / v[k];
        let Ire_k = ik_conj.re;
        let Iim_k = -ik_conj.im;

        let diag_offset = pattern.diag_ptrs[k] - y_start;
        let j_ptr = j_values.as_mut_ptr();
        
        let out_j11 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j11_starts[k]), active_end) };
        let out_j21 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j21_starts[k]), pq_end) };
        let out_j12 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j12_starts[k]), active_end) };
        let out_j22 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j22_starts[k]), pq_end) };

        for offset in 0..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            
            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
            if offset < pq_end {
                out_j21[offset] = -(ei * Va_re + fi * Va_im);
            }

            let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
            let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;
            out_j12[offset] = ei * Vm_re + fi * Vm_im;
            if offset < pq_end {
                out_j22[offset] = fi * Vm_re - ei * Vm_im;
            }
        }

        // Diagonal corrections
        unsafe {
            *j_values.get_unchecked_mut(pattern.j11_starts[k] + diag_offset) += -qk;
            *j_values.get_unchecked_mut(pattern.j21_starts[k] + diag_offset) += pk;
            *j_values.get_unchecked_mut(pattern.j12_starts[k] + diag_offset) += enk * Ire_k + fnk * Iim_k;
            *j_values.get_unchecked_mut(pattern.j22_starts[k] + diag_offset) += fnk * Ire_k - enk * Iim_k;
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
        let out_j11 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j11_starts[k]), active_end) };
        let out_j21 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j21_starts[k]), pq_end) };

        for offset in 0..active_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];
            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;
            let ei = v[i].re;
            let fi = v[i].im;
            out_j11[offset] = fi * Va_re - ei * Va_im;
            if offset < pq_end {
                out_j21[offset] = -(ei * Va_re + fi * Va_im);
            }
        }
        unsafe {
            *j_values.get_unchecked_mut(pattern.j11_starts[k] + diag_offset) += -qk;
        }
    }
}
