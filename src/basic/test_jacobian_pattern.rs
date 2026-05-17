//! Symbolic pattern verification: old build_jacobian pattern vs new JacobianPattern.
//! Run with: cargo test --release test_jacobian_pattern -- --nocapture

#[cfg(test)]
mod tests {
    use nalgebra_sparse::CscMatrix;
    use num_complex::Complex64;

    use crate::basic::new_dsdvbus::JacobianPattern;
    use crate::basic::newtonpf::Slice;
    use crate::basic::sparse::{conj::RealImage, stack::*};

    /// Build a small 6x6 Ybus-like CSC matrix (already permuted PV→PQ→Ext):
    /// PV buses: {0,1}  PQ buses: {2,3}  External: {4,5}
    fn make_ybus() -> (CscMatrix<Complex64>, usize, usize) {
        let n = 6;
        let npv = 2;
        let npq = 2;
        let col_offsets = vec![0, 4, 8, 14, 20, 24, 28];
        let row_indices = vec![
            0, 1, 2, 3, // col 0 (PV)
            0, 1, 2, 3, // col 1 (PV)
            0, 1, 2, 3, 4, 5, // col 2 (PQ)
            0, 1, 2, 3, 4, 5, // col 3 (PQ)
            2, 3, 4, 5, // col 4 (Ext)
            2, 3, 4, 5, // col 5 (Ext)
        ];
        let values: Vec<Complex64> = (0..28)
            .map(|i| Complex64::new(1.0 + i as f64 * 0.1, (i as f64 * 0.05).sin()))
            .collect();
        (
            CscMatrix::try_from_csc_data(n, n, col_offsets, row_indices, values).unwrap(),
            npv,
            npq,
        )
    }

    fn old_jacobian_pattern(
        ybus: &CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
    ) -> (Vec<usize>, Vec<usize>, usize) {
        use crate::basic::dsbus_dv::dSbus_dV;
        use nalgebra::*;
        let n = ybus.ncols();
        let v = DVector::from_fn(n, |i, _| Complex64::from_polar(1.0, (i as f64) * 0.1));
        let vnorm = v.map(|e| e.simd_signum());
        let (ds_dvm, ds_dva) = dSbus_dV(ybus, &v, &vnorm);
        let n_ext = n - npv - npq;

        // --- inline build_jacobian (non-cached) ---
        let ds_dva = ds_dva.block((0, 0), (ds_dva.nrows() - n_ext, ds_dva.ncols() - n_ext));
        let ds_dvm = ds_dvm.block((0, 0), (ds_dvm.nrows() - n_ext, ds_dvm.ncols() - n_ext));
        let (real, imag) = ds_dva.real_imag();
        let (real2, imag2) = ds_dvm.real_imag();
        let j11 = real;
        let j12 = real2.columns(npv, real2.ncols());
        let j21 = imag.block((npv, 0), (imag.nrows() - npv, imag.ncols()));
        let j22 = imag2.block((npv, npv), (imag2.nrows() - npv, imag2.ncols() - npv));
        let j_old = csc_vstack(&[&csc_hstack(&[&j11, &j12]), &csc_hstack(&[&j21, &j22])]);
        let n = j_old.nrows();
        let (cp, ri, _) = j_old.disassemble();
        (cp, ri, n)
    }

    fn patterns_equal(
        old_ptrs: &[usize],
        old_rows: &[usize],
        old_ncol: usize,
        new: &JacobianPattern,
    ) -> bool {
        if old_ncol != new.j_col_ptrs.len() - 1 {
            eprintln!(
                "Col count: old={} new={}",
                old_ncol,
                new.j_col_ptrs.len() - 1
            );
            return false;
        }
        for col in 0..old_ncol {
            let os = old_ptrs[col];
            let oe = old_ptrs[col + 1];
            let ns = new.j_col_ptrs[col];
            let ne = new.j_col_ptrs[col + 1];
            if (oe - os) != (ne - ns) {
                eprintln!("Col {} nnz: old={} new={}", col, oe - os, ne - ns);
                return false;
            }
            if &old_rows[os..oe] != &new.j_row_indices[ns..ne] {
                eprintln!(
                    "Col {} rows differ: old={:?} new={:?}",
                    col,
                    &old_rows[os..oe],
                    &new.j_row_indices[ns..ne]
                );
                return false;
            }
        }
        true
    }

    #[test]
    fn test_jacobian_pattern_6bus() {
        let (ybus, npv, npq) = make_ybus();
        let (old_ptrs, old_rows, old_ncol) = old_jacobian_pattern(&ybus, npv, npq);
        let new_pat =
            JacobianPattern::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);

        println!("=== Old Jacobian ===");
        for col in 0..old_ncol {
            let s = old_ptrs[col];
            let e = old_ptrs[col + 1];
            println!("  col {:2}: {:?}", col, &old_rows[s..e]);
        }
        println!("=== New JacobianPattern ===");
        for col in 0..new_pat.j_col_ptrs.len() - 1 {
            let s = new_pat.j_col_ptrs[col];
            let e = new_pat.j_col_ptrs[col + 1];
            println!("  col {:2}: {:?}", col, &new_pat.j_row_indices[s..e]);
        }

        assert!(patterns_equal(&old_ptrs, &old_rows, old_ncol, &new_pat));
    }
}
