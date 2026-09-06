//! Borrowed Jacobian block fill, derived from the V3/V4 column sweeps.
//!
//! The containing matrix owns the column starts and symbolic cuts. This
//! operator owns no cache and knows nothing about the surrounding system.
//! CSR(J) and CSC(J^T) select the same row traversal.
//!
//! The shared edge quantity is C = diag(V) conj(Y) diag(conj(V)).
//! C(Y,V)^T = C(Y^T,conj(V)): the row sweep uses mirrored Y values and
//! conjugated voltage operands in the same V3 arithmetic. Transposing the
//! real Jacobian also swaps its cross blocks and moves magnitude scaling
//! to the original variable bus (the inner bus in the row sweep).

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Positions supplied by the owner of the destination sparse matrix.
#[derive(Clone, Copy)]
pub(crate) struct JacobianBlock<'a> {
    pub column_starts: &'a [usize],
    pub base: usize,
    pub prefix: usize,
}
impl JacobianBlock<'_> {
    #[inline(always)]
    fn start(self, column: usize) -> usize {
        self.base + self.column_starts[column] + self.prefix
    }
}

pub(crate) struct JacobianOperator<'a> {
    pub ybus: &'a CscMatrix<Complex64>,
    pub v: &'a [Complex64],
    pub inv_vmag: &'a [f64],
    pub scalc: &'a [Complex64],
    pub pq_ends: &'a [usize],
    pub active_ends: &'a [usize],
    pub diag_ptrs: &'a [usize],
    /// For row traversal: position of Y[k,i] in the live Y CSC values.
    /// Only the pattern must be symmetric; the numerical values need not be.
    pub mirror: &'a [usize],
    pub npq: usize,
    pub npv: usize,
}
impl JacobianOperator<'_> {
    #[inline(always)]
    pub fn fill<const CSR: bool, const TRANSPOSE: bool>(
        &self,
        block: JacobianBlock<'_>,
        values: &mut [f64],
    ) {
        if CSR != TRANSPOSE {
            debug_assert_eq!(self.mirror.len(), self.ybus.nnz());
            self.fill_columns::<true, true>(0..self.npq, block, values);
            self.fill_columns::<true, false>(self.npq..self.npq + self.npv, block, values);
        } else {
            self.fill_columns::<false, true>(0..self.npq, block, values);
            self.fill_columns::<false, false>(self.npq..self.npq + self.npv, block, values);
        }
    }

    #[inline(always)]
    fn fill_columns<const ROW: bool, const PQ: bool>(
        &self,
        columns: std::ops::Range<usize>,
        block: JacobianBlock<'_>,
        values: &mut [f64],
    ) {
        let cp = self.ybus.col_offsets();
        let ri = self.ybus.row_indices();
        let y = self.ybus.values();
        let n_active = self.npq + self.npv;
        for k in columns {
            let start = cp[k];
            let pq = self.pq_ends[k];
            let active = self.active_ends[k];
            let first = block.start(k);
            let second = if PQ { block.start(n_active + k) } else { 0 };
            let ek = self.v[k].re;
            let fk = if ROW { -self.v[k].im } else { self.v[k].im };
            let inv_k = self.inv_vmag[k];
            let ptr = values.as_mut_ptr();
            // These are disjoint segments in the containing matrix's value array,
            // with exactly the same local offsets as the corresponding Y column.
            let a = unsafe { std::slice::from_raw_parts_mut(ptr.add(first), active) };
            let b = unsafe { std::slice::from_raw_parts_mut(ptr.add(first + active), pq) };
            let c = unsafe {
                std::slice::from_raw_parts_mut(ptr.add(second), if PQ { active } else { 0 })
            };
            let d = unsafe {
                std::slice::from_raw_parts_mut(
                    ptr.add(second + if PQ { active } else { 0 }),
                    if PQ { pq } else { 0 },
                )
            };

            for t in 0..pq {
                let p = start + t;
                let i = ri[p];
                let yi = y[if ROW { self.mirror[p] } else { p }];
                let vi = self.v[i];
                let fi = if ROW { -vi.im } else { vi.im };
                let va_re = yi.re * ek - yi.im * fk;
                let va_im = yi.re * fk + yi.im * ek;
                let j11 = fi * va_re - vi.re * va_im;
                let j21 = -(vi.re * va_re + fi * va_im);
                let inv = if ROW { self.inv_vmag[i] } else { inv_k };
                unsafe {
                    *a.get_unchecked_mut(t) = j11;
                    *b.get_unchecked_mut(t) = if ROW { -j21 * inv } else { j21 };
                    if PQ {
                        *c.get_unchecked_mut(t) = if ROW { j21 } else { -j21 * inv };
                        *d.get_unchecked_mut(t) = j11 * inv;
                    }
                }
            }
            for t in pq..active {
                let p = start + t;
                let i = ri[p];
                let yi = y[if ROW { self.mirror[p] } else { p }];
                let vi = self.v[i];
                let fi = if ROW { -vi.im } else { vi.im };
                let va_re = yi.re * ek - yi.im * fk;
                let va_im = yi.re * fk + yi.im * ek;
                unsafe {
                    *a.get_unchecked_mut(t) = fi * va_re - vi.re * va_im;
                    if PQ {
                        let j21 = -(vi.re * va_re + fi * va_im);
                        *c.get_unchecked_mut(t) = if ROW { j21 } else { -j21 * inv_k };
                    }
                }
            }
            let diag = self.diag_ptrs[k] - start;
            let s = self.scalc[k];
            a[diag] -= s.im;
            if PQ {
                b[diag] += if ROW { s.re * inv_k } else { s.re };
                c[diag] += if ROW { s.re } else { s.re * inv_k };
                d[diag] += s.im * inv_k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::new_dsdvbus2::JacobianPattern2;
    use crate::basic::new_dsdvbus3::fill_jacobian_v3;
    use crate::lm::cache::YbusAnalysisCache;
    use nalgebra::DVector;
    use nalgebra_sparse::CooMatrix;

    #[test]
    fn storage_and_transpose_views_match_derivatives() {
        // Numerically asymmetric Y and nonzero angles catch accidental use of
        // Y[j,i], conjugation, or swapping the voltage roles with storage indices.
        let mut coo = CooMatrix::new(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                coo.push(
                    i,
                    j,
                    Complex64::new(
                        0.3 + (2 * i + j) as f64 * 0.17,
                        -0.8 + (i + 3 * j) as f64 * 0.11,
                    ),
                );
            }
        }
        let mut y = CscMatrix::from(&coo);
        let cache = YbusAnalysisCache::build(&y, 1, 2);
        let pat = JacobianPattern2::build_from_permuted(y.col_offsets(), y.row_indices(), 1, 2);
        let v: Vec<_> = (0..4)
            .map(|i| Complex64::from_polar(0.93 + i as f64 * 0.06, -0.17 + i as f64 * 0.13))
            .collect();
        let vn: Vec<_> = v.iter().map(|v| v / v.norm()).collect();
        for scale in [1.0, 1.13] {
            // Reuse the symbolic cache after a numeric admittance change.
            for value in y.values_mut() {
                *value *= scale;
            }
            let current = &y * &DVector::from_vec(v.clone());
            let s: Vec<_> = v
                .iter()
                .zip(current.iter())
                .map(|(v, i)| v * i.conj())
                .collect();
            let inv_vmag: Vec<_> = v.iter().map(|v| 1.0 / v.norm()).collect();
            let op = JacobianOperator {
                ybus: &y,
                v: &v,
                inv_vmag: &inv_vmag,
                scalc: &s,
                pq_ends: cache.pq_ends(),
                active_ends: cache.active_ends(),
                diag_ptrs: cache.diag_ptrs(),
                mirror: cache.y_trans(),
                npq: 2,
                npv: 1,
            };
            let layout = JacobianBlock {
                column_starts: &pat.j_col_ptrs,
                base: 0,
                prefix: 0,
            };
            let mut csc = vec![f64::NAN; pat.nnz_j];
            let mut csr = csc.clone();
            let mut csc_t = csc.clone();
            let mut csr_t = csc.clone();
            op.fill::<false, false>(layout, &mut csc);
            op.fill::<true, false>(layout, &mut csr);
            op.fill::<false, true>(layout, &mut csc_t);
            op.fill::<true, true>(layout, &mut csr_t);
            assert_eq!(csr, csc_t);
            assert_eq!(csc, csr_t);
            let mut reference = vec![0.; pat.nnz_j];
            fill_jacobian_v3(&y, &v, &vn, &s, &pat, 1, 2, &mut reference);
            for (a, b) in csc.iter().zip(&reference) {
                assert!((a - b).abs() < 2e-14);
            }
            // An enclosing matrix supplies irregular column gaps and a block
            // base. Only J's slots may change; prefix/tail entries stay intact.
            let mut starts = vec![0];
            for col in 0..5 {
                starts.push(starts[col] + pat.j_col_ptrs[col + 1] - pat.j_col_ptrs[col] + col + 2);
            }
            for transpose in [false, true] {
                let block = JacobianBlock {
                    column_starts: &starts,
                    base: 3,
                    prefix: 1,
                };
                let mut embedded = vec![12345.0; 3 + starts[5]];
                let mut written = vec![false; embedded.len()];
                if transpose {
                    op.fill::<false, true>(block, &mut embedded);
                } else {
                    op.fill::<false, false>(block, &mut embedded);
                }
                let expected = if transpose { &csc_t } else { &csc };
                for col in 0..5 {
                    for p in pat.j_col_ptrs[col]..pat.j_col_ptrs[col + 1] {
                        let target = 3 + starts[col] + 1 + p - pat.j_col_ptrs[col];
                        assert_eq!(embedded[target], expected[p]);
                        written[target] = true;
                    }
                }
                for (value, written) in embedded.iter().zip(written) {
                    if !written {
                        assert_eq!(*value, 12345.0);
                    }
                }
            }
            let dense = nalgebra::DMatrix::from(
                &CscMatrix::try_from_csc_data(
                    5,
                    5,
                    pat.j_col_ptrs.clone(),
                    pat.j_row_indices.clone(),
                    csc,
                )
                .unwrap(),
            );
            let transposed = nalgebra::DMatrix::from(
                &CscMatrix::try_from_csc_data(
                    5,
                    5,
                    pat.j_col_ptrs.clone(),
                    pat.j_row_indices.clone(),
                    csc_t,
                )
                .unwrap(),
            );
            // The transposed sweep reorders the complex products, so equality
            // is algebraic rather than bitwise. Check at rounding scale.
            let error = (&dense.transpose() - &transposed).amax();
            assert!(error < 8.0 * f64::EPSILON * dense.amax().max(1.0));
            for col in 0..5 {
                let eval = |h: f64| {
                    let mut x = v.clone();
                    let k = if col < 3 { col } else { col - 3 };
                    x[k] = if col < 3 {
                        Complex64::from_polar(v[k].norm(), v[k].arg() + h)
                    } else {
                        Complex64::from_polar(v[k].norm() + h, v[k].arg())
                    };
                    let currents = &y * &DVector::from_vec(x.clone());
                    let powers: Vec<_> = x
                        .iter()
                        .zip(currents.iter())
                        .map(|(v, i)| v * i.conj())
                        .collect();
                    vec![
                        powers[0].re,
                        powers[1].re,
                        powers[2].re,
                        powers[0].im,
                        powers[1].im,
                    ]
                };
                let plus = eval(1e-6);
                let minus = eval(-1e-6);
                for row in 0..5 {
                    let fd = (plus[row] - minus[row]) / 2e-6;
                    assert!(
                        (dense[(row, col)] - fd).abs() < 2e-9,
                        "({row},{col}): {} vs {fd}",
                        dense[(row, col)]
                    );
                }
            }
        }
    }
}
