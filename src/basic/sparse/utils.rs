use nalgebra::Scalar;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use num_traits::One;

/// Creates a CSC permutation matrix P such that (P*v)[i] = v[p[i]].
///
/// This represents a row permutation where the element at index `p[i]` in the original vector
/// moves to index `i` in the permuted vector.
///
/// Mathematically, P_{i, p[i]} = 1.
/// In CSC format, this means column `j = p[i]` has a non-zero element (1.0) at row `i`.
pub fn csc_permutation<T>(n: usize, p: &[usize]) -> CscMatrix<T>
where
    T: Scalar + One,
{
    assert_eq!(p.len(), n, "Permutation vector length must match n");

    let mut row_indices = vec![0; n];
    for (i, &old_idx) in p.iter().enumerate() {
        row_indices[old_idx] = i;
    }

    let col_offsets: Vec<usize> = (0..=n).collect();
    let values = vec![T::one(); n];

    CscMatrix::try_from_csc_data(n, n, col_offsets, row_indices, values)
        .expect("Valid permutation data should never fail CSC construction")
}

/// Creates a CSR permutation matrix P such that (P*v)[i] = v[p[i]].
///
/// Mathematically, P_{i, p[i]} = 1.
/// In CSR format, this means row `i` has a non-zero element (1.0) at column `p[i]`.
pub fn csr_permutation<T>(n: usize, p: &[usize]) -> CsrMatrix<T>
where
    T: Scalar + One,
{
    assert_eq!(p.len(), n, "Permutation vector length must match n");

    let row_offsets: Vec<usize> = (0..=n).collect();
    let col_indices = p.to_vec();
    let values = vec![T::one(); n];

    CsrMatrix::try_from_csr_data(n, n, row_offsets, col_indices, values)
        .expect("Valid permutation data should never fail CSR construction")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{vector, DVector};

    #[test]
    fn test_csc_permutation() {
        let n = 3;
        let p = vec![2, 0, 1]; // v_new[0]=v[2], v_new[1]=v[0], v_new[2]=v[1]
        let p_mat: CscMatrix<f64> = csc_permutation(n, &p);

        let v = vector![10.0, 20.0, 30.0];
        let v_perm = &p_mat * &v;

        assert_eq!(v_perm, vector![30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_csc_permutation_identity() {
        let n = 4;
        let p: Vec<usize> = (0..n).collect();
        let p_mat: CscMatrix<f64> = csc_permutation(n, &p);

        let v = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let v_perm = &p_mat * &v;

        assert_eq!(v, v_perm);
    }
}
