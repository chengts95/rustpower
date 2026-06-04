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

/// Performs a direct O(NNZ) sort-free permutation and format conversion from CSR to CSC.
///
/// Computes `Y_new = P * Y_old * P^T`, where `Y_old` is CSR, and outputs `Y_new` as CSC.
/// 
/// # Mathematical Algorithm: The "Sort-Free Scatter" Trick
/// 
/// In a valid CSC matrix, elements within each column MUST be strictly sorted by their row index.
/// To achieve this *without sorting*, we make the outer loop iterate over `new_row` in strictly
/// ascending order (0, 1, 2, ..., N-1). 
/// 
/// For each `new_row`, we map back to the `old_row` = `p_vec[new_row]`.
/// Because `Y_old` is in CSR format, we can instantly retrieve all non-zero elements of `old_row` in O(1).
/// Each element has an `old_col`, which maps to `new_col` = `p_inv[old_col]`.
/// We then scatter the value into the `new_col` bucket.
/// 
/// Because we process `new_row` sequentially, any element we drop into a `new_col` bucket
/// is guaranteed to have a `new_row` index strictly greater than the element we dropped into 
/// that same bucket during an earlier iteration. 
/// Thus, the CSC structure is perfectly formed and sorted in a single O(NNZ) pass!
///
/// # Arguments
/// * `n` - Dimension of the square matrix.
/// * `csr_indptr` - Row offsets of the input CSR matrix.
/// * `csr_indices` - Column indices of the input CSR matrix.
/// * `csr_data` - Values of the input CSR matrix.
/// * `p_vec` - Vector mapping new row indices to old row indices: `old_row = p_vec[new_row]`.
/// * `p_inv` - Vector mapping old column indices to new column indices: `new_col = p_inv[old_col]`.
pub fn permute_csr_to_csc_sort_free<T>(
    n: usize,
    csr_indptr: &[usize],
    csr_indices: &[usize],
    csr_data: &[T],
    p_vec: &[usize],
    p_inv: &[usize],
) -> CscMatrix<T>
where
    T: Scalar + Clone + Default,
{
    let nnz = csr_data.len();
    
    // Step 1: Count non-zeros per new column to pre-allocate CSC indptr
    let mut nnz_per_new_col = vec![0; n];
    for &old_col in csr_indices.iter() {
        let new_col = p_inv[old_col];
        nnz_per_new_col[new_col] += 1;
    }

    // Step 2: Build CSC col_offsets (indptr)
    let mut csc_indptr = vec![0; n + 1];
    for i in 0..n {
        csc_indptr[i + 1] = csc_indptr[i] + nnz_per_new_col[i];
    }

    // Step 3: Scatter data (Magically Sorted by design)
    let mut current_col_head = csc_indptr.clone();
    let mut csc_indices = vec![0; nnz];
    let mut csc_data = vec![T::default(); nnz];

    // The outer loop guarantees ascending `new_row` insertion!
    for new_row in 0..n {
        let old_row = p_vec[new_row];
        let start = csr_indptr[old_row];
        let end = csr_indptr[old_row + 1];
        
        for idx in start..end {
            let old_col = csr_indices[idx];
            let val = csr_data[idx].clone();
            let new_col = p_inv[old_col];
            
            let insert_idx = current_col_head[new_col];
            csc_indices[insert_idx] = new_row;
            csc_data[insert_idx] = val;
            current_col_head[new_col] += 1;
        }
    }

    CscMatrix::try_from_csc_data(n, n, csc_indptr, csc_indices, csc_data)
        .expect("Sort-free scatter produced invalid CSC matrix")
}

/// Permutes a CSC matrix `Y_new = P * Y_old * P^T`.
///
/// This iterates over columns, scatters elements into new columns, and performs a local 
/// sort on each column bucket. For highly sparse matrices (like power grids with avg degree ~3),
/// this local sorting is extremely fast ($O(NNZ \cdot \log(\text{avg\_degree}))$).
pub fn permute_csc_to_csc_local_sort<T>(
    csc: &CscMatrix<T>,
    p_vec: &[usize], // new -> old
    p_inv: &[usize], // old -> new
) -> CscMatrix<T>
where
    T: Scalar + Clone + Default,
{
    let n = csc.ncols();
    let indptr = csc.col_offsets();
    let indices = csc.row_indices();
    let data = csc.values();

    let mut nnz_per_new_col = vec![0; n];
    for old_col in 0..n {
        let new_col = p_inv[old_col];
        nnz_per_new_col[new_col] = indptr[old_col + 1] - indptr[old_col];
    }

    let mut csc_indptr = vec![0; n + 1];
    for i in 0..n {
        csc_indptr[i + 1] = csc_indptr[i] + nnz_per_new_col[i];
    }

    let mut csc_indices = vec![0; data.len()];
    let mut csc_data = vec![T::default(); data.len()];
    let mut current_col_head = csc_indptr.clone();

    for old_col in 0..n {
        let new_col = p_inv[old_col];
        let start = indptr[old_col];
        let end = indptr[old_col + 1];
        
        for idx in start..end {
            let old_row = indices[idx];
            let new_row = p_inv[old_row];
            let val = data[idx].clone();
            
            let insert_idx = current_col_head[new_col];
            csc_indices[insert_idx] = new_row;
            csc_data[insert_idx] = val;
            current_col_head[new_col] += 1;
        }
    }

    // Local sort within each column bucket
    for col in 0..n {
        let start = csc_indptr[col];
        let end = csc_indptr[col + 1];
        if end - start > 1 {
            let mut row_val_pairs: Vec<(usize, T)> = csc_indices[start..end]
                .iter()
                .zip(csc_data[start..end].iter())
                .map(|(&r, v)| (r, v.clone()))
                .collect();
            row_val_pairs.sort_by_key(|&(r, _)| r);
            for (i, (r, v)) in row_val_pairs.into_iter().enumerate() {
                csc_indices[start + i] = r;
                csc_data[start + i] = v;
            }
        }
    }

    CscMatrix::try_from_csc_data(n, n, csc_indptr, csc_indices, csc_data).unwrap()
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

    #[test]
    fn test_permute_csr_to_csc_sort_free() {
        let n = 3;
        // Y_old CSR matrix:
        // [1.0, 2.0, 0.0]
        // [0.0, 3.0, 4.0]
        // [5.0, 0.0, 6.0]
        let csr_indptr = vec![0, 2, 4, 6];
        let csr_indices = vec![0, 1, 1, 2, 0, 2];
        let csr_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // Permutation: swap row 0 and row 2
        // p_vec maps new_row to old_row: [2, 1, 0]
        // p_inv maps old_col to new_col: [2, 1, 0]
        let p_vec = vec![2, 1, 0];
        let p_inv = vec![2, 1, 0];

        // Expected Y_new = P * Y_old * P^T
        // P swaps rows 0 and 2. P^T swaps cols 0 and 2.
        // Y_new:
        // [6.0, 0.0, 5.0]
        // [4.0, 3.0, 0.0]
        // [0.0, 2.0, 1.0]
        
        let y_new_csc = permute_csr_to_csc_sort_free(
            n, &csr_indptr, &csr_indices, &csr_data, &p_vec, &p_inv
        );

        // CSC representation of Y_new:
        // col 0: [6.0, 4.0] at rows [0, 1]
        // col 1: [3.0, 2.0] at rows [1, 2]
        // col 2: [5.0, 1.0] at rows [0, 2]
        let expected_indptr = vec![0, 2, 4, 6];
        let expected_indices = vec![0, 1, 1, 2, 0, 2];
        let expected_data = vec![6.0, 4.0, 3.0, 2.0, 5.0, 1.0];

        assert_eq!(y_new_csc.col_offsets(), expected_indptr.as_slice());
        assert_eq!(y_new_csc.row_indices(), expected_indices.as_slice());
        assert_eq!(y_new_csc.values(), expected_data.as_slice());
    }
}
