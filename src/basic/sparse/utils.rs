use nalgebra::Scalar;
use nalgebra_sparse::pattern::SparsityPattern;
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use num_traits::One;
use std::mem::MaybeUninit;

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
/// Thus, the CSC structure is formed and sorted by the scatter pass, with O(n + NNZ) total work.
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
    let (col_offsets, row_indices, values) =
        permute_csr_to_csc_parts(n, csr_indptr, csr_indices, csr_data, p_vec, p_inv);
    CscMatrix::try_from_csc_data(n, n, col_offsets, row_indices, values)
        .expect("Sort-free scatter produced invalid CSC matrix")
}

/// Converts and permutes trusted CSR input without scanning the resulting CSC pattern.
///
/// Computes `P * Y * P^T`, exactly as [`permute_csr_to_csc_sort_free`]. Column offsets
/// are reused as counts and scatter cursors, then restored. Output rows and values
/// are allocated uninitialized and written once. The trusted input contract permits
/// unchecked indexing and avoids the final full pattern validation.
///
/// # Safety
///
/// The caller must ensure all of the following:
/// - `csr_indptr` has length `n + 1`, starts at zero, ends at `csr_data.len()`,
///   and is nondecreasing; `csr_indices.len() == csr_data.len()`.
/// - Every CSR column index is below `n`, and each row contains no duplicate column.
///   Column indices within a CSR row need not be sorted.
/// - `p_vec` and `p_inv` are mutually inverse permutations of `0..n`.
///
/// These conditions make the counts exact and ensure each output entry is written
/// once. Processing new rows in ascending order gives strictly increasing row
/// indices within every output column, so the CSC pattern meets nalgebra's invariants.
pub unsafe fn permute_csr_to_csc_sort_free_unchecked<T>(
    n: usize,
    csr_indptr: &[usize],
    csr_indices: &[usize],
    csr_data: &[T],
    p_vec: &[usize],
    p_inv: &[usize],
) -> CscMatrix<T>
where
    T: Scalar + Copy,
{
    let nnz = csr_data.len();
    let mut col_offsets = vec![0usize; n + 1];
    let mut row_indices = Box::<[usize]>::new_uninit_slice(nnz);
    let mut values = Box::<[T]>::new_uninit_slice(nnz);

    // SAFETY: The caller guarantees all source indices and permutation maps are
    // in range, and row offsets cover exactly nnz entries. Thus column counts
    // sum to nnz and their prefix sums cannot overflow or leave the output.
    unsafe {
        for &old_col in csr_indices {
            let new_col = *p_inv.get_unchecked(old_col);
            *col_offsets.get_unchecked_mut(new_col) += 1;
        }
        let mut total = 0;
        for count in &mut col_offsets[..n] {
            let next = *count;
            *count = total;
            total += next;
        }
        col_offsets[n] = nnz;

        let rows_ptr = row_indices.as_mut_ptr();
        let values_ptr = values.as_mut_ptr();
        for new_row in 0..n {
            let old_row = *p_vec.get_unchecked(new_row);
            let start = *csr_indptr.get_unchecked(old_row);
            let end = *csr_indptr.get_unchecked(old_row + 1);
            for entry in start..end {
                let old_col = *csr_indices.get_unchecked(entry);
                let new_col = *p_inv.get_unchecked(old_col);
                let cursor = col_offsets.get_unchecked_mut(new_col);
                let destination = *cursor;
                *cursor += 1;
                // Every original row is visited once. Exact counts reserve
                // disjoint column ranges, so each slot is written exactly once.
                rows_ptr.add(destination).write(MaybeUninit::new(new_row));
                values_ptr
                    .add(destination)
                    .write(MaybeUninit::new(*csr_data.get_unchecked(entry)));
            }
        }
    }

    // Cursors now hold column ends. Shift them to recover column starts.
    let mut last = 0;
    for offset in &mut col_offsets {
        let end = *offset;
        *offset = last;
        last = end;
    }

    // SAFETY: The scatter above initializes every one of the nnz slots exactly
    // once before either buffer is exposed as initialized T/usize storage.
    let row_indices = unsafe { row_indices.assume_init().into_vec() };
    let values = unsafe { values.assume_init().into_vec() };

    #[cfg(debug_assertions)]
    assert!(
        SparsityPattern::try_from_offsets_and_indices(
            n,
            n,
            col_offsets.clone(),
            row_indices.clone()
        )
        .is_ok(),
        "Unchecked permutation contract violated"
    );

    // SAFETY: Exact column counts and the ascending-new-row scatter establish a
    // valid CSC pattern under the caller's CSR and permutation guarantees above.
    let pattern = unsafe {
        SparsityPattern::from_offset_and_indices_unchecked(n, n, col_offsets, row_indices)
    };
    CscMatrix::try_from_pattern_and_values(pattern, values)
        .expect("Scatter allocates exactly one value per output row index")
}

fn permute_csr_to_csc_parts<T>(
    n: usize,
    csr_indptr: &[usize],
    csr_indices: &[usize],
    csr_data: &[T],
    p_vec: &[usize],
    p_inv: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<T>)
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

    (csc_indptr, csc_indices, csc_data)
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
    use nalgebra::{DVector, vector};
    use nalgebra_sparse::CooMatrix;
    use num_complex::Complex64;

    fn assert_unchecked_matches_coo(
        n: usize,
        indptr: &[usize],
        indices: &[usize],
        values: &[Complex64],
        from_perm: &[usize],
    ) {
        let mut to_perm = vec![0; n];
        for (new, &old) in from_perm.iter().enumerate() {
            to_perm[old] = new;
        }
        let mut reference = CooMatrix::new(n, n);
        for old_row in 0..n {
            for entry in indptr[old_row]..indptr[old_row + 1] {
                reference.push(to_perm[old_row], to_perm[indices[entry]], values[entry]);
            }
        }
        let reference = CscMatrix::from(&reference);
        let checked = permute_csr_to_csc_sort_free(n, indptr, indices, values, from_perm, &to_perm);
        // SAFETY: Test inputs contain unique in-range columns and valid row offsets;
        // from_perm is a permutation and to_perm was constructed as its inverse.
        let unchecked = unsafe {
            permute_csr_to_csc_sort_free_unchecked(n, indptr, indices, values, from_perm, &to_perm)
        };
        for actual in [&checked, &unchecked] {
            assert_eq!(actual.col_offsets(), reference.col_offsets());
            assert_eq!(actual.row_indices(), reference.row_indices());
            assert_eq!(actual.values(), reference.values());
        }
    }

    #[test]
    fn unchecked_permutation_handles_unsorted_csr_empty_rows_and_complex_values() {
        // Asymmetric CSR, an empty row/column, unsorted column indices, explicit zero.
        let indptr = [0, 2, 2, 4, 5];
        let indices = [3, 0, 3, 1, 0];
        let values = [
            Complex64::new(2.0, -7.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-4.0, 3.0),
            Complex64::new(8.0, -2.0),
            Complex64::new(1.0, 9.0),
        ];
        assert_unchecked_matches_coo(4, &indptr, &indices, &values, &[2, 0, 3, 1]);
        assert_unchecked_matches_coo(4, &indptr, &indices, &values, &[0, 1, 2, 3]);
    }

    #[test]
    fn unchecked_permutation_matches_all_small_patterns_and_permutations() {
        fn permutations(values: &mut [usize], index: usize, output: &mut Vec<Vec<usize>>) {
            if index == values.len() {
                output.push(values.to_vec());
            } else {
                for next in index..values.len() {
                    values.swap(index, next);
                    permutations(values, index + 1, output);
                    values.swap(index, next);
                }
            }
        }
        for n in 0..=3 {
            let mut orders = Vec::new();
            permutations(&mut (0..n).collect::<Vec<_>>(), 0, &mut orders);
            for mask in 0usize..(1usize << (n * n)) {
                let mut indptr = vec![0];
                let mut indices = Vec::new();
                let mut values = Vec::new();
                for row in 0..n {
                    // Reverse input columns: output ordering must come from scatter.
                    for col in (0..n).rev() {
                        if mask & (1 << (row * n + col)) != 0 {
                            indices.push(col);
                            values.push(Complex64::new((row + 1) as f64, -(col as f64 + 0.5)));
                        }
                    }
                    indptr.push(indices.len());
                }
                for order in &orders {
                    assert_unchecked_matches_coo(n, &indptr, &indices, &values, order);
                }
            }
        }
    }

    #[test]
    fn unchecked_permutation_handles_larger_irregular_patterns() {
        let mut state = 0x519e_37a2_6b40_dfc1u64;
        for n in [17usize, 64, 129] {
            let mut indptr = vec![0];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in 0..n {
                for col in (0..n).rev() {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if row % 11 != 0 && state >> 60 < 4 {
                        indices.push(col);
                        values.push(Complex64::new(row as f64 + 0.25, -(col as f64 + 0.5)));
                    }
                }
                indptr.push(indices.len());
            }
            let mut order: Vec<_> = (0..n).collect();
            for position in (1..n).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                order.swap(position, state as usize % (position + 1));
            }
            assert_unchecked_matches_coo(n, &indptr, &indices, &values, &order);
        }
    }

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

        let y_new_csc =
            permute_csr_to_csc_sort_free(n, &csr_indptr, &csr_indices, &csr_data, &p_vec, &p_inv);

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
