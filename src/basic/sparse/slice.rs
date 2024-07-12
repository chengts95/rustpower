use nalgebra::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// Slices a column span from a CSC matrix.
///
/// # Parameters
///
/// * `mat` - A reference to the input CSC matrix.
/// * `start_col` - The starting column index (inclusive).
/// * `end_col` - The ending column index (exclusive).
///
/// # Returns
///
/// A new CSC matrix containing the sliced columns.
///
/// # Panics
///
/// Panics if `start_col` is not less than `end_col`.
#[inline(always)]
pub fn slice_csc_matrix<T: Clone>(
    mat: &CscMatrix<T>,
    start_col: usize,
    end_col: usize,
) -> CscMatrix<T> {
    assert!(start_col < end_col, "illegal indices");
    let col_start_idx = mat.col_offsets()[start_col];
    let col_end_idx = mat.col_offsets()[end_col];

    let new_values = mat.values()[col_start_idx..col_end_idx].to_vec();
    let new_row_indices = mat.row_indices()[col_start_idx..col_end_idx].to_vec();
    let new_col_offsets = mat.col_offsets()[start_col..=end_col]
        .iter()
        .map(|&x| x - col_start_idx)
        .collect::<Vec<_>>();

    CscMatrix::try_from_csc_data(
        mat.nrows(),
        end_col - start_col,
        new_col_offsets,
        new_row_indices,
        new_values,
    )
    .unwrap()
}

/// Slices a column span from a CSC matrix into a destination CSC matrix.
///
/// # Parameters
///
/// * `mat` - A reference to the input CSC matrix.
/// * `start_col` - The starting column index (inclusive).
/// * `end_col` - The ending column index (exclusive).
/// * `dest` - A mutable reference to the destination CSC matrix.
///
/// # Panics
///
/// Panics if `start_col` is not less than `end_col`.
#[inline(always)]
pub fn slice_csc_matrix_to<T: Clone>(
    mat: &CscMatrix<T>,
    start_col: usize,
    end_col: usize,
    dest: &mut CscMatrix<T>,
) {
    assert!(start_col < end_col, "illegal indices");
    let col_start_idx = mat.col_offsets()[start_col];
    let col_end_idx = mat.col_offsets()[end_col];

    let new_values = &mat.values()[col_start_idx..col_end_idx];
    dest.values_mut().clone_from_slice(new_values);
}

/// Slices a block from a CSC matrix.
///
/// # Parameters
///
/// * `mat` - A reference to the input CSC matrix.
/// * `star_pos` - The starting position (row, col) of the block.
/// * `shape` - The shape (rows, cols) of the block.
///
/// # Returns
///
/// A new CSC matrix containing the sliced block.
#[inline(always)]
pub fn slice_csc_matrix_block<T: Clone + Scalar + ClosedAddAssign + num_traits::Zero>(
    mat: &CscMatrix<T>,
    star_pos: (usize, usize),
    shape: (usize, usize),
) -> CscMatrix<T> {
    let (start_row, start_col) = star_pos;
    let (end_row, end_col) = (shape.0 + start_row, shape.1 + start_col);

    let mut new_col_offsets = vec![0; shape.1 + 1];
    let mut new_values = Vec::new();
    let mut new_row_indices = Vec::new();

    for col in start_col..end_col {
        let col_start_idx = mat.col_offsets()[col];
        let col_end_idx = mat.col_offsets()[col + 1];

        for idx in col_start_idx..col_end_idx {
            let row = mat.row_indices()[idx];
            if row >= start_row && row < end_row {
                new_values.push(mat.values()[idx].clone());
                new_row_indices.push(row - start_row);
            }
        }

        new_col_offsets[col - start_col + 1] = new_values.len();
    }

    CscMatrix::try_from_csc_data(
        shape.0,
        shape.1,
        new_col_offsets,
        new_row_indices,
        new_values,
    )
    .unwrap()
}

/// Slices a block from a CSC matrix into a destination CSC matrix.
///
/// # Parameters
///
/// * `mat` - A reference to the input CSC matrix.
/// * `star_pos` - The starting position (row, col) of the block.
/// * `shape` - The shape (rows, cols) of the block.
/// * `dest` - A mutable reference to the destination CSC matrix.
#[inline(always)]
pub fn slice_csc_matrix_block_to<T: Copy + Clone + Scalar + ClosedAddAssign + num_traits::Zero>(
    mat: &CscMatrix<T>,
    star_pos: (usize, usize),
    shape: (usize, usize),
    dest: &mut CscMatrix<T>,
) {
    let (start_row, start_col) = star_pos;
    let (end_row, end_col) = (shape.0 + start_row, shape.1 + start_col);

    let new_values = dest.values_mut();

    let mut i = 0;
    for col in start_col..end_col {
        let col_start_idx = mat.col_offsets()[col];
        let col_end_idx = mat.col_offsets()[col + 1];

        for idx in col_start_idx..col_end_idx {
            let row = mat.row_indices()[idx];
            if row >= start_row && row < end_row {
                new_values[i] = mat.values()[idx];
                i += 1;
            }
        }
    }
}

/// Retrieves the non-zero indices of a block from a CSC matrix.
///
/// # Parameters
///
/// * `mat` - A reference to the input CSC matrix.
/// * `star_pos` - The starting position (row, col) of the block.
/// * `shape` - The shape (rows, cols) of the block.
///
/// # Returns
///
/// A vector containing the indices of non-zero elements in the block.
#[inline(always)]
pub fn csc_matrix_block_nnz_indices<T: Copy + Clone + Scalar + ClosedAddAssign + num_traits::Zero>(
    mat: &CscMatrix<T>,
    star_pos: (usize, usize),
    shape: (usize, usize),
) -> Vec<usize> {
    let (start_row, start_col) = star_pos;
    let (end_row, end_col) = (shape.0 + start_row, shape.1 + start_col);

    let mut new_idx = vec![];

    for col in start_col..end_col {
        let col_start_idx = mat.col_offsets()[col];
        let col_end_idx = mat.col_offsets()[col + 1];

        for idx in col_start_idx..col_end_idx {
            let row = mat.row_indices()[idx];
            if row >= start_row && row < end_row {
                new_idx.push(idx);
            }
        }
    }
    new_idx
}