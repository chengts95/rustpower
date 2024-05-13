use nalgebra::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};

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

#[inline(always)]
pub fn slice_csc_matrix_block<T: Clone + Scalar + ClosedAdd + num_traits::Zero>(
    mat: &CscMatrix<T>,
    star_pos: (usize, usize),
    shape: (usize, usize),
) -> CscMatrix<T> {
    let (start_row, start_col) = star_pos;
    let (end_row, end_col) = (shape.0 + start_row, shape.1 + start_col);

    let coo_triplets: Vec<_> = mat
        .triplet_iter()
        .filter_map(|(r, c, v)| {
            if r >= start_row && r < end_row && c >= start_col && c < end_col {
                Some((r - start_row, c - start_col, v.clone()))
            } else {
                None
            }
        })
        .collect();
    let mut coo = CooMatrix::new(shape.0, shape.1);
    coo.reserve(coo_triplets.len());
    for i in coo_triplets {
        coo.push(i.0, i.1, i.2);
    }
    CscMatrix::from(&coo)
}
