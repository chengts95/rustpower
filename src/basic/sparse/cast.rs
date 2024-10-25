use nalgebra::*;
use nalgebra_sparse::CscMatrix;
use simba::scalar::{SubsetOf, SupersetOf};
pub(crate) trait Cast<T> {
    type Mat;

    /// Returns the conjugate of the matrix.
    fn cast(&self) -> Self::Mat;
}
pub(crate) trait DownCast<T> {
    type Mat;

    /// Returns the conjugate of the matrix.
    fn down_cast(&self) -> Self::Mat;
}
impl Cast<Complex<f64>> for CscMatrix<f64> {
    type Mat = CscMatrix<Complex<f64>>;

    fn cast(&self) -> Self::Mat {
        unsafe {
            CscMatrix::try_from_csc_data(
                self.nrows(),
                self.ncols(),
                self.col_offsets().to_vec(),
                self.row_indices().to_vec(),
                self.values()
                    .iter()
                    .map(|x| Complex::new(*x, 0.0))
                    .collect(),
            )
            .unwrap_unchecked()
        }
    }
}

