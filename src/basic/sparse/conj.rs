use nalgebra::*;
use nalgebra_sparse::{CscMatrix, CsrMatrix};

/// Trait for computing the conjugate of a matrix.
pub(crate) trait Conjugate {
    type Mat;

    /// Returns the conjugate of the matrix.
    fn conjugate(&self) -> Self::Mat;

    /// Computes the conjugate of the matrix in-place.
    fn conjugate_mut(&mut self);
}

impl<T: SimdRealField> Conjugate for CscMatrix<Complex<T>>
where
    Complex<T>: SimdComplexField,
{
    type Mat = CscMatrix<Complex<T>>;

    fn conjugate(&self) -> Self::Mat {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let values = values.conjugate();

        unsafe {
            CscMatrix::try_from_csc_data(
                self.nrows(),
                self.ncols(),
                self.col_offsets().to_vec(),
                self.row_indices().to_vec(),
                values.as_slice().to_vec(),
            )
            .unwrap_unchecked()
        }
    }

    fn conjugate_mut(&mut self) {
        let mut values = unsafe {
            let data = ViewStorageMut::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values_mut().as_mut_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        values.conjugate_mut();
    }
}

/// Trait for extracting the real and imaginary parts of a matrix.
pub(crate) trait RealImage {
    type Mat;

    /// Returns the real and imaginary parts of the matrix.
    fn real_imag(&self) -> (Self::Mat, Self::Mat);

    /// Returns the real part of the matrix.
    fn real(&self) -> Self::Mat;

    /// Returns the imaginary part of the matrix.
    fn imag(&self) -> Self::Mat;
}

impl<T: SimdRealField> RealImage for CscMatrix<Complex<T>>
where
    Complex<T>: SimdComplexField,
{
    type Mat = CscMatrix<<Complex<T> as SimdComplexField>::SimdRealField>;

    fn real(&self) -> Self::Mat {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v = values.map(|e| e.simd_real());
        let real_mat = unsafe {
            CscMatrix::try_from_pattern_and_values(self.pattern().clone(), v.as_slice().to_vec())
                .unwrap_unchecked()
        };
        real_mat
    }

    fn imag(&self) -> Self::Mat {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v = values.map(|e| e.simd_imaginary());
        let imag_mat = unsafe {
            CscMatrix::try_from_pattern_and_values(self.pattern().clone(), v.as_slice().to_vec())
                .unwrap_unchecked()
        };
        imag_mat
    }

    fn real_imag(&self) -> (Self::Mat, Self::Mat) {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v1 = values.map(|e| e.simd_real());
        let v2 = values.map(|e| e.simd_imaginary());
        let real_mat = unsafe {
            CscMatrix::try_from_pattern_and_values(self.pattern().clone(), v1.as_slice().to_vec())
                .unwrap_unchecked()
        };
        let imag_mat = unsafe {
            CscMatrix::try_from_pattern_and_values(self.pattern().clone(), v2.as_slice().to_vec())
                .unwrap_unchecked()
        };

        (real_mat, imag_mat)
    }
}

impl<T: SimdRealField> RealImage for CsrMatrix<Complex<T>>
where
    Complex<T>: SimdComplexField,
{
    type Mat = CsrMatrix<<Complex<T> as SimdComplexField>::SimdRealField>;

    fn real(&self) -> Self::Mat {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v = values.map(|e| e.simd_real());
        let real_mat = unsafe {
            CsrMatrix::try_from_pattern_and_values(self.pattern().clone(), v.as_slice().to_vec())
                .unwrap_unchecked()
        };
        real_mat
    }

    fn imag(&self) -> Self::Mat {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v = values.map(|e| e.simd_imaginary());
        let imag_mat = unsafe {
            CsrMatrix::try_from_pattern_and_values(self.pattern().clone(), v.as_slice().to_vec())
                .unwrap_unchecked()
        };
        imag_mat
    }

    fn real_imag(&self) -> (Self::Mat, Self::Mat) {
        let values = unsafe {
            let data = ViewStorage::<_, Dyn, U1, U1, Dyn>::from_raw_parts(
                self.values().as_ptr(),
                (nalgebra::Dyn(self.nnz()), U1),
                (U1, nalgebra::Dyn(self.nnz())),
            );
            Matrix::from_data_statically_unchecked(data)
        };
        let v1 = values.map(|e| e.simd_real());
        let v2 = values.map(|e| e.simd_imaginary());
        let real_mat = unsafe {
            CsrMatrix::try_from_pattern_and_values(self.pattern().clone(), v1.as_slice().to_vec())
                .unwrap_unchecked()
        };
        let imag_mat = unsafe {
            CsrMatrix::try_from_pattern_and_values(self.pattern().clone(), v2.as_slice().to_vec())
                .unwrap_unchecked()
        };

        (real_mat, imag_mat)
    }
}

// 测试模块
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::*;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    /// Tests the conjugate operation.
    #[test]
    fn test_conj() {
        let mut a = CooMatrix::new(6, 6);
        a.push(0, 0, Complex::new(1.0, -1.0));
        a.push(2, 1, Complex::new(3.0, 1.0));
        a.push(3, 3, Complex::new(5.0, -2.0));
        a.push(4, 4, Complex::new(4.0, 2.0));
        a.push(5, 5, Complex::new(6.0, -2.2));
        let a: CscMatrix<_> = (&a).into();
        let mut b = CooMatrix::new(6, 6);
        b.push(0, 0, Complex::new(1.0, 1.0));
        b.push(2, 1, Complex::new(3.0, -1.0));
        b.push(3, 3, Complex::new(5.0, 2.0));
        b.push(4, 4, Complex::new(4.0, -2.0));
        b.push(5, 5, Complex::new(6.0, 2.2));
        let b: CscMatrix<_> = (&b).into();
        println!("a={}", DMatrix::from(&a));
        println!("b={}", DMatrix::from(&b));
        println!("conj(a)={}", DMatrix::from(&a.conjugate()));
        assert!(a.conjugate() == b, "matrices do not match!")
    }

    /// Tests the in-place conjugate operation.
    #[test]
    fn test_conj_mut() {
        let mut a = CooMatrix::new(6, 6);
        a.push(0, 0, Complex::new(1.0, -1.0));
        a.push(2, 1, Complex::new(3.0, 1.0));
        a.push(3, 3, Complex::new(5.0, -2.0));
        a.push(4, 4, Complex::new(4.0, 2.0));
        a.push(5, 5, Complex::new(6.0, -2.2));
        let mut a: CscMatrix<_> = (&a).into();
        let mut b = CooMatrix::new(6, 6);
        b.push(0, 0, Complex::new(1.0, 1.0));
        b.push(2, 1, Complex::new(3.0, -1.0));
        b.push(3, 3, Complex::new(5.0, 2.0));
        b.push(4, 4, Complex::new(4.0, -2.0));
        b.push(5, 5, Complex::new(6.0, 2.2));
        let b: CscMatrix<_> = (&b).into();
        a.conjugate_mut();
        println!("a={}", DMatrix::from(&a));
        println!("b={}", DMatrix::from(&b));
        println!("conj(a)={}", DMatrix::from(&a));
        assert!(a == b, "matrices do not match!")
    }
}
