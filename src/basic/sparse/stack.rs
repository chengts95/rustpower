use na::Scalar;
use nalgebra_sparse::{pattern::SparsityPattern, *};

pub enum Format {
    Csr,
    Csc,
}
// 定义矩阵格式的枚举
#[derive(Debug)]
pub enum SparseMatrix<T> {
    Csr(CsrMatrix<T>), // 代表 CSR 矩阵
    Csc(CscMatrix<T>), // 代表 CSC 矩阵
}
impl<T: Clone> From<&CsrMatrix<T>> for SparseMatrix<T> {
    fn from(value: &CsrMatrix<T>) -> Self {
        Self::Csr(value.clone())
    }
}
impl<T: Clone> From<&CscMatrix<T>> for SparseMatrix<T> {
    fn from(value: &CscMatrix<T>) -> Self {
        Self::Csc(value.clone())
    }
}
trait SpConvert {
    type DT;
    type S;
    fn from_csc(value: &CscMatrix<Self::DT>) -> Self::S;
    fn from_csr(value: &CsrMatrix<Self::DT>) -> Self::S;
}
impl<T: Clone + Scalar> SpConvert for CscMatrix<T> {
    type DT = T;
    type S = CscMatrix<T>;
    fn from_csc(value: &CscMatrix<Self::DT>) -> Self::S {
        value.clone()
    }

    fn from_csr(value: &CsrMatrix<Self::DT>) -> Self::S {
        value.into()
    }
}
impl<T: Clone + Scalar> SpConvert for CsrMatrix<T> {
    type DT = T;
    type S = CsrMatrix<T>;
    fn from_csr(value: &CsrMatrix<Self::DT>) -> Self::S {
        value.clone()
    }

    fn from_csc(value: &CscMatrix<Self::DT>) -> Self::S {
        value.into()
    }
}

impl<T: Scalar> SparseMatrix<T> {
    pub fn to_csc(&self) -> CscMatrix<T> {
        match self {
            SparseMatrix::Csr(a) => a.into(),
            SparseMatrix::Csc(a) => a.clone(),
        }
    }
    pub fn to_csr(&self) -> CsrMatrix<T> {
        match self {
            SparseMatrix::Csc(a) => a.into(),
            SparseMatrix::Csr(a) => a.clone(),
        }
    }
}
trait SpMat {
    type DT;
    fn values(&self) -> &[Self::DT];
    fn pattern(&self) -> &SparsityPattern;
    fn nnz(&self) -> usize;
    fn format() -> Format;
}
impl<T> SpMat for CscMatrix<T> {
    type DT = T;
    fn pattern(&self) -> &SparsityPattern {
        self.pattern()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
    fn values(&self) -> &[Self::DT] {
        self.values()
    }

    fn format() -> Format {
        Format::Csc
    }
}
impl<T> SpMat for CsrMatrix<T> {
    type DT = T;
    fn pattern(&self) -> &SparsityPattern {
        self.pattern()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
    fn values(&self) -> &[Self::DT] {
        self.values()
    }

    fn format() -> Format {
        Format::Csr
    }
}
#[inline]
fn minor_dim_stack<MT: SpMat<DT = T>, T: Clone>(
    matrices: &[&MT],
) -> (usize, usize, Vec<T>, Vec<usize>, Vec<usize>) {
    let pattern = matrices[0].pattern();
    let zminor_dim = pattern.minor_dim();
    let mut total_mjs = 0;
    let mut nnz = 0;

    // 预先计算总列数和非零元素的总数
    for mat in matrices {
        let pattern = mat.pattern();
        let minor_dim = pattern.minor_dim();
        assert_eq!(
            minor_dim, zminor_dim,
            "All matrices must have the same number of rows"
        );
        total_mjs += pattern.major_dim();
        nnz += mat.nnz();
    }
    let mut data: Vec<T> = Vec::with_capacity(nnz);
    let mut indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut indptr: Vec<usize> = Vec::with_capacity(total_mjs + 1);
    let mut current_offset = 0;
    for mat in matrices {
        let (pattern, values) = (mat.pattern(), mat.values());
        let major_dim = pattern.major_dim();
        indptr.extend(
            pattern.major_offsets()[..major_dim]
                .iter()
                .map(|x| x + current_offset),
        );
        indices.extend_from_slice(pattern.minor_indices());

        data.extend_from_slice(values);
        current_offset += values.len();
    }
    indptr.push(nnz);
    (zminor_dim, total_mjs, data, indices, indptr)
}
fn major_dim_stack<MT: SpMat<DT = T>, T: Clone>(
    matrices: &[&MT],
) -> (usize, usize, Vec<T>, Vec<usize>, Vec<usize>) {
    let pattern = matrices[0].pattern();
    let major_dim = pattern.major_dim();
    let mut minor_dim = 0;
    let mut nnz = 0;
    for mat in matrices {
        let p = mat.pattern();
        assert_eq!(
            p.major_dim(),
            pattern.major_dim(),
            "All matrices must have the same number of cols/rows"
        );
        minor_dim += p.minor_dim();
        nnz += mat.nnz();
    }
    let mut data: Vec<T> = Vec::with_capacity(nnz);
    let mut indices: Vec<usize> = Vec::with_capacity(nnz);
    let mut indptr: Vec<usize> = Vec::new();
    indptr.resize(major_dim + 1, 0);

    for i in 0..major_dim {
        let mut offset = 0;
        let mut count = 0;
        for mat in matrices {
            let pattern = mat.pattern();
            let start = pattern.major_offsets()[i];
            let end = pattern.major_offsets()[i + 1];
            let values = &mat.values()[start..end];
            let m_indices = &pattern.minor_indices()[start..end];
            data.extend_from_slice(values);
            indices.extend(m_indices.iter().map(|x| x + offset));
            offset += pattern.minor_dim();
            count += values.len();
        }
        indptr[i + 1] = indptr[i] + count;
    }
    (major_dim, minor_dim, data, indices, indptr)
}

pub fn csc_hstack<T: Clone>(matrices: &[&CscMatrix<T>]) -> CscMatrix<T> {
    let (zminor_dim, total_mjs, data, indices, indptr) = minor_dim_stack(matrices);
    unsafe {
        let new_pattern = SparsityPattern::from_offset_and_indices_unchecked(
            total_mjs, zminor_dim, indptr, indices,
        );
        CscMatrix::try_from_pattern_and_values(new_pattern, data).unwrap_unchecked()
    }
}
pub fn csr_vstack<T: Clone>(matrices: &[&CsrMatrix<T>]) -> CsrMatrix<T> {
    let (zminor_dim, total_mjs, data, indices, indptr) = minor_dim_stack(matrices);
    unsafe {
        let new_pattern = SparsityPattern::from_offset_and_indices_unchecked(
            total_mjs, zminor_dim, indptr, indices,
        );
        CsrMatrix::try_from_pattern_and_values(new_pattern, data).unwrap_unchecked()
    }
}
pub fn csc_vstack<T: Clone>(matrices: &[&CscMatrix<T>]) -> CscMatrix<T> {
    let (major_dim, minor_dim, data, indices, indptr) = major_dim_stack(matrices);

    unsafe {
        let new_pattern = SparsityPattern::from_offset_and_indices_unchecked(
            major_dim, minor_dim, indptr, indices,
        );
        CscMatrix::try_from_pattern_and_values(new_pattern, data).unwrap_unchecked()
    }
}

pub fn csr_hstack<T: Clone>(matrices: &[&CsrMatrix<T>]) -> CsrMatrix<T> {
    let (major_dim, minor_dim, data, indices, indptr) = major_dim_stack(matrices);

    unsafe {
        let new_pattern = SparsityPattern::from_offset_and_indices_unchecked(
            major_dim, minor_dim, indptr, indices,
        );
        CsrMatrix::try_from_pattern_and_values(new_pattern, data).unwrap_unchecked()
    }
}

fn vstack<T: Clone + Scalar, U: SpMat<DT = T> + SpConvert<DT = T, S = U>>(
    matrices: &[&SparseMatrix<T>],
) -> U {
    match U::format() {
        Format::Csr => {
            let mats: Vec<_> = matrices.iter().map(|x| x.to_csr()).collect();
            let matsref: Vec<_> = mats.iter().map(|x| x).collect();
            U::from_csr(&csr_vstack(matsref.as_slice()))
        }
        Format::Csc => {
            let mats: Vec<_> = matrices.iter().map(|x| x.to_csc()).collect();
            let matsref: Vec<_> = mats.iter().map(|x| x).collect();
            U::from_csc(&csc_vstack(matsref.as_slice()))
        }
    }
}
// 测试模块
#[cfg(test)]
mod tests {
    use nalgebra::*;
    // 导入外部作用域中的所有内容
    use super::*;

    // 一个测试案例
    #[test]
    fn test_csc_hstack() {
        // 创建第一个稀疏矩阵
        let mut mat1 = CooMatrix::new(3, 2);

        mat1.push(2, 1, 3);

        // 创建第二个稀疏矩阵
        let mut mat2 = CooMatrix::new(3, 3);
        mat2.push(0, 0, 2);
        mat2.push(1, 1, 4);
        mat2.push(2, 2, 6);

        let mut mat3 = CooMatrix::new(3, 5);
        mat3.push(2, 1, 3);
        mat3.push(0, 2, 2);
        mat3.push(1, 3, 4);
        mat3.push(2, 4, 6);
        let b = CscMatrix::from(&mat3);

        println!("a={:?}", CscMatrix::from(&mat1).disassemble());
        println!("b={:?}", CscMatrix::from(&mat2).disassemble());

        let a = csc_hstack(&[&CscMatrix::from(&mat1), &CscMatrix::from(&mat2)]);
        println!("hstack(a,b)={:?}", a.clone().disassemble());
        println!(
            "hstack(a,b) should be = {:?}",
            CscMatrix::from(&mat3).disassemble()
        );
        assert!(a == b, "matrices do not match!")
    }

    #[test]
    fn test_csr_hstack() {
        // 创建第一个稀疏矩阵
        let mut mat1 = CooMatrix::new(3, 2);

        mat1.push(2, 1, 3);

        // 创建第二个稀疏矩阵
        let mut mat2 = CooMatrix::new(3, 3);
        mat2.push(0, 0, 2);
        mat2.push(1, 1, 4);
        mat2.push(2, 2, 6);

        let mut mat3 = CooMatrix::new(3, 5);
        mat3.push(2, 1, 3);
        mat3.push(0, 2, 2);
        mat3.push(1, 3, 4);
        mat3.push(2, 4, 6);
        let b = CsrMatrix::from(&mat3);
        let a = csr_hstack(&[&CsrMatrix::from(&mat1), &CsrMatrix::from(&mat2)]);

        println!("a={:?}", CsrMatrix::from(&mat1).disassemble());
        println!("b={:?}", CsrMatrix::from(&mat2).disassemble());
        println!("hstack(a,b)={:?}", a.clone().disassemble());
        println!(
            "hstack(a,b) should be = {:?}",
            CsrMatrix::from(&mat3).disassemble()
        );
        assert!(a == b, "matrices do not match!")
    }
    // 一个测试案例
    #[test]
    fn test_csr_vstack() {
        // 创建第一个稀疏矩阵
        let mut mat1 = CooMatrix::new(2, 3);
        mat1.push(1, 2, 3);

        // 创建第二个稀疏矩阵
        let mut mat2 = CooMatrix::new(3, 3);
        mat2.push(0, 0, 2);
        mat2.push(1, 1, 4);
        mat2.push(2, 2, 6);

        let mut mat3 = CooMatrix::new(5, 3);
        mat3.push(1, 2, 3);
        mat3.push(2, 0, 2);
        mat3.push(3, 1, 4);
        mat3.push(4, 2, 6);
        let b = CsrMatrix::from(&mat3);
        let a = csr_vstack(&[&CsrMatrix::from(&mat1), &CsrMatrix::from(&mat2)]);

        println!("a={:?}", CsrMatrix::from(&mat1).disassemble());
        println!("b={:?}", CsrMatrix::from(&mat2).disassemble());
        println!("vstack(a,b)={:?}", a.clone().disassemble());
        println!(
            "vstack(a,b) should be = {:?}",
            CsrMatrix::from(&mat3).disassemble()
        );
        assert!(a == b, "matrices do not match!")
    }

    #[test]
    fn test_csc_vstack() {
        // 创建第一个稀疏矩阵
        let mut mat1 = CooMatrix::new(2, 3);
        mat1.push(1, 2, 3);

        // 创建第二个稀疏矩阵
        let mut mat2 = CooMatrix::new(3, 3);
        mat2.push(0, 0, 2);
        mat2.push(1, 1, 4);
        mat2.push(2, 2, 6);

        let mut mat3 = CooMatrix::new(5, 3);
        mat3.push(1, 2, 3);
        mat3.push(2, 0, 2);
        mat3.push(3, 1, 4);
        mat3.push(4, 2, 6);
        let b = CscMatrix::from(&mat3);
        let a = csc_vstack(&[&CscMatrix::from(&mat1), &CscMatrix::from(&mat2)]);

        println!("a={:?}", CscMatrix::from(&mat1).disassemble());
        println!("b={:?}", CscMatrix::from(&mat2).disassemble());
        println!("vstack(a,b)={:?}", a.clone().disassemble());
        println!(
            "vstack(a,b) should be = {:?}",
            CscMatrix::from(&mat3).disassemble()
        );
        assert!(a == b, "matrices do not match!")
    }

    #[test]
    fn test_vstack() {
        // 创建第一个稀疏矩阵
        let mut mat1 = CooMatrix::new(2, 3);
        mat1.push(1, 2, 3);

        // 创建第二个稀疏矩阵
        let mut mat2 = CooMatrix::new(3, 3);
        mat2.push(0, 0, 2);
        mat2.push(1, 1, 4);
        mat2.push(2, 2, 6);

        let mut mat3 = CooMatrix::new(5, 3);
        mat3.push(1, 2, 3);
        mat3.push(2, 0, 2);
        mat3.push(3, 1, 4);
        mat3.push(4, 2, 6);
        let b = CscMatrix::from(&mat3);
        let a: CscMatrix<_> = vstack(&[
            &SparseMatrix::from(&CscMatrix::from(&mat1)),
            &SparseMatrix::from(&CsrMatrix::from(&mat2)),
        ]);
        let aa: CscMatrix<_> = vstack(&[
            &SparseMatrix::from(&CsrMatrix::from(&mat1)),
            &SparseMatrix::from(&CscMatrix::from(&mat2)),
        ]);
        let aaa: CscMatrix<_> = vstack(&[
            &SparseMatrix::from(&CscMatrix::from(&mat1)),
            &SparseMatrix::from(&CscMatrix::from(&mat2)),
        ]);

        println!("a={:?}", CscMatrix::from(&mat1).disassemble());
        println!("b={:?}", CscMatrix::from(&mat2).disassemble());
        println!("vstack(a,b)={:?}", a.clone().disassemble());
        println!(
            "vstack(a,b) should be = {:?}",
            CscMatrix::from(&mat3).disassemble()
        );
        assert!(a == b, "matrices do not match!");
        assert!(aa == b, "matrices do not match!");
        assert!(aaa == b, "matrices do not match!")
    }
}
