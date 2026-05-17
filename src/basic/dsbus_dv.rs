use crate::basic::sparse::conj::Conjugate;
use nalgebra::*;
use nalgebra_sparse::CscMatrix;
/// Computes the Jacobian matrices of power injections with respect to voltage magnitudes and angles.
///
/// This function calculates the Jacobian matrices of power injections with respect to voltage magnitudes and angles,
/// given the admittance matrix `Ybus`, the complex voltage vector `v`, and the normalized complex voltage vector `Vnorm`.
///
/// # Arguments
///
/// * `Ybus` - A nalgebra_sparse CSC (Compressed Sparse Column) matrix representing the nodal admittance matrix.
/// * `v` - A dense vector of complex numbers representing the voltage phasors.
/// * `Vnorm` - A dense vector of complex numbers representing the normalized voltage phasors.
///
/// # Returns
///
/// A tuple `(dS_dVm, dS_dVa)` containing the Jacobian matrices:
///
/// * `dS_dVm` - The Jacobian matrix of power injections with respect to voltage magnitudes.
/// * `dS_dVa` - The Jacobian matrix of power injections with respect to voltage angles.
///
/// # Notes
///
/// * This function assumes that `Ybus`, `v`, and `Vnorm` have compatible dimensions.
/// * The Jacobian matrices are computed using the formulae for power injections in a power system.
/// * This method is from MatPower:
///  R. D. Zimmerman, "AC Power Flows, Generalized OPF Costs and
///  their Derivatives using Complex Matrix Notation", MATPOWER
///  Technical Note 2, February 2010.U{http://www.pserc.cornell.edu/matpower/TN2-OPF-Derivatives.pdf}
///  @author: Ray Zimmerman (PSERC Cornell)
///
#[allow(non_snake_case)]
pub fn dSbus_dV_old(
    Ybus: &CscMatrix<Complex<f64>>,
    v: &DVector<Complex<f64>>,
    Vnorm: &DVector<Complex<f64>>,
) -> (CscMatrix<Complex<f64>>, CscMatrix<Complex<f64>>) {
    // let dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV);
    let diagpattern = CscMatrix::identity(v.len());
    let ibus = Ybus * v;
    let mut diagVnorm = diagpattern.clone();
    let mut diagV = diagpattern.clone();
    let mut diagIbus = diagpattern.clone();
    diagVnorm.values_mut().copy_from_slice(Vnorm.as_slice());
    diagV.values_mut().copy_from_slice(v.as_slice());
    diagIbus.values_mut().copy_from_slice(ibus.as_slice());

    let dS_dVm = &diagV * (Ybus * &diagVnorm).conjugate() + diagIbus.conjugate() * &diagVnorm;
    let dS_dVa = &diagV * (diagIbus - Ybus * &diagV).conjugate() * Complex::<f64>::i();
    (dS_dVm, dS_dVa)
}
#[allow(non_snake_case)]
pub fn dSbus_dV(
    Ybus: &CscMatrix<Complex<f64>>,
    v: &DVector<Complex<f64>>,
    Vnorm: &DVector<Complex<f64>>,
) -> (CscMatrix<Complex<f64>>, CscMatrix<Complex<f64>>) {
    let n = Ybus.ncols();

    // 唯一允许的矩阵运算：SpMV (稀疏矩阵乘稠密向量)，这在底层非常快
    let ibus = Ybus * v;

    // 获取 Ybus 最底层的连续内存视图 (Views)
    let col_offsets = Ybus.col_offsets();
    let row_indices = Ybus.row_indices();
    let y_vals = Ybus.values();

    let nnz = y_vals.len();

    // 直接一次性分配好两个输出矩阵的值数组，不碰结构
    let mut dsm_vals = vec![Complex::new(0.0, 0.0); nnz];
    let mut dsa_vals = vec![Complex::new(0.0, 0.0); nnz];

    let v_slice = v.as_slice();
    let vnorm_slice = Vnorm.as_slice();
    let ibus_slice = ibus.as_slice();

    let j_unit = Complex::new(0.0, 1.0);

    // 核心：单趟 O(NNZ) 连续内存遍历，打爆缓存命中率
    for j in 0..n {
        let start = col_offsets[j] as usize;
        let end = col_offsets[j + 1] as usize;

        for idx in start..end {
            let i = row_indices[idx] as usize;
            let y_ij = y_vals[idx];

            // MatPower 元素级硬解推导：
            // 对于非对角线元素 (i != j)，diagIbus 为 0。
            // 对于对角线元素 (i == j)，加入 diagIbus 的影响。
            if i == j {
                dsm_vals[idx] = v_slice[i] * (y_ij * vnorm_slice[i]).conj()
                    + ibus_slice[i].conj() * vnorm_slice[i];
                dsa_vals[idx] = j_unit * v_slice[i] * (ibus_slice[i] - y_ij * v_slice[i]).conj();
            } else {
                dsm_vals[idx] = v_slice[i] * (y_ij * vnorm_slice[j]).conj();
                dsa_vals[idx] = j_unit * v_slice[i] * (-y_ij * v_slice[j]).conj();
            }
        }
    }

    // 直接复用 Ybus 的结构指针暴力构造！零结构运算！
    // 注：由于 Ybus.col_offsets() 是 &[usize]，转换为 CSR/CSC 数据结构可能需要 vec 复制，
    // 但相比之前几十次的 SpGEMM 内部重新猜测列宽，这点克隆开销完全可以忽略不计。
    let dS_dVm =
        CscMatrix::try_from_csc_data(n, n, col_offsets.to_vec(), row_indices.to_vec(), dsm_vals)
            .unwrap();

    let dS_dVa =
        CscMatrix::try_from_csc_data(n, n, col_offsets.to_vec(), row_indices.to_vec(), dsa_vals)
            .unwrap();

    (dS_dVm, dS_dVa)
}
