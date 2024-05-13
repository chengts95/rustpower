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
pub fn dSbus_dV(
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
