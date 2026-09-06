//! 仅在 benchmark 功能启用时，向独立性能程序开放内部基线。
pub use super::dsbus_dv::{dSbus_dV, dSbus_dV_old};
pub use super::new_dsdvbus2::{JacobianPattern2, fill_jacobian_v2};
pub use super::new_dsdvbus3::fill_jacobian_v3;
pub use super::new_dsdvbus4::{fill_j_and_jt_exp, fill_jacobian_v4};
pub use super::pf_old_impl::{
    JacobianCache, build_jacobian, build_jacobian_cached, newton_pf_old, newton_pf_v0,
};

#[inline(always)]
pub fn csc_matvec_and_scalc(
    cp: &[usize],
    ri: &[usize],
    y: &[num_complex::Complex64],
    v: &[num_complex::Complex64],
    i: &mut [num_complex::Complex64],
    s: &mut [num_complex::Complex64],
) {
    super::newtonpf::csc_matvec_and_scalc(cp, ri, y, v, i, s)
}
