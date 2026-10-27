//! Rust seeded AMD with the same proven one-block KLU layout as the C adapter.
use crate::{
    Control,
    powerflow::{OrderingWorkspace, YbusPattern},
};
use rustpower_klu_sys::*;
use std::{mem::size_of, ptr};

unsafe fn allocate<T>(len: usize, common: *mut klu_l_common) -> *mut T {
    klu_l_malloc(len, size_of::<T>(), common).cast()
}

/// The pattern validator establishes sorted, symmetric active adjacency, all
/// diagonals, and one connected component. The scalar J therefore has one BTF
/// block and a full structural matching. KLU owns every returned allocation.
pub(crate) unsafe fn analyze(
    p: &[usize],
    rows: &[usize],
    npv: usize,
    npq: usize,
    common: *mut klu_l_common,
) -> *mut klu_l_symbolic {
    let Ok(pattern) = YbusPattern::analyze(p, rows, npv, npq) else {
        return ptr::null_mut();
    };
    let Ok(ordering) = OrderingWorkspace::new().order(&pattern, &Control::default()) else {
        return ptr::null_mut();
    };
    let n = pattern.scalar_dimension();
    let info = ordering.info();
    (*common).status = KLU_OK as i32;
    (*common).work = 0.0;
    let mut symbolic = allocate::<klu_l_symbolic>(1, common);
    if symbolic.is_null() {
        return symbolic;
    }
    symbolic.write(std::mem::zeroed());
    let s = &mut *symbolic;
    s.n = n as i64;
    s.nz = pattern.scalar_nnz() as i64;
    s.P = allocate(n, common);
    s.Q = allocate(n, common);
    s.R = allocate(n + 1, common);
    s.Lnz = allocate(n, common);
    if s.P.is_null() || s.Q.is_null() || s.R.is_null() || s.Lnz.is_null() {
        klu_l_free_symbolic(&mut symbolic, common);
        return ptr::null_mut();
    }
    for (k, &variable) in ordering.permutation().iter().enumerate() {
        *s.P.add(k) = variable as i64;
        *s.Q.add(k) = variable as i64;
    }
    s.ordering = 0;
    s.do_btf = i64::from((*common).btf != 0);
    s.structural_rank = if s.do_btf != 0 { n as i64 } else { -1 };
    (*common).structural_rank = s.structural_rank;
    s.nblocks = 1;
    s.maxblock = n as i64;
    *s.R = 0;
    *s.R.add(1) = n as i64;
    s.nzoff = 0;
    s.lnz = (info.lnz + n) as f64;
    s.unz = s.lnz;
    *s.Lnz = s.lnz;
    s.est_flops = 2.0 * info.n_mult_subs_lu as f64 + info.n_div as f64;
    s.symmetry = 1.0;
    symbolic
}
