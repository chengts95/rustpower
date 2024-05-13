#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use klu_sys::*;
use std::alloc::{alloc, Layout};
pub struct KLUSolver {
    pub common: *mut klu_l_common,
    pub symbolic: *mut klu_l_symbolic,
    pub numeric: *mut klu_l_numeric,
}

impl Default for KLUSolver {
    fn default() -> Self {
        unsafe {
            let tmp = KLUSolver {
                common: alloc(Layout::for_value(&klu_l_common::default())) as *mut klu_l_common,
                symbolic: std::ptr::null_mut() as *mut klu_l_symbolic,
                numeric: std::ptr::null_mut() as *mut klu_l_numeric,
            };

            klu_l_defaults(tmp.common);
            tmp
        }
    }
}
impl Drop for KLUSolver {
    fn drop(&mut self) {
        unsafe {
            klu_l_free_symbolic(&mut self.symbolic as *mut *mut klu_l_symbolic, self.common);

            klu_l_free_numeric(&mut self.numeric as *mut *mut klu_l_numeric, self.common);
        };
    }
}
impl KLUSolver {
    pub unsafe fn solve_sym(&mut self, Ap: *mut i64, Ai: *mut i64, n: i64) -> i64 {
        if !self.symbolic.is_null() {
            klu_l_free_symbolic(&mut self.symbolic as *mut *mut klu_l_symbolic, self.common);
        }
        self.symbolic = klu_l_analyze(n, Ap, Ai, self.common);
        (*self.common).status.into()
    }
    pub unsafe fn factor(&mut self, Ap: *mut i64, Ai: *mut i64, Ax: *mut f64) -> i64 {
        if !self.numeric.is_null() {
            klu_l_free_numeric(&mut self.numeric as *mut *mut klu_l_numeric, self.common);
        }
        self.numeric = klu_l_factor(Ap, Ai, Ax, self.symbolic, self.common);
        (*self.common).status.into()
    }

    pub unsafe fn refactor(&mut self, Ap: *mut i64, Ai: *mut i64, Ax: *mut f64, _n: i64) -> i64 {
        klu_l_refactor(Ap, Ai, Ax, self.symbolic, self.numeric, self.common);
        (*self.common).status.into()
    }

    pub unsafe fn solve(&mut self, b: *mut f64, n: i64, bn: i64) -> i64 {
        klu_l_solve(self.symbolic, self.numeric, n, bn, b, self.common);
        (*self.common).status.into()
    }
    pub fn reset(&mut self) {
        unsafe {
            klu_l_free_symbolic(&mut self.symbolic as *mut *mut klu_l_symbolic, self.common);

            klu_l_free_numeric(&mut self.numeric as *mut *mut klu_l_numeric, self.common);

            *self.common = klu_l_common::default();
            self.symbolic = std::ptr::null_mut();
            self.numeric = std::ptr::null_mut();

            klu_l_defaults(self.common);
        }
    }
}
#[test]
fn drop_test() {
    let klu = KLUSolver::default();
    drop(klu);
}
#[test]
fn reset_test() {
    let mut klu = KLUSolver::default();
    klu.reset();
}
unsafe impl Send for KLUSolver {}
unsafe impl Sync for KLUSolver {}
