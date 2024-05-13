use rsparse::{self, data, lusol};

#[cfg(feature = "klu")]
#[derive(Default)]
pub struct KLUSolver(pub klu_rs::KLUSolver);

#[derive(Default)]
pub struct RSparseSolver;

#[allow(non_snake_case)]
pub trait Solve {
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        _b: &mut [f64],
        _n: usize,
    ) -> Result<(), &'static str>;
}

#[cfg(feature = "klu")]
#[allow(non_snake_case)]
impl Solve for KLUSolver {
    #[allow(unused)]
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        unsafe {
            let mut ret = self.0.solve_sym(
                Ap.as_mut_ptr() as *mut i64,
                Ai.as_mut_ptr() as *mut i64,
                n as i64,
            );
            ret |= self.0.factor(
                Ap.as_mut_ptr() as *mut i64,
                Ai.as_mut_ptr() as *mut i64,
                Ax.as_mut_ptr()
            );
            ret |= self.0.solve(b.as_mut_ptr(), n as i64, 1);
            if ret != 0 {
                return Err("error occured when calling klu routines!");
            }
        }
        Ok(())
    }
}
#[cfg(feature = "klu")]
#[test]
fn drop_test() {
    let klu = KLUSolver::default();
    drop(klu);
}
#[cfg(feature = "klu")]
#[test]
fn reset_test() {
    let mut klu = KLUSolver::default();
    klu.0.reset();
}
#[allow(non_snake_case)]
impl Solve for RSparseSolver {
    #[allow(unused)]
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        let mut mat: data::Sprs = rsparse::data::Sprs::zeros(Ap.len() - 1, Ap.len() - 1, Ai.len());

        let p = unsafe { std::slice::from_raw_parts_mut(Ap.as_mut_ptr() as *mut isize, Ap.len()) };
        unsafe {
            mat.i.swap_with_slice(Ai);
            mat.p.swap_with_slice(p);
            mat.x.swap_with_slice(Ax);
        }

        lusol(&mat, b, 1, 1e-6);
        Ok(())
    }
}
