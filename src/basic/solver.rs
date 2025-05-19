use rsparse::{
    self,
    data::{self, Numeric, Symb},
    lsolve, lu, sqr, usolve,
};

#[cfg(feature = "klu")]
use rustpower_sol_klu as klu_rs;

#[cfg(feature = "klu")]
#[derive(Default)]
pub struct KLUSolver(pub klu_rs::KLUSolver);

#[derive(Default)]
pub struct RSparseSolver {
    x: Option<Vec<f64>>,
    symbolic: Option<Symb>,
}

#[allow(non_snake_case)]
/// A trait for solving sparse linear systems.
pub trait Solve {
    /// Solves the sparse linear system.
    ///
    /// # Parameters
    ///
    /// * `Ap` - Column pointers of the matrix.
    /// * `Ai` - Row indices of the matrix.
    /// * `Ax` - Non-zero values of the matrix.
    /// * `_b` - Right-hand side vector.
    /// * `_n` - Dimension of the system.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
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
    /// Solves the sparse linear system using the KLU solver.
    ///
    /// # Parameters
    ///
    /// * `Ap` - Column pointers of the matrix.
    /// * `Ai` - Row indices of the matrix.
    /// * `Ax` - Non-zero values of the matrix.
    /// * `b` - Right-hand side vector.
    /// * `n` - Dimension of the system.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
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
                Ax.as_mut_ptr(),
            );
            ret |= self.0.solve(b.as_mut_ptr(), n as i64, 1);
            if ret != 0 {
                return Err("error occurred when calling KLU routines!");
            }
        }
        Ok(())
    }
}

#[cfg(feature = "klu")]
#[test]
/// Tests the drop functionality of the KLU solver.
fn drop_test() {
    let klu = KLUSolver::default();
    drop(klu);
}

#[cfg(feature = "klu")]
#[test]
/// Tests the reset functionality of the KLU solver.
fn reset_test() {
    let mut klu = KLUSolver::default();
    klu.0.reset();
}

fn ipvec_identity<T: Numeric<T>>(b: &[T], x: &mut [T]) {
    x.copy_from_slice(b);
}

fn ipvec_perm<T: Numeric<T>>(p: &[isize], b: &[T], x: &mut [T]) {
    for k in 0..b.len() {
        x[p[k] as usize] = b[k];
    }
}

fn ipvec<T: Numeric<T>>(p: &Option<Vec<isize>>, b: &[T], x: &mut [T]) {
    match p {
        Some(pvec) => ipvec_perm(pvec, b, x),
        None => ipvec_identity(b, x),
    }
}
#[allow(non_snake_case)]
impl Solve for RSparseSolver {
    #[allow(unused)]
    /// Solves the sparse linear system using the RSparse solver.
    ///
    /// # Parameters
    ///
    /// * `Ap` - Column pointers of the matrix.
    /// * `Ai` - Row indices of the matrix.
    /// * `Ax` - Non-zero values of the matrix.
    /// * `b` - Right-hand side vector.
    /// * `n` - Dimension of the system.
    ///
    /// # Returns
    ///
    /// A result indicating success or failure.
    fn solve(
        &mut self,
        Ap: &mut [usize],
        Ai: &mut [usize],
        Ax: &mut [f64],
        b: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        let n = Ap.len() - 1;
        let p: Vec<isize> = Ap.iter().map(|&v| v as isize).collect();
        let mut a = data::Sprs {
            m: n,
            n: n,
            i: Ai.to_vec(),
            p,
            x: Ax.to_vec(),
            nzmax: Ax.len(),
        };
        if self.symbolic.is_none() {
            self.symbolic = Some(sqr(&a, 1, false));
            self.x = Some(vec![0.0; n]);
        }
        let mut x = self.x.as_mut().unwrap();
        let mut s = self.symbolic.as_mut().unwrap();
        let n = lu(&a, &mut s, 1e-6).map_err(|_| "LU factorization failed")?; // numeric LU factorization
        ipvec(&n.pinv, b, &mut x[..]); // x = P*b
        lsolve(&n.l, &mut x); // x = L\x
        usolve(&n.u, &mut x); // x = U\x
        ipvec(&s.q, &x, &mut b[..]); // b = Q*x

        Ok(())
    }
}
