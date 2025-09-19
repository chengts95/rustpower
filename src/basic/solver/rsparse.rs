use rsparse::{
    self,
    data::{self, Numeric, Symb},
    lsolve, lu, sqr, usolve,
};

use super::Solve;

#[derive(Default)]
pub struct RSparseSolver {
    x: Option<Vec<f64>>,
    symbolic: Option<Symb>,
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
    
    fn reset(& mut self) {
        self.symbolic = None;
    }
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
