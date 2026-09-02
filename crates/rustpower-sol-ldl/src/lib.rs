//! Unsafe-ish wrapper over SuiteSparse LDL (int32) with AMD ordering.
//!
//! Target matrices: symmetric **quasi-definite** systems (e.g. the LM
//! augmented KKT `[μI Jᵀ; J −I]`), for which a no-pivot LDLᵀ factorization
//! exists and is stable for *any* fill-reducing permutation (Vanderbei 1995).
//!
//! Input convention (LDL package): only the **upper triangular part** of A,
//! stored in column-compressed form with ascending row indices per column.
//!
//! Lifecycle mirrors our KLU wrapper: `analyze` once per sparsity pattern
//! (AMD ordering + symbolic factorization), then `factor` + `solve_in_place`
//! per right-hand side. Values-only changes reuse the symbolic phase.

use rustpower_ldl_sys::*;

pub struct LDLSolver {
    pub n: i32,
    /// AMD fill-reducing permutation and its inverse.
    pub p: Vec<i32>,
    pub pinv: Vec<i32>,
    /// Symbolic factorization outputs.
    pub lp: Vec<i32>,
    pub parent: Vec<i32>,
    pub lnz: Vec<i32>,
    pub flag: Vec<i32>,
    /// Numeric factors (allocated after `analyze`).
    pub li: Vec<i32>,
    pub lx: Vec<f64>,
    pub d: Vec<f64>,
    /// Workspaces for ldl_numeric / triangular solves.
    pub ywork: Vec<f64>,
    pub pattern: Vec<i32>,
    pub xwork: Vec<f64>,
    pub analyzed: bool,
}

impl Default for LDLSolver {
    fn default() -> Self {
        Self {
            n: 0,
            p: Vec::new(),
            pinv: Vec::new(),
            lp: Vec::new(),
            parent: Vec::new(),
            lnz: Vec::new(),
            flag: Vec::new(),
            li: Vec::new(),
            lx: Vec::new(),
            d: Vec::new(),
            ywork: Vec::new(),
            pattern: Vec::new(),
            xwork: Vec::new(),
            analyzed: false,
        }
    }
}

impl LDLSolver {
    /// AMD ordering + symbolic LDLᵀ on the upper-triangular CSC pattern.
    /// AMD ordering + symbolic LDLᵀ on the CSC pattern.
    /// With a permutation, LDL accesses the upper triangle of **PAP′**, so
    /// `ap`/`ai` must contain the FULL symmetric pattern (entries ignored
    /// when they fall below the permuted diagonal; duplicates are summed).
    /// Returns 0 on success.
    pub unsafe fn analyze(&mut self, n: i32, ap: &[i32], ai: &[i32]) -> i32 {
        self.n = n;
        let n_us = n as usize;
        self.p = vec![0; n_us];
        self.pinv = vec![0; n_us];
        self.lp = vec![0; n_us + 1];
        self.parent = vec![0; n_us];
        self.lnz = vec![0; n_us];
        self.flag = vec![0; n_us];

        // Fill-reducing ordering (AMD works on the graph of A+Aᵀ; a full
        // symmetric pattern is exactly that).
        let status = amd_order(
            n,
            ap.as_ptr(),
            ai.as_ptr(),
            self.p.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        // AMD_OK = 0, AMD_OK_BUT_JUMBLED = 1 are both usable.
        if status != AMD_OK as i32 && status != AMD_OK_BUT_JUMBLED as i32 {
            return status;
        }
        // NOTE: ldl_symbolic itself overwrites Pinv from P (Pinv is an
        // output); no need to compute it here.

        ldl_symbolic(
            n,
            ap.as_ptr() as *mut i32,
            ai.as_ptr() as *mut i32,
            self.lp.as_mut_ptr(),
            self.parent.as_mut_ptr(),
            self.lnz.as_mut_ptr(),
            self.flag.as_mut_ptr(),
            self.p.as_mut_ptr(),
            self.pinv.as_mut_ptr(),
        );

        let l_nnz = self.lp[n_us] as usize;
        self.li = vec![0; l_nnz];
        self.lx = vec![0.0; l_nnz];
        self.d = vec![0.0; n_us];
        self.ywork = vec![0.0; n_us];
        self.pattern = vec![0; n_us];
        self.xwork = vec![0.0; n_us];
        self.analyzed = true;
        0
    }

    /// Numeric LDLᵀ factorization; pattern and permutation from `analyze`.
    /// `ldl_numeric` returns `n` on success, or the index `k < n` of a zero
    /// pivot (must not happen for a quasi-definite matrix with μ > 0).
    /// Returns 0 on success, -1 on a zero pivot.
    pub unsafe fn factor(&mut self, ap: &[i32], ai: &[i32], ax: &[f64]) -> i32 {
        let ret = ldl_numeric(
            self.n,
            ap.as_ptr() as *mut i32,
            ai.as_ptr() as *mut i32,
            ax.as_ptr() as *mut f64,
            self.lp.as_mut_ptr(),
            self.parent.as_mut_ptr(),
            self.lnz.as_mut_ptr(),
            self.li.as_mut_ptr(),
            self.lx.as_mut_ptr(),
            self.d.as_mut_ptr(),
            self.ywork.as_mut_ptr(),
            self.pattern.as_mut_ptr(),
            self.flag.as_mut_ptr(),
            self.p.as_mut_ptr(),
            self.pinv.as_mut_ptr(),
        );
        if ret == self.n { 0 } else { -1 }
    }

    /// Solves A x = b in place on `b`, using the current factors.
    /// Sequence: x = P b → L⁻¹ → D⁻¹ → L⁻ᵀ → b = Pᵀ x.
    pub unsafe fn solve_in_place(&mut self, b: &mut [f64]) {
        let n = self.n;
        ldl_perm(n, self.xwork.as_mut_ptr(), b.as_mut_ptr(), self.p.as_mut_ptr());
        ldl_lsolve(
            n,
            self.xwork.as_mut_ptr(),
            self.lp.as_mut_ptr(),
            self.li.as_mut_ptr(),
            self.lx.as_mut_ptr(),
        );
        ldl_dsolve(n, self.xwork.as_mut_ptr(), self.d.as_mut_ptr());
        ldl_ltsolve(
            n,
            self.xwork.as_mut_ptr(),
            self.lp.as_mut_ptr(),
            self.li.as_mut_ptr(),
            self.lx.as_mut_ptr(),
        );
        ldl_permt(n, b.as_mut_ptr(), self.xwork.as_mut_ptr(), self.p.as_mut_ptr());
    }

    /// Inertia of the factorized matrix: (positive, negative) pivots of D.
    /// For the LM augmented system this must be exactly (n_δ, n_residual).
    pub fn inertia(&self) -> (usize, usize) {
        let pos = self.d.iter().filter(|&&x| x > 0.0).count();
        (pos, self.d.len() - pos)
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

unsafe impl Send for LDLSolver {}
unsafe impl Sync for LDLSolver {}

#[test]
fn drop_test() {
    let ldl = LDLSolver::default();
    drop(ldl);
}
