#[derive(Debug, Clone)]
pub struct Control {
    /// "dense" if degree > dense * sqrt(n)
    pub dense: f64,
    /// Do aggressive absorption.
    pub aggressive: bool,
}

impl Default for Control {
    fn default() -> Self {
        Self {
            dense: 10.0,
            aggressive: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Info {
    /// Return value of order and l_order.
    pub status: Status,
    /// A is n-by-n.
    pub n: usize,
    /// Number of nonzeros in A.
    pub nz: usize,
    /// Symmetry of pattern (true is sym., false is unsym.)
    pub symmetry: bool,
    /// Number of entries on diagonal.
    pub nz_diag: usize,
    /// nz in A+A'.
    pub nz_a_plus_at: usize,
    /// Number of "dense" rows/columns in A.
    pub n_dense: usize,
    /// Number of garbage collections in AMD.
    pub n_cmp_a: usize,
    /// Approx. nz in L, excluding the diagonal.
    pub lnz: usize,
    /// Number of fl. point divides for LU and LDL'.
    pub n_div: usize,
    /// Number of fl. point (*,-) pairs for LDL'.
    pub n_mult_subs_ldl: usize,
    /// Number of fl. point (*,-) pairs for LU.
    pub n_mult_subs_lu: usize,
    /// Max nz. in any column of L, incl. diagonal.
    pub d_max: usize,
}

impl Info {
    pub fn new(n: usize) -> Info {
        Info {
            status: Status::OK,
            n,
            nz: 0,
            symmetry: false,
            nz_diag: 0,
            nz_a_plus_at: 0,
            n_dense: 0,
            n_cmp_a: 0,
            lnz: 0,
            n_div: 0,
            n_mult_subs_ldl: 0,
            n_mult_subs_lu: 0,
            d_max: 0,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Status {
    OK,
    /// Input arguments are not valid.
    Invalid,
    /// Input matrix is OK for order, but columns were not sorted, and/or
    /// duplicate entries were present. AMD had to do extra work before
    /// ordering the matrix. This is a warning, not an error.
    OkButJumbled,
}
