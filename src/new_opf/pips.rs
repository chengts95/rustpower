pub use crate::opf::pips::{PipsOpt, PipsResult};
use super::problem::NewOPFData;
use super::numeric::numeric_fill;
use crate::opf::cost;
use crate::opf::constraints;

use crate::new_opf::v3_symbolic::V3SymbolicCache;
use crate::new_opf::v3_numeric::v3_numeric_fill;

/// Optimized PIPS solver using V3 Revolutionary Symbolic-Cached Assembly.
pub fn pips(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);

    crate::opf::pips::pips(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, cost_mult| {
            v3_numeric_fill(data, &v3_cache, x, lam_eq, mu_ineq, cost_mult)
        },
        x0, xmin, xmax, opt
    )
}
