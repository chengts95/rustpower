pub use crate::opf::pips::{PipsOpt, PipsResult};
use super::problem::NewOPFData;
use crate::opf::cost;
use crate::opf::constraints;
use crate::new_opf::v3_symbolic::V3SymbolicCache;

/// Optimized PIPS solver using V3/V4 Revolutionary Scalar Assembly and Persistent KLU.
pub fn pips(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::opf::pips::pips_with_solver(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
            // V4 (Rectangular Rotate + Merged Slacks Penalty)
            crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
                data, &v3_cache, x, lam_eq, mu_ineq, Some(z_ineq), cost_mult
            )
        },
        x0, xmin, xmax, 
        PipsOpt { merged_slacks: true, ..opt }, 
        &mut persistent_solver
    )
}
