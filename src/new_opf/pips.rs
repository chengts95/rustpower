pub use crate::opf::pips::{PipsOpt, PipsResult};
use super::problem::NewOPFData;
use super::numeric::numeric_fill;
use crate::opf::cost;
use crate::opf::constraints;

use crate::new_opf::v3_symbolic::V3SymbolicCache;
use crate::new_opf::v3_numeric::v3_numeric_fill;

use crate::new_opf::v3_numeric_scalar::v3_scalar_numeric_fill;

/// Optimized PIPS solver using V3 Revolutionary Symbolic-Cached Assembly and Persistent KLU.
pub fn pips(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    
    // Preliminary KKT Symbolic Analysis for V5 Shadow Check
    let kkt_cache = crate::new_opf::v3_symbolic::KKTSymbolicCache::analyze(&v3_cache, data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::opf::pips::pips_with_solver(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
            let lxx = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
                data, &v3_cache, x, lam_eq, mu_ineq, Some(z_ineq), cost_mult
            );
            
            // --- V5 SHADOW CHECK ---
            let mut test_kkt_vals = vec![0.0f64; kkt_cache.kkt_skeleton.nnz()];
            for (idx, &val) in lxx.values().iter().enumerate() {
                test_kkt_vals[kkt_cache.lxx_to_kkt[idx]] = val;
            }
            
            lxx
        },
        x0, xmin, xmax, opt, &mut persistent_solver
    )
}
