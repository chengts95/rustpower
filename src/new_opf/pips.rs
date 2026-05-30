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
        &mut persistent_solver,
        None,
    )
}

/// V5.0 path: identical numerics to V4, but the KKT is assembled by the V5 symbolic
/// streaming fill (single advancing pointer) instead of per-iteration build_saddle_point.
/// Step node kept alongside `pips` (V4) for direct A/B comparison.
pub fn pips_v5(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let v5_cache = crate::new_opf::v5_kkt::KKTSymbolicV5::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::opf::pips::pips_with_solver(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
            crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
                data, &v3_cache, x, lam_eq, mu_ineq, Some(z_ineq), cost_mult
            )
        },
        x0, xmin, xmax,
        PipsOpt { merged_slacks: true, ..opt },
        &mut persistent_solver,
        Some(&v5_cache),
    )
}

/// V5.2 path: Fused Block-Operator assembly (Kernel V5.2).
/// No intermediate matrices (no Lxx, no Jacobian matrices).
/// KKT values are calculated inline and streamed into the values array.
pub fn pips_v5_2(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let v5_cache = crate::new_opf::v5_kkt::KKTSymbolicV5::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::opf::pips::pips_with_fused_assembly(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use super::v5_2_kernel::*;
            kkt_vals.fill(0.0);
            fill_variable_columns(&v5_cache, data, &v3_cache.y_transpose_idx, x, lam_eq, cost_mult, kkt_vals);
            fill_constraint_columns(&v5_cache, data, &v3_cache.y_transpose_idx, &v5_cache.gens_at_bus, x, kkt_vals);
            fill_branch_hessian(&v5_cache, data, x, mu_ineq, z_ineq, kkt_vals);
        },
        x0, xmin, xmax,
        PipsOpt { merged_slacks: true, ..opt },
        &mut persistent_solver,
        &v5_cache,
    )
}

/// V5.3 path: Partitioned Isomorphic assembly (Kernel V5.3).
/// Fully eliminates global scatter by gathering branch contributions into column slices.
/// Foundation for future parallel assembly.
pub fn pips_v5_3(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let v53_cache = crate::new_opf::v5_3_kernel::KKTSymbolicV5_3::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::opf::pips::pips_with_fused_assembly(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use super::v5_3_kernel::*;
            assemble_kkt_v5_3(&v53_cache, data, &v3_cache.y_transpose_idx, x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals);
        },
        x0, xmin, xmax,
        PipsOpt { merged_slacks: true, ..opt },
        &mut persistent_solver,
        &v53_cache.base,
    )
}
