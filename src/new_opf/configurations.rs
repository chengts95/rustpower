//! Named combinations of assembly and evaluation, preserving historical experiments.
//! V4 remains the legacy `pips()` entry point; new explicit selection defaults to V5.6.
use crate::new_opf::assembly::v3::symbolic::V3SymbolicCache;
use crate::new_opf::evaluation::v1 as constraints;
use crate::new_opf::model::NewOPFData;
pub use crate::new_opf::solution::{PipsOpt, PipsResult};
use crate::opf::cost;

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

    crate::new_opf::interior_point::pips_with_solver(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
            // V4 (Rectangular Rotate + Merged Slacks Penalty)
            crate::new_opf::assembly::v4::curvature::v4_rect_numeric_fill(
                data,
                &v3_cache,
                x,
                lam_eq,
                mu_ineq,
                Some(z_ineq),
                cost_mult,
            )
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
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
    let v5_cache = crate::new_opf::assembly::v5::symbolic::KKTSymbolicV5::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::new_opf::interior_point::pips_with_solver(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
            crate::new_opf::assembly::v4::curvature::v4_rect_numeric_fill(
                data,
                &v3_cache,
                x,
                lam_eq,
                mu_ineq,
                Some(z_ineq),
                cost_mult,
            )
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
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
    let v5_cache = crate::new_opf::assembly::v5::symbolic::KKTSymbolicV5::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::new_opf::interior_point::pips_with_fused_assembly(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use crate::new_opf::assembly::v5::scatter::*;
            kkt_vals.fill(0.0);
            fill_variable_columns(
                &v5_cache,
                data,
                &v3_cache.y_transpose_idx,
                x,
                lam_eq,
                cost_mult,
                kkt_vals,
            );
            fill_constraint_columns(
                &v5_cache,
                data,
                &v3_cache.y_transpose_idx,
                &v5_cache.gens_at_bus,
                x,
                kkt_vals,
            );
            fill_branch_hessian(&v5_cache, data, x, mu_ineq, z_ineq, kkt_vals);
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
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
    let v53_cache = crate::new_opf::assembly::v5::partitioned::KKTSymbolicV5_3::build(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::new_opf::interior_point::pips_with_fused_assembly(
        |x| cost::opf_costfcn(data, x),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use crate::new_opf::assembly::v5::partitioned::*;
            assemble_kkt_v5_3(
                &v53_cache,
                data,
                &v3_cache.y_transpose_idx,
                x,
                lam_eq,
                mu_ineq,
                z_ineq,
                cost_mult,
                kkt_vals,
            );
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
        &mut persistent_solver,
        &v53_cache.base,
    )
}

/// V5.6 path: V5.3 KKT assembly + V5.6 G/H direct-fill with linear bound constraints
/// baked into a static dg/dh structure. No merge_constraints/transpose/hstack in the loop.
pub fn pips_v5_6(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let v53_cache = crate::new_opf::assembly::v5::partitioned::KKTSymbolicV5_3::build(data);
    let ev = crate::new_opf::evaluation::v5::merged::V56Evaluator::new(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::new_opf::interior_point::pips_with_fused_assembly_v56(
        |x| cost::opf_costfcn(data, x),
        |x, g, h, dg_v, dh_v| ev.update(data, x, g, h, dg_v, dh_v),
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use crate::new_opf::assembly::v5::partitioned::*;
            assemble_kkt_v5_3(
                &v53_cache,
                data,
                &v3_cache.y_transpose_idx,
                x,
                lam_eq,
                mu_ineq,
                z_ineq,
                cost_mult,
                kkt_vals,
            );
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
        &mut persistent_solver,
        &v53_cache.base,
        ev.dg_cp.clone(),
        ev.dg_ri.clone(),
        ev.dg_vals0.clone(),
        ev.dh_cp.clone(),
        ev.dh_ri.clone(),
        ev.dh_vals0.clone(),
        ev.neqnln,
        ev.niqnln,
        ev.neq,
        ev.niq,
    )
}

/// V5.5 path: Zero-Allocation Jacobian + Partitioned Isomorphic assembly.
/// Eliminates intermediate matrices for both Hessian and Jacobian during the iteration loop.
pub fn pips_v5_5(
    data: &NewOPFData,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult {
    let v3_cache = V3SymbolicCache::analyze(data);
    let v53_cache = crate::new_opf::assembly::v5::partitioned::KKTSymbolicV5_3::build(data);
    let v55_evaluator = crate::new_opf::evaluation::v5::nonlinear::V55Evaluator::new(data);
    let mut persistent_solver = crate::basic::solver::DefaultSolver::default();

    crate::new_opf::interior_point::pips_with_fused_assembly_v55(
        |x| cost::opf_costfcn(data, x),
        |x, g, h, dgn_v, dhn_v| v55_evaluator.update(data, x, g, h, dgn_v, dhn_v),
        |x| {
            let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
            (h, g, dh, dg)
        },
        |x, lam_eq, mu_ineq, z_ineq, cost_mult, kkt_vals| {
            use crate::new_opf::assembly::v5::partitioned::*;
            assemble_kkt_v5_3(
                &v53_cache,
                data,
                &v3_cache.y_transpose_idx,
                x,
                lam_eq,
                mu_ineq,
                z_ineq,
                cost_mult,
                kkt_vals,
            );
        },
        x0,
        xmin,
        xmax,
        PipsOpt {
            merged_slacks: true,
            ..opt
        },
        &mut persistent_solver,
        &v53_cache.base,
    )
}

/// Reproducible solver configurations. All variants share the same input/output types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Configuration {
    V1,
    V4,
    V5_0,
    V5_2,
    V5_3,
    V5_5,
    #[default]
    V5_6,
}

impl Configuration {
    /// Assembly and evaluator names, kept explicit for experiment reporting.
    pub fn strategies(self) -> (&'static str, &'static str) {
        match self {
            Self::V1 => ("v1", "v1"),
            Self::V4 => ("v4", "v1"),
            Self::V5_0 => ("v5.0", "v1"),
            Self::V5_2 => ("v5.2", "v1"),
            Self::V5_3 => ("v5.3", "v1"),
            Self::V5_5 => ("v5.3", "v5.nonlinear"),
            Self::V5_6 => ("v5.3", "v5.merged"),
        }
    }

    /// Solve with model-derived bounds and an initial point in the model's packed order.
    /// Each call creates the selected strategy's caches and a fresh linear solver.
    pub fn solve(self, data: &NewOPFData, initial: Vec<f64>, options: PipsOpt) -> PipsResult {
        assert_eq!(
            initial.len(),
            data.nx(),
            "OPF initial point must match the model dimension"
        );
        let (lower, upper) = data.bounds();
        match self {
            Self::V1 => crate::opf::pips::pips(
                |x| cost::opf_costfcn(data, x),
                |x| {
                    let (g, h, dg, dh) = constraints::opf_consfcn(data, x);
                    (h, g, dh, dg)
                },
                |x, l, m, _z, c| crate::new_opf::assembly::v1::opf_hessfcn(data, x, l, m, c),
                initial,
                lower,
                upper,
                PipsOpt {
                    merged_slacks: false,
                    ..options
                },
            ),
            Self::V4 => pips(data, initial, lower, upper, options),
            Self::V5_0 => pips_v5(data, initial, lower, upper, options),
            Self::V5_2 => pips_v5_2(data, initial, lower, upper, options),
            Self::V5_3 => pips_v5_3(data, initial, lower, upper, options),
            Self::V5_5 => pips_v5_5(data, initial, lower, upper, options),
            Self::V5_6 => pips_v5_6(data, initial, lower, upper, options),
        }
    }
}
