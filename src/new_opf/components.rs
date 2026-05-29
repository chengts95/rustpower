use bevy_ecs::prelude::*;
use nalgebra_sparse::CscMatrix;
use crate::basic::solver::DefaultSolver;

/// Lagrange multiplier for bus power balance (P and Q).
#[derive(Component, Debug, Clone, Default)]
pub struct LambdaBus {
    pub p: f64,
    pub q: f64,
}

/// Lagrange multiplier for branch flow limits.
#[derive(Component, Debug, Clone, Default)]
pub struct MuFlow {
    pub from: f64,
    pub to: f64,
}

/// Persistent workspace for OPF calculations to avoid repeated allocations.
#[derive(Resource)]
pub struct OPFWorkspace {
    pub solver: DefaultSolver,
    /// Pre-allocated KKT matrix skeleton [M dg; dg^T 0]
    pub kkt_skeleton: Option<CscMatrix<f64>>,
    /// Cached mapping for fast numeric assembly
    pub mapping: Option<KKTMapping>,
}

impl Default for OPFWorkspace {
    fn default() -> Self {
        Self {
            solver: DefaultSolver::default(),
            kkt_skeleton: None,
            mapping: None,
        }
    }
}

pub struct KKTMapping {
    pub lxx_ptrs: Vec<usize>,
    pub dg_ptrs: Vec<usize>,
}

/// Solved active power dispatch for a generator (p.u.).
#[derive(Component, Debug, Clone, Default)]
pub struct OpfResultPg(pub f64);

/// Solved reactive power dispatch for a generator (p.u.).
#[derive(Component, Debug, Clone, Default)]
pub struct OpfResultQg(pub f64);

/// Solved voltage magnitude for a bus (p.u.).
#[derive(Component, Debug, Clone, Default)]
pub struct OpfResultVm(pub f64);

/// Solved voltage angle for a bus (rad).
#[derive(Component, Debug, Clone, Default)]
pub struct OpfResultVa(pub f64);

/// Branch apparent-power limit |S_max|, in per-unit on system base.
///
/// Optional OPF-only component. Attached by `attach_line_flow_limits` /
/// `attach_trafo_flow_limits` from the matching pandapower fields. Absence on a
/// branch entity means "no flow limit" (treated as infinity by the OPF solver).
#[derive(Component, Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BranchFlowLimit {
    pub rate_a_pu: f64,
}
