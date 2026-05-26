use bevy_ecs::prelude::Component;
use num_complex::Complex64;

/// Lagrange multiplier for bus power balance (P and Q).
/// Typically associated with Bus entities.
#[derive(Component, Debug, Clone, Default)]
pub struct LambdaBus {
    pub p: f64,
    pub q: f64,
}

/// Lagrange multiplier for branch flow limits (from and to side).
/// Associated with Line or Transformer entities.
#[derive(Component, Debug, Clone, Default)]
pub struct MuFlow {
    pub from: f64,
    pub to: f64,
}

/// Lagrange multipliers for variable limits (upper and lower).
#[derive(Component, Debug, Clone, Default)]
pub struct MuLimit {
    pub lower: f64,
    pub upper: f64,
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
