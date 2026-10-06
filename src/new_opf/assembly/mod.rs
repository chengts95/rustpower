//! Curvature and KKT assembly. Each version owns its symbolic mappings and kernels.
//! V1 adapts the independent reference; `mapped` retains the earlier mapped prototype.
pub mod mapped;
pub mod v1;
pub mod v3;
pub mod v4;
pub mod v5;
