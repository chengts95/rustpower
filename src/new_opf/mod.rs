//! AC-OPF implementations with common inputs, outputs, and named configurations.
//!
//! Start with [`Configuration`] and [`model`]. Numerical assembly and constraint
//! evaluation are independent choices, recorded in [`configurations`]. The
//! conventional reference implementation remains in [`crate::opf`].
pub mod adapters;
pub mod assembly;
pub mod configurations;
pub mod evaluation;
pub mod interior_point;
pub mod model;
pub mod solution;
pub mod verification;

pub use adapters::ecs::components::*;
pub use assembly::mapped::symbolic::SymbolicCache;
pub use configurations::{Configuration, pips};
pub use model::NewOPFData;
pub use solution::{PipsOpt, PipsResult};

// Compatibility module names for existing callers. New code uses the folders above.
#[doc(hidden)]
pub use adapters::ecs::{components, translate};
#[doc(hidden)]
pub use assembly::mapped::{numeric, symbolic};
#[doc(hidden)]
pub use assembly::v3::{
    fused as v3_numeric_fused, numeric as v3_numeric, scalar as v3_numeric_scalar,
    symbolic as v3_symbolic,
};
#[doc(hidden)]
pub use assembly::v4::curvature as v4_numeric_rect;
#[doc(hidden)]
pub use assembly::v5::{partitioned as v5_3_kernel, scatter as v5_2_kernel, symbolic as v5_kkt};
#[doc(hidden)]
pub use configurations as pips;
#[doc(hidden)]
pub use evaluation::v5::{merged as v5_6_evaluator, nonlinear as v5_5_evaluator};
#[doc(hidden)]
pub use model as problem;
#[doc(hidden)]
pub use verification as math_verify;

#[cfg(test)]
mod tests;
