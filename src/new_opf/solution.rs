//! Shared solver options and outputs for every named configuration.
//!
//! `PipsResult::x` uses the model's `[angle, magnitude, Pg, Qg]` ranges, in the
//! original model order. Angles are radians; powers are per unit; `f` is the
//! unscaled objective. A nonconverged result still contains its last iterate.
//!
//! Current limitation: bound multipliers are zero placeholders and equality
//! multipliers retain objective scaling. Do not interpret them as market prices.
pub use crate::opf::pips::{PipsOpt, PipsResult, PipsTiming};
