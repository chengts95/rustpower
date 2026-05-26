#![doc = include_str!("../doc/manifest.md")]
mod basic;
pub mod opf;
pub mod new_opf;
pub use basic::ecs::network;
pub mod io;
pub mod testcases;
pub mod timeseries;

#[cfg(feature = "python")]
pub mod python;
pub mod prelude {
    use crate::basic;
    pub use crate::io::pandapower;
    pub use basic::*;

    pub use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    pub use crate::basic::ecs::post_processing::PostProcessing;
    pub use ecs::{elements::PPNetwork, plugin::default_app, powerflow::prelude::PowerFlowResult};
}
