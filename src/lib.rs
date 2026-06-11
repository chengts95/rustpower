mod basic;
pub mod bevy_cmdbuffer;
pub mod io;
pub mod testcases;
pub mod timeseries;

#[cfg(feature = "python")]
pub mod python;

pub mod prelude {
    pub use crate::basic::*;
    pub use crate::io::pandapower;
    pub use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    pub use crate::basic::ecs::post_processing::PostProcessing;
    pub use crate::basic::ecs::elements::PPNetwork;
    pub use crate::basic::ecs::powerflow::prelude::PowerFlowResult;
    pub use crate::basic::ecs::plugin::{default_app, IwamotoPlugin, CustomSolverActive};
}
