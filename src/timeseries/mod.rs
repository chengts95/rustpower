#[cfg(feature = "archive")]
pub mod archive;
pub mod scheduled;
pub mod sim_time;
pub mod state;
use bevy_app::plugin_group;
use sim_time::TimePlugin;

use crate::timeseries::{
    archive::TimeSeriesArchivePlugin, scheduled::ScheduledEventPlugin, state::StateTransferPlugin,
};

plugin_group! {
    /// Doc comments and annotations are supported: they will be added to the generated plugin
    /// group.

    pub struct TimeSeriesDefaultPlugins {
     : TimePlugin ,
     :StateTransferPlugin ,
     :ScheduledEventPlugin,

        #[cfg(feature = "archive")]
        crate::timeseries:::TimeSeriesArchivePlugin,


    }

}
