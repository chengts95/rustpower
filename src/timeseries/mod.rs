#[cfg(feature = "archive")]
pub mod archive;
pub mod sim_time;
pub mod state;

use bevy_app::plugin_group;
use sim_time::TimePlugin;

plugin_group! {
    /// Doc comments and annotations are supported: they will be added to the generated plugin
    /// group.

    pub struct TimeSeriesDefaultPlugins {
     : TimePlugin ,
        // #[cfg(feature = "archive")]
        // crate::io::archive::aurora_format:::ArchivePlugin,


    }

}
