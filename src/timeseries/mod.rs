/// Optional plugin for recording time-series data during simulation.
/// Only compiled when the `archive` feature is enabled.
#[cfg(feature = "archive")]
pub mod archive;

/// Plugin that enables time-driven scheduled event injection (e.g., switching operations).
pub mod scheduled;

/// Plugin that provides global simulation time tracking and step advancement logic.
pub mod sim_time;

/// Plugin for transferring system state across simulation frames or external interfaces.
pub mod state;
use bevy_app::plugin_group;
use sim_time::TimePlugin;

use crate::timeseries::{
    archive::TimeSeriesArchivePlugin, scheduled::ScheduledEventPlugin, state::StateTransferPlugin,
};

plugin_group! {

    /// Default set of plugins for time-series simulation control and I/O.
///
/// This group includes systems for:
/// - Advancing and tracking simulation time (`TimePlugin`)
/// - Executing scheduled control events at specified timestamps (`ScheduledEventPlugin`)
/// - Propagating or transferring simulation state (`StateTransferPlugin`)
/// - Optionally archiving time-series results (`TimeSeriesArchivePlugin`, feature-gated)
/// # Feature Flags
/// - `archive`: Enables persistent time-series logging via `TimeSeriesArchivePlugin`.
    #[derive(Debug)]
    pub struct TimeSeriesDefaultPlugins {
        :TimePlugin,
        :StateTransferPlugin,
        :ScheduledEventPlugin,

        #[cfg(feature = "archive")]
        crate::timeseries:::TimeSeriesArchivePlugin,
    }
}
