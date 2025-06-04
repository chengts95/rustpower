use std::sync::Arc;

use bevy_app::{App, Plugin};

use crate::{
    io::archive::aurora_format::{ArchivePlugin, ArchiveSnapshotRes},
    timeseries::{
        scheduled::{ScheduledLog, ScheduledStaticActions},
        sim_time::{DeltaTime, Time},
        state::TimeSeriesData,
    },
};
#[allow(unused_macros)]

macro_rules! register_all {
    ($reg:expr, [$($ty:ty),* $(,)?]) => {
        $( ($reg).register::<$ty>(); )*
    };
}
macro_rules! register_res_all {
    ($reg:expr, [$($ty:ty),* $(,)?]) => {
        $( ($reg).resource_register::<$ty>(); )*
    };
}
trait ArcMutExt<T> {
    unsafe fn get_mut_unchecked(&mut self) -> &mut T;
}

impl<T> ArcMutExt<T> for Arc<T> {
    unsafe fn get_mut_unchecked(&mut self) -> &mut T {
        let ptr = Arc::as_ptr(self) as *mut T;
        unsafe { &mut *ptr }
    }
}
/// Plugin that integrates time-series related resources into the archive system.
///
/// Registers all relevant time-dependent resources such as:
/// - [`TimeSeriesData`] for voltage history
/// - [`ScheduledLog`] for executed events
/// - [`Time`], [`DeltaTime`] for time progression
/// - [`ScheduledStaticActions`] for queued actions
///
/// # Dependencies
/// Automatically includes [`ArchivePlugin`] to enable snapshot/export functionality.
///
/// # Behavior
/// - Registers resources to both output archive and case file registry.
/// - Uses `unsafe` to bypass interior mutability of archive registries.
///
/// # Usage
/// Add this plugin when using the `archive` feature to enable persistent logging or exporting of time-series data.
#[derive(Default)]
pub struct TimeSeriesArchivePlugin;

impl Plugin for TimeSeriesArchivePlugin {
    fn build(&self, app: &mut App) {
        // Ensure ArchivePlugin is loaded
        if !app.is_plugin_added::<ArchivePlugin>() {
            app.add_plugins(ArchivePlugin);
        }

        // Access global archive snapshot resource
        let a = app.world_mut().resource_mut::<ArchiveSnapshotRes>();

        // Register output-only resources (e.g., for result visualization)
        let mut output_reg = a.0.output_reg.clone();
        let d = unsafe { output_reg.get_mut_unchecked() };
        register_res_all!(d, [TimeSeriesData]);

        // Register input/output resources (for full snapshot/restore)
        let mut reg = a.0.case_file_reg.clone();
        let d = unsafe { reg.get_mut_unchecked() };
        register_res_all!(d, [ScheduledLog, Time, DeltaTime]);
        register_all!(d, [ScheduledStaticActions]);
    }
}
