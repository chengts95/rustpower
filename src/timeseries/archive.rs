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
#[derive(Default)]
pub struct TimeSeriesArchivePlugin;

impl Plugin for TimeSeriesArchivePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ArchivePlugin>() {
            app.add_plugins(ArchivePlugin);
        }
        let a = app.world_mut().resource_mut::<ArchiveSnapshotRes>();
        let mut output_reg = a.0.output_reg.clone();

        let d = unsafe { output_reg.get_mut_unchecked() };
        register_res_all!(d, [TimeSeriesData]);
        let mut reg = a.0.case_file_reg.clone();
        let d = unsafe { reg.get_mut_unchecked() };
        register_res_all!(d, [ScheduledLog, Time, DeltaTime]);
        register_all!(d, [ScheduledStaticActions]);
    }
}
