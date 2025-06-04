
use std::sync::Arc;

use crate::basic::ecs::post_processing::SBusResult;
use crate::basic::ecs::post_processing::VBusResult;
use crate::basic::ecs::powerflow::systems::PowerFlowConfig;
use crate::prelude::default_app;
use bevy_app::prelude::*;
pub use bevy_archive::archetype_archive::load_world_resource;
pub use bevy_archive::archetype_archive::save_world_resource;

pub use bevy_archive::prelude::*;
use bevy_ecs::entity::Entity;
use bevy_ecs::hierarchy::ChildOf;
use bevy_ecs::resource::Resource;
use serde::Deserialize;
use serde::Serialize;
#[derive(Default)]
pub struct ArchivePlugin;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ChildOfWrapper(pub u32);
impl From<&ChildOf> for ChildOfWrapper {
    fn from(c: &ChildOf) -> Self {
        ChildOfWrapper(c.0.index())
    }
}

impl Into<ChildOf> for ChildOfWrapper {
    fn into(self) -> ChildOf {
        ChildOf(Entity::from_raw(self.0))
    }
}
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
#[derive(Debug, Clone, Default)]
pub struct ArchiveSnapshotReg {
    pub case_file_reg: Arc<SnapshotRegistry>,
    pub pf_state_reg: Arc<SnapshotRegistry>,
    pub output_reg: Arc<SnapshotRegistry>,
}
#[derive(Resource, Clone)]
pub struct ArchiveSnapshotRes(pub Arc<ArchiveSnapshotReg>); // Defines the interface for creating and restoring snapshots of the RustPower application state.
///
/// This trait provides methods to convert the application state into a case file or simulation state,
/// and to restore the application state from a case file.
pub trait RustPowerSnapshotTrait {
    /// Converts the current application state into a case file.
    ///
    /// # Returns
    /// - `Ok(AuroraWorldManifest)` on success, containing the manifest of the world state.
    /// - `Err(String)` on failure, containing an error message.
    fn to_case_file(&self) -> Result<AuroraWorldManifest, String>;

    /// Converts the current application state into a simulation state.
    ///
    /// # Returns
    /// - `Ok(AuroraWorldManifest)` on success, containing the manifest of the world state.
    /// - `Err(String)` on failure, containing an error message.
    fn to_sim_states(&self) -> Result<AuroraWorldManifest, String>;

    /// Restores the application state from a case file.
    ///
    /// # Parameters
    /// - `manifest`: The `AuroraWorldManifest` containing the world state to restore.
    ///
    /// # Returns
    /// - `Ok(Self)` on success, returning a new instance of the application with the restored state.
    /// - `Err(String)` on failure, containing an error message.
    fn from_case_file(manifest: AuroraWorldManifest) -> Result<Self, String>
    where
        Self: Sized;
}

/// Implementation of the `RustPowerSnapshotTrait` for `App`.
impl RustPowerSnapshotTrait for App {
    fn to_case_file(&self) -> Result<AuroraWorldManifest, String> {
        let reg = self
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let case_reg = &reg.0.case_file_reg;
        save_world_manifest(self.world(), case_reg)
    }

    fn to_sim_states(&self) -> Result<AuroraWorldManifest, String> {
        let reg = self
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let sim_reg = &reg.0.pf_state_reg;
        save_world_manifest(self.world(), sim_reg)
    }

    fn from_case_file(manifest: AuroraWorldManifest) -> Result<Self, String>
    where
        Self: Sized,
    {
        let mut app = default_app();
        app.add_plugins(ArchivePlugin);

        let archive = app
            .world()
            .get_resource::<ArchiveSnapshotRes>()
            .ok_or("Missing ArchiveSnapshotRes")?;
        let case = archive.0.case_file_reg.clone();
        load_world_manifest(app.world_mut(), &manifest, &case)?;

        Ok(app)
    }
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
impl Plugin for ArchivePlugin {
    fn build(&self, app: &mut App) {
        use crate::prelude::ecs::elements::*;
        let mut reg = ArchiveSnapshotReg::default();
        let mut case_file_reg = Arc::new(build_snapshot_registry());
        reg.case_file_reg = case_file_reg.clone();
        let d = unsafe { case_file_reg.get_mut_unchecked() };
        register_res_all!(d, [PowerFlowConfig, PFCommonData,]);
        let pf_reg = Arc::new({
            let mut pf_reg = SnapshotRegistry::default();
            pf_reg.register_with::<ChildOf, ChildOfWrapper>();
            register_all!(pf_reg, [Admittance, Port2, VBase,]);

            pf_reg
        });

        let out_reg = Arc::new({
            let mut out_reg = SnapshotRegistry::default();
            register_all!(out_reg, [BusID, VBusResult, SBusResult]);
            out_reg
        });
        reg.pf_state_reg = pf_reg;
        reg.output_reg = out_reg;

        app.insert_resource(ArchiveSnapshotRes(Arc::new(reg)));
    }
}
