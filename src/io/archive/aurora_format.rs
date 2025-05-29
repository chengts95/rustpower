use bevy_app::prelude::*;
pub use bevy_archive::archetype_archive::load_world_resource;
pub use bevy_archive::archetype_archive::save_world_resource;
pub use bevy_archive::prelude::*;
use bevy_ecs::entity::Entity;
use bevy_ecs::hierarchy::ChildOf;
use serde::Deserialize;
use serde::Serialize;

use crate::basic::ecs::network::DataOps;
use crate::basic::ecs::network::PowerGrid;
use crate::basic::ecs::powerflow::systems::PowerFlowConfig;
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

impl PowerGrid {
    pub fn register_types(&mut self) {
        self.app_mut().add_plugins(ArchivePlugin);
    }
    pub fn get_snapshot_reg(&self) -> Option<&SnapshotRegistry> {
        self.world().get_resource::<SnapshotRegistry>()
    }
}
impl Plugin for ArchivePlugin {
    fn build(&self, app: &mut App) {
        use crate::prelude::ecs::elements::*;
        let mut reg = build_snapshot_registry();
        reg.register_with::<ChildOf, ChildOfWrapper>();
        register_res_all!(reg, [PFCommonData, PowerFlowConfig]);
        app.insert_resource(reg);
    }
}
