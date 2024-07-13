use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

/// Represents the ground node in the network.
pub const GND: i32 = -1;

pub struct PowerGrid {
    data_storage: App,
}

pub trait DataOps {
    fn get_entity_mut(&mut self, entity: Entity) -> Option<EntityWorldMut<'_>>;
    fn get_mut<T>(&mut self, entity: Entity) -> Option<Mut<T>>
    where
        T: Component;
    fn get<T>(&self, entity: Entity) -> Option<&T>
    where
        T: Component;
    fn world_mut(&mut self) -> &mut World;
    fn world(&self) -> &World;
}

impl DataOps for PowerGrid {
    fn world(&self) -> &World {
        self.data_storage.world()
    }
    fn world_mut(&mut self) -> &mut World {
        self.data_storage.world_mut()
    }
    fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.world().get(entity)
    }
    fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<Mut<T>> {
        self.world_mut().get_mut(entity)
    }
    fn get_entity_mut(&mut self, entity: Entity) -> Option<EntityWorldMut<'_>> {
        self.world_mut().get_entity_mut(entity)
    }
}
