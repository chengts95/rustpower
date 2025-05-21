use std::ptr::NonNull;

use bevy_ecs::{
    component::ComponentId,
    prelude::*,
    ptr::{Aligned, OwningPtr},
};
use bumpalo::Bump;

pub struct DeferredBundleBuilder<'a> {
    entity: &'a mut EntityWorldMut<'a>,
    ids: Vec<ComponentId>,
    ptrs: Vec<OwningPtr<'a, Aligned>>,
    bump: &'a Bump,
}

impl<'a> DeferredBundleBuilder<'a> {
    pub fn new(entity: &'a mut EntityWorldMut<'a>, bump: &'a Bump) -> Self {
        Self {
            entity,
            ids: vec![],
            ptrs: vec![],
            bump,
        }
    }
    pub fn insert<T: Component>(&mut self, value: T) {
        let world = unsafe { self.entity.world_mut() };
        let id = world
            .component_id::<T>()
            .unwrap_or_else(|| world.register_component::<T>());
        let ptr = self.bump.alloc(value) as *mut T;
        let ptr = unsafe { OwningPtr::new(NonNull::new_unchecked(ptr.cast())) };
        self.insert_by_id(id, ptr);
    }
    // pub fn insert_if_new_by_id(&mut self, id: ComponentId, ptr: OwningPtr<'a>) {
    //     if self.entity.contains_id(id) {
    //         return;
    //     }
    //     self.insert_by_id(id, ptr);
    // }
    pub fn insert_by_id(&mut self, id: ComponentId, ptr: OwningPtr<'a>) {
        self.ids.push(id);
        self.ptrs.push(ptr);
    }

    pub fn commit(mut self) {
        let entity = self.entity;
        unsafe { entity.insert_by_ids(&self.ids, self.ptrs.drain(..)) };
    }
}

pub trait DeferBundle {
    fn insert_to(self, builder: &mut DeferredBundleBuilder);
}

pub struct DeferBundleSpawner {
    bump: Bump,
}

impl DeferBundleSpawner {
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }
    pub fn spawn_batch<T: DeferBundle, U>(&mut self, world: &mut World, data: U)
    where
        U: IntoIterator<Item = T>,
    {
        for d in data {
            let mut entity = world.spawn_empty();
            let mut builder = DeferredBundleBuilder::new(&mut entity, &self.bump);
            d.insert_to(&mut builder);
            builder.commit();
        }
        self.bump.reset();
    }
}
