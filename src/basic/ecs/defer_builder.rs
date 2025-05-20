use std::ptr::NonNull;

use bevy_ecs::{
    component::ComponentId,
    prelude::*,
    ptr::{Aligned, OwningPtr},
};
use bumpalo::Bump;
pub struct DeferredEntityBuilder<'a> {
    entity: &'a mut EntityWorldMut<'a>,
    ids: Vec<ComponentId>,
    ptrs: Vec<OwningPtr<'a, Aligned>>,
    bump: &'a Bump,
}

impl<'a> DeferredEntityBuilder<'a> {
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
    pub fn insert_if_new_by_id(&mut self, id: ComponentId, ptr: OwningPtr<'a>) {
        if self.entity.contains_id(id) {
            return;
        }
        self.insert_by_id(id, ptr);
    }
    pub fn insert_by_id(&mut self, id: ComponentId, ptr: OwningPtr<'a>) {
        if let Some(i) = self.ids.iter().position(|&existing| existing == id) {
            self.ptrs[i] = ptr; // replace old value
        } else {
            self.ids.push(id);
            self.ptrs.push(ptr);
        }
    }

    pub fn commit(mut self) {
        let entity = self.entity;
        unsafe { entity.insert_by_ids(&self.ids, self.ptrs.drain(..)) };
    }
}
pub trait DeferBundle {
    fn insert_to(&self, builder: &mut DeferredEntityBuilder);
}
