//! # Deferred Bundle System
//!
//! This module provides an ECS component bundle builder that defers allocation and insertion.
//! It allows for batching component construction using a bump allocator before committing them
//! into the world, offering performance and ergonomics benefits when spawning many similar entities.
//!
//! Inspired by direct memory control needs in dynamic ECS scenarios, this interface cleanly separates
//! construction, collection, and commit phases.

use std::ptr::NonNull;

use bevy_ecs::{
    component::ComponentId,
    prelude::*,
    ptr::{Aligned, OwningPtr},
};
use bumpalo::Bump;

/// A deferred component bundle builder that accumulates raw pointers and type IDs
/// before committing the bundle into an [`EntityWorldMut`].
///
/// Backed by a [`Bump`] allocator, this allows zero-copy-like deferred allocation of components.
pub struct DeferredBundleBuilder<'a> {
    entity: &'a mut EntityWorldMut<'a>,
    ids: Vec<ComponentId>,
    ptrs: Vec<OwningPtr<'a, Aligned>>,
    bump: &'a Bump,
}

impl<'a> DeferredBundleBuilder<'a> {
    /// Creates a new builder tied to the target entity and allocator.
    pub fn new(entity: &'a mut EntityWorldMut<'a>, bump: &'a Bump) -> Self {
        Self {
            entity,
            ids: vec![],
            ptrs: vec![],
            bump,
        }
    }

    /// Inserts a component into the builder.
    ///
    /// The component is allocated into the bump allocator, converted to a pointer,
    /// and associated with its [`ComponentId`] for later bulk commit.
    pub fn insert<T: Component>(&mut self, value: T) {
        let world = unsafe { self.entity.world_mut() };
        let id = world
            .component_id::<T>()
            .unwrap_or_else(|| world.register_component::<T>());
        let ptr = self.bump.alloc(value) as *mut T;
        let ptr = unsafe { OwningPtr::new(NonNull::new_unchecked(ptr.cast())) };
        self.insert_by_id(id, ptr);
    }

    /// Inserts a component by pre-resolved [`ComponentId`] and pointer.
    ///
    /// This can be used for dynamic scenarios where the component type is not statically known.
    pub fn insert_by_id(&mut self, id: ComponentId, ptr: OwningPtr<'a>) {
        self.ids.push(id);
        self.ptrs.push(ptr);
    }

    /// Finalizes the builder and commits all collected components to the target entity.
    ///
    /// This function consumes the builder and transfers ownership of component pointers into the ECS world.
    pub fn commit(mut self) {
        let entity = self.entity;
        unsafe { entity.insert_by_ids(&self.ids, self.ptrs.drain(..)) };
    }
}

/// A trait representing any structure that can insert itself into a [`DeferredBundleBuilder`].
///
/// Used to enable bulk spawning of heterogeneous component sets via [`DeferBundleSpawner`].
pub trait DeferBundle {
    /// Inserts the content of this structure into a builder.
    fn insert_to(self, builder: &mut DeferredBundleBuilder);
}

/// A batch spawner that defers component allocation using [`bumpalo`] and applies them via [`DeferBundle`].
///
/// Useful for spawning a large number of entities efficiently without temporary heap allocations.
pub struct DeferBundleSpawner {
    bump: Bump,
}

impl DeferBundleSpawner {
    /// Creates a new spawner with its own bump allocator.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Spawns a batch of [`DeferBundle`] instances into the world.
    ///
    /// Each item in the iterator is used to construct a new entity via deferred allocation.
    /// After the batch is spawned, the allocator is reset.
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
