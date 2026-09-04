use bevy_ecs::component::ComponentId;
use bevy_ecs::prelude::*;
use bevy_ecs::ptr::{Aligned, OwningPtr};
use bumpalo::Bump;
use std::ptr::NonNull;

// Safety: Must be called with a pointer to the correct type T.
pub type DropFn = unsafe fn(OwningPtr<'_, Aligned>);

#[derive(Clone, Copy, Debug)]
pub struct ArgMeta {
    pub comp_id: ComponentId,
    pub payload_ptr: NonNull<u8>,
    pub drop_fn: Option<DropFn>,
}

#[derive(Clone, Copy, Debug)]
pub enum OpHead {
    ModifyEntity {
        entity: Entity,
        args_ptr: NonNull<ArgMeta>,
        count: u16,
    },
    BatchInsert {
        entities_ptr: NonNull<Entity>,
        payload_ptr: NonNull<u8>,
        count: u32,
        comp_id: ComponentId,
        stride: usize,
        drop_fn: Option<DropFn>,
    },
    RemoveComponents {
        entity: Entity,
        ids_ptr: NonNull<ComponentId>,
        count: u16,
    },
    Despawn(Entity),
}

/// Harvard Architecture Command Buffer.
pub struct HarvardCommandBuffer {
    ops: Vec<OpHead>,
    meta_bump: Bump,
    data_bump: Bump,
    pending_entity: Option<Entity>,
    pending_args: Vec<ArgMeta>,
}

impl Default for HarvardCommandBuffer {
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            meta_bump: Bump::new(),
            data_bump: Bump::new(),
            pending_entity: None,
            pending_args: Vec::new(),
        }
    }
}

impl Drop for HarvardCommandBuffer {
    fn drop(&mut self) {
        for arg in &self.pending_args {
            if let Some(drop_fn) = arg.drop_fn {
                let ptr = unsafe { OwningPtr::new(arg.payload_ptr) };
                unsafe { drop_fn(ptr) };
            }
        }
        for op in &self.ops {
            match op {
                OpHead::ModifyEntity {
                    args_ptr, count, ..
                } => {
                    let args =
                        unsafe { std::slice::from_raw_parts(args_ptr.as_ptr(), *count as usize) };
                    for arg in args {
                        if let Some(drop_fn) = arg.drop_fn {
                            let ptr = unsafe { OwningPtr::new(arg.payload_ptr) };
                            unsafe { drop_fn(ptr) };
                        }
                    }
                }
                OpHead::BatchInsert {
                    payload_ptr,
                    count,
                    stride,
                    drop_fn,
                    ..
                } => {
                    if let Some(drop_fn) = drop_fn {
                        let mut ptr = payload_ptr.as_ptr();
                        for _ in 0..*count {
                            let owning_ptr = unsafe { OwningPtr::new(NonNull::new_unchecked(ptr)) };
                            unsafe { drop_fn(owning_ptr) };
                            ptr = unsafe { ptr.add(*stride) };
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

impl HarvardCommandBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    fn flush(&mut self) {
        if let Some(entity) = self.pending_entity.take() {
            if !self.pending_args.is_empty() {
                let mut i = 0;
                while i < self.pending_args.len() {
                    let id = self.pending_args[i].comp_id;
                    let mut overwritten = false;
                    for j in (i + 1)..self.pending_args.len() {
                        if self.pending_args[j].comp_id == id {
                            overwritten = true;
                            break;
                        }
                    }
                    if overwritten {
                        if let Some(drop_fn) = self.pending_args[i].drop_fn {
                            let ptr = unsafe { OwningPtr::new(self.pending_args[i].payload_ptr) };
                            unsafe { drop_fn(ptr) };
                        }
                        self.pending_args.remove(i);
                    } else {
                        i += 1;
                    }
                }
                if !self.pending_args.is_empty() {
                    let slice = self.meta_bump.alloc_slice_copy(&self.pending_args);
                    let count = slice.len() as u16;
                    let args_ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) };
                    self.ops.push(OpHead::ModifyEntity {
                        entity,
                        args_ptr,
                        count,
                    });
                }
                self.pending_args.clear();
            }
        }
    }

    pub fn insert<T: Component>(&mut self, world: &mut World, entity: Entity, component: T) {
        let comp_id = world.register_component::<T>();
        let ptr = self.data_bump.alloc(component) as *mut T;
        let payload_ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };
        let drop_fn: DropFn = |ptr| unsafe { ptr.drop_as::<T>() };
        self.insert_raw(entity, comp_id, payload_ptr, Some(drop_fn));
    }

    fn insert_raw(
        &mut self,
        entity: Entity,
        comp_id: ComponentId,
        payload_ptr: NonNull<u8>,
        drop_fn: Option<DropFn>,
    ) {
        if self.pending_entity != Some(entity) {
            self.flush();
            self.pending_entity = Some(entity);
        }
        self.pending_args.push(ArgMeta {
            comp_id,
            payload_ptr,
            drop_fn,
        });
    }

    pub fn apply(&mut self, world: &mut World) {
        self.flush();
        for op in &self.ops {
            match op {
                OpHead::ModifyEntity {
                    entity,
                    args_ptr,
                    count,
                } => {
                    let args =
                        unsafe { std::slice::from_raw_parts(args_ptr.as_ptr(), *count as usize) };
                    let ids: Vec<ComponentId> = args.iter().map(|a| a.comp_id).collect();
                    let ptrs = args
                        .iter()
                        .map(|a| unsafe { OwningPtr::new(a.payload_ptr) });
                    let _ = world.spawn_empty_at(*entity);
                    let mut entity_mut = world.entity_mut(*entity);
                    unsafe { entity_mut.insert_by_ids(&ids, ptrs) };
                }
                OpHead::BatchInsert {
                    entities_ptr,
                    payload_ptr,
                    count,
                    comp_id,
                    stride,
                    ..
                } => {
                    let entities = unsafe {
                        std::slice::from_raw_parts(entities_ptr.as_ptr(), *count as usize)
                    };
                    let mut ptr = payload_ptr.as_ptr();
                    for &entity in entities {
                        let _ = world.spawn_empty_at(entity);
                        let owning_ptr = unsafe { OwningPtr::new(NonNull::new_unchecked(ptr)) };
                        let mut entity_mut = world.entity_mut(entity);
                        unsafe { entity_mut.insert_by_id(*comp_id, owning_ptr) };
                        ptr = unsafe { ptr.add(*stride) };
                    }
                }
                OpHead::RemoveComponents {
                    entity,
                    ids_ptr,
                    count,
                } => {
                    let ids =
                        unsafe { std::slice::from_raw_parts(ids_ptr.as_ptr(), *count as usize) };
                    if let Ok(mut entity_mut) = world.get_entity_mut(*entity) {
                        entity_mut.remove_by_ids(ids);
                    }
                }
                OpHead::Despawn(entity) => {
                    world.despawn(*entity);
                }
            }
        }
        self.ops.clear();
        self.pending_args.clear();
        self.pending_entity = None;
        self.meta_bump.reset();
        self.data_bump.reset();
    }

    pub fn insert_bundle<B: DeferBundle>(&mut self, world: &mut World, entity: Entity, bundle: B) {
        bundle.insert_into(self, world, entity);
    }
}

pub trait DeferBundle {
    fn insert_into(self, buffer: &mut HarvardCommandBuffer, world: &mut World, entity: Entity);
}

// Support tuples as well
macro_rules! impl_defer_bundle {
    ($($name:ident),*) => {
        impl<$($name: Component),*> DeferBundle for ($($name,)*) {
            #[allow(non_snake_case)]
            fn insert_into(self, buffer: &mut HarvardCommandBuffer, world: &mut World, entity: Entity) {
                let ($($name,)*) = self;
                $(buffer.insert(world, entity, $name);)*
            }
        }
    }
}

impl_defer_bundle!(A);
impl_defer_bundle!(A, B);
impl_defer_bundle!(A, B, C);
impl_defer_bundle!(A, B, C, D);
impl_defer_bundle!(A, B, C, D, E);
impl_defer_bundle!(A, B, C, D, E, F);
impl_defer_bundle!(A, B, C, D, E, F, G);
impl_defer_bundle!(A, B, C, D, E, F, G, H);
impl_defer_bundle!(A, B, C, D, E, F, G, H, I);
impl_defer_bundle!(A, B, C, D, E, F, G, H, I, J);

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct CompA(u32);

    #[derive(Component, Debug, PartialEq, Eq)]
    struct CompB(u32);

    #[derive(Component)]
    struct DropTracker(Arc<AtomicUsize>);

    impl Drop for DropTracker {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_dedup_keeps_latest_and_drops_earlier() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut buffer = HarvardCommandBuffer::new();

        buffer.insert(&mut world, entity, CompA(1));
        buffer.insert(&mut world, entity, CompB(100));
        buffer.insert(&mut world, entity, CompA(2));
        buffer.insert(&mut world, entity, CompA(3));

        buffer.apply(&mut world);

        assert_eq!(world.get::<CompA>(entity), Some(&CompA(3)));
        assert_eq!(world.get::<CompB>(entity), Some(&CompB(100)));
    }

    #[test]
    fn test_dedup_drops_overwritten_components() {
        let mut world = World::new();
        let entity = world.spawn_empty().id();
        let mut buffer = HarvardCommandBuffer::new();

        let drop_count = Arc::new(AtomicUsize::new(0));

        buffer.insert(&mut world, entity, DropTracker(drop_count.clone()));
        buffer.insert(&mut world, entity, DropTracker(drop_count.clone()));
        buffer.insert(&mut world, entity, DropTracker(drop_count.clone()));

        // Before apply, 2 overwritten DropTrackers should be dropped during flush/apply
        buffer.apply(&mut world);
        assert_eq!(drop_count.load(Ordering::SeqCst), 2);

        // Despawn entity, the 3rd one should be dropped
        world.despawn(entity);
        assert_eq!(drop_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_buffer_reuse_after_apply() {
        let mut world = World::new();
        let mut buffer = HarvardCommandBuffer::new();

        for i in 0..10 {
            let entity = world.spawn_empty().id();
            buffer.insert(&mut world, entity, CompA(i));
            buffer.apply(&mut world);
            assert_eq!(world.get::<CompA>(entity), Some(&CompA(i)));
        }
    }
}
