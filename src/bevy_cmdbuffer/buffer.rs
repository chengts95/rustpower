use bevy_ecs::prelude::*;
use bevy_ecs::ptr::{Aligned, OwningPtr};
use bevy_ecs::component::ComponentId;
use bumpalo::Bump;
use std::ptr::NonNull;
use crate::bevy_cmdbuffer::ArenaBox;

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
/// Note: This is NOT Send/Sync due to raw pointers and bump allocators.
/// It should be held by a single owner (like PowerGrid) and applied to the World.
pub struct HarvardCommandBuffer {
    ops: Vec<OpHead>,
    meta_bump: Bump,
    data_bump: Bump,
    
    // Staging for ModifyEntity Write Combining
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
                OpHead::ModifyEntity { args_ptr, count, .. } => {
                    let args = unsafe { std::slice::from_raw_parts(args_ptr.as_ptr(), *count as usize) };
                    for arg in args {
                        if let Some(drop_fn) = arg.drop_fn {
                             let ptr = unsafe { OwningPtr::new(arg.payload_ptr) };
                             unsafe { drop_fn(ptr) };
                        }
                    }
                }
                OpHead::BatchInsert { payload_ptr, count, stride, drop_fn, .. } => {
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

    pub fn data_bump(&self) -> &Bump {
        &self.data_bump
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
                        self.pending_args.swap_remove(i);
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

    pub fn insert_generic<T: Component>(&mut self, world: &World, entity: Entity, component: T) {
        let comp_id = world.component_id::<T>().expect("Component not registered");
        let ptr = self.data_bump.alloc(component) as *mut T;
        let payload_ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };
        let drop_fn: DropFn = |ptr| unsafe { ptr.drop_as::<T>() };
        
        self.insert_raw(entity, comp_id, payload_ptr, Some(drop_fn));
    }

    pub fn insert<T: Component>(&mut self, world: &World, entity: Entity, component: T) {
        self.insert_generic::<T>(world, entity, component);
    }

    pub fn remove<T: Component>(&mut self, world: &World, entity: Entity) {
        let comp_id = world.component_id::<T>().expect("Component not registered");
        self.remove_raw(entity, &[comp_id]);
    }

    pub fn insert_batch<T: Component, I>(&mut self, world: &World, entities: &[Entity], components: I)
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        self.flush();
        let comp_id = world.component_id::<T>().expect("Component not registered");
        let components_iter = components.into_iter();
        let slice = self.data_bump.alloc_slice_fill_iter(components_iter);
        let count = slice.len();
        
        if count != entities.len() {
            panic!("Batch insert mismatch: {} entities vs {} components", entities.len(), count);
        }
        if count == 0 { return; }

        let payload_ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr() as *mut u8) };
        let drop_fn: DropFn = |ptr| unsafe { ptr.drop_as::<T>() };
        let entities_slice = self.meta_bump.alloc_slice_copy(entities);
        let entities_ptr = unsafe { NonNull::new_unchecked(entities_slice.as_mut_ptr()) };

        self.ops.push(OpHead::BatchInsert {
            entities_ptr,
            payload_ptr,
            count: count as u32,
            comp_id,
            stride: std::mem::size_of::<T>(),
            drop_fn: Some(drop_fn),
        });
    }

    pub fn insert_box(&mut self, entity: Entity, comp_id: ComponentId, payload: ArenaBox<'_>) {
        let ptr = payload.ptr;
        let drop_fn = unsafe { std::mem::transmute::<_, DropFn>(payload.drop_fn) };
        let payload_ptr = NonNull::new(ptr.as_ptr() as *mut u8).expect("ArenaBox ptr is null");
        self.insert_raw(entity, comp_id, payload_ptr, Some(drop_fn));
    }

    fn insert_raw(&mut self, entity: Entity, comp_id: ComponentId, payload_ptr: NonNull<u8>, drop_fn: Option<DropFn>) {
        if self.pending_entity != Some(entity) {
            self.flush();
            self.pending_entity = Some(entity);
        }
        self.pending_args.push(ArgMeta { comp_id, payload_ptr, drop_fn });
    }

    pub fn remove_raw(&mut self, entity: Entity, components: &[ComponentId]) {
        self.flush();
        if components.is_empty() { return; }
        let slice = self.meta_bump.alloc_slice_copy(components);
        let ids_ptr = unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) };
        self.ops.push(OpHead::RemoveComponents {
            entity,
            ids_ptr,
            count: slice.len() as u16,
        });
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.flush();
        self.ops.push(OpHead::Despawn(entity));
    }

    pub fn apply(&mut self, world: &mut World) {
        self.flush();
        for op in &self.ops {
            match op {
                OpHead::ModifyEntity { entity, args_ptr, count } => {
                    let args = unsafe { std::slice::from_raw_parts(args_ptr.as_ptr(), *count as usize) };
                    let ids: Vec<ComponentId> = args.iter().map(|a| a.comp_id).collect();
                    let ptrs = args.iter().map(|a| unsafe { OwningPtr::new(a.payload_ptr) });
                    let _ = world.spawn_empty_at(*entity);
                    let mut entity_mut = world.entity_mut(*entity);
                    unsafe { entity_mut.insert_by_ids(&ids, ptrs) };
                }
                OpHead::BatchInsert { entities_ptr, payload_ptr, count, comp_id, stride, .. } => {
                    let entities = unsafe { std::slice::from_raw_parts(entities_ptr.as_ptr(), *count as usize) };
                    let mut ptr = payload_ptr.as_ptr();
                    for &entity in entities {
                        let _ = world.spawn_empty_at(entity);
                        let owning_ptr = unsafe { OwningPtr::new(NonNull::new_unchecked(ptr)) };
                        let mut entity_mut = world.entity_mut(entity);
                        unsafe { entity_mut.insert_by_id(*comp_id, owning_ptr) };
                        ptr = unsafe { ptr.add(*stride) };
                    }
                }
                OpHead::RemoveComponents { entity, ids_ptr, count } => {
                    let ids = unsafe { std::slice::from_raw_parts(ids_ptr.as_ptr(), *count as usize) };
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
    }

    pub fn reset(&mut self) {
        for arg in &self.pending_args {
            if let Some(drop_fn) = arg.drop_fn {
                let ptr = unsafe { OwningPtr::new(arg.payload_ptr) };
                unsafe { drop_fn(ptr) };
            }
        }
        for op in &self.ops {
            match op {
                OpHead::ModifyEntity { args_ptr, count, .. } => {
                    let args = unsafe { std::slice::from_raw_parts(args_ptr.as_ptr(), *count as usize) };
                    for arg in args {
                        if let Some(drop_fn) = arg.drop_fn {
                             let ptr = unsafe { OwningPtr::new(arg.payload_ptr) };
                             unsafe { drop_fn(ptr) };
                        }
                    }
                }
                OpHead::BatchInsert { payload_ptr, count, stride, drop_fn, .. } => {
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
        self.ops.clear();
        self.pending_args.clear();
        self.pending_entity = None;
        self.meta_bump.reset();
        self.data_bump.reset();
    }

    pub fn insert_bundle<B: BundleInserter>(&mut self, world: &World, entity: Entity, bundle: B) {
        bundle.insert_into(self, world, entity);
    }
}

pub trait BundleInserter {
    fn insert_into(self, buffer: &mut HarvardCommandBuffer, world: &World, entity: Entity);
}

macro_rules! impl_bundle_inserter {
    ($($name:ident),*) => {
        impl<$($name: Component),*> BundleInserter for ($($name,)*) {
            #[allow(non_snake_case)]
            fn insert_into(self, buffer: &mut HarvardCommandBuffer, world: &World, entity: Entity) {
                let ($($name,)*) = self;
                $(buffer.insert(world, entity, $name);)*
            }
        }
    }
}

impl_bundle_inserter!(A);
impl_bundle_inserter!(A, B);
impl_bundle_inserter!(A, B, C);
impl_bundle_inserter!(A, B, C, D);
impl_bundle_inserter!(A, B, C, D, E);
impl_bundle_inserter!(A, B, C, D, E, F);
impl_bundle_inserter!(A, B, C, D, E, F, G);
impl_bundle_inserter!(A, B, C, D, E, F, G, H);
impl_bundle_inserter!(A, B, C, D, E, F, G, H, I);
impl_bundle_inserter!(A, B, C, D, E, F, G, H, I, J);
