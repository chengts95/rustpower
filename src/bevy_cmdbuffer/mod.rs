pub mod buffer;
pub use buffer::*;

use bevy_ecs::ptr::{Aligned, OwningPtr, PtrMut};
use bevy_ecs::prelude::*;
use std::ptr::NonNull;

// this allows to have a type erased box that can drop the inner type correctly
// it must be dropped manually or it will leak memory.
pub struct ArenaBox<'a> {
    pub ptr: OwningPtr<'a, Aligned>,
    pub drop_fn: unsafe fn(OwningPtr<'a, Aligned>),
}

impl<'a> ArenaBox<'a> {
    pub fn new<T>(ptr: OwningPtr<'a, Aligned>) -> Self {
        Self {
            ptr,
            drop_fn: |ptr| unsafe {
                ptr.drop_as::<T>();
            },
        }
    }
    pub fn manual_drop(self) {
        unsafe { (self.drop_fn)(self.ptr) }
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    pub fn get_ptr_mut(&mut self) -> PtrMut<'a> {
        // SAFETY: We have &mut self, so we have exclusive access to the owning ptr and its data.
        unsafe { PtrMut::new(NonNull::new_unchecked(self.ptr.as_ptr() as *mut u8)) }
    }
}
