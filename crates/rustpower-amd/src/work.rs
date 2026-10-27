use std::ops::{Deref, DerefMut, Index, IndexMut};
// Private workspace view. FAST is reachable only through the unsafe public APIs.
pub(crate) struct Work<'a, T, const FAST: bool>(pub &'a mut [T]);
impl<T, const FAST: bool> Deref for Work<'_, T, FAST> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.0
    }
}
impl<T, const FAST: bool> DerefMut for Work<'_, T, FAST> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.0
    }
}
impl<T, const FAST: bool> Index<usize> for Work<'_, T, FAST> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        if FAST {
            debug_assert!(i < self.0.len());
            unsafe { self.0.get_unchecked(i) }
        } else {
            &self.0[i]
        }
    }
}
impl<T, const FAST: bool> IndexMut<usize> for Work<'_, T, FAST> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        if FAST {
            debug_assert!(i < self.0.len());
            unsafe { self.0.get_unchecked_mut(i) }
        } else {
            &mut self.0[i]
        }
    }
}
