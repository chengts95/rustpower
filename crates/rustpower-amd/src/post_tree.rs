use crate::internal::*;

pub fn post_tree(
    root: usize,
    mut k: usize,
    child: &mut [isize], // input of size nn, undefined on output
    sibling: &[isize],   // input of size nn, not modified
    order: &mut [isize], // output of size nn
    stack: &mut [isize],
    nn: usize,
) -> usize {
    /*if false {
        // Recursive version (stack[] is not used):
        // this is simple, but can can cause stack overflow if nn is large
        i = root;
        f = child[i];
        while f != EMPTY {
            k = post_tree(f, k, child, sibling, order, stack, nn);
            f = sibling[f];
        }
        order[i] = k;
        k += 1;
        return k;
    }*/

    // Non-recursive version, using an explicit stack.

    // Push root on the stack.
    let mut head: isize = 0;
    stack[0] = root as isize;

    while head >= 0 {
        // Get head of stack.
        debug_assert!((head as usize) < nn);
        let i = stack[head as usize];
        debug1_print!("head of stack {} \n", i);
        debug_assert!(i >= 0 && i < nn as isize);

        if child[i as usize] != EMPTY {
            // The children of i are not yet ordered
            // push each child onto the stack in reverse order
            // so that small ones at the head of the list get popped first
            // and the biggest one at the end of the list gets popped last.
            let mut f = child[i as usize];
            while f != EMPTY {
                head += 1;
                debug_assert!((head as usize) < nn);
                debug_assert!(f >= 0 && f < nn as isize);
                f = sibling[f as usize];
            }
            let mut h = head;
            debug_assert!((head as usize) < nn);
            let mut f = child[i as usize];
            while f != EMPTY {
                debug_assert!(h > 0);
                stack[h as usize] = f;
                h -= 1;
                debug1_println!("push {} on stack", f);
                debug_assert!(f >= 0 && f < nn as isize);
                f = sibling[f as usize];
            }
            debug_assert!(stack[h as usize] == i);

            // Delete child list so that i gets ordered next time we see it.
            child[i as usize] = EMPTY;
        } else {
            // The children of i (if there were any) are already ordered
            // remove i from the stack and order it. Front i is kth front.
            head -= 1;
            debug1_print!("pop {} order {}\n", i, k);
            order[i as usize] = k as isize;
            k += 1;
            debug_assert!(k <= nn);
        }

        #[cfg(feature = "debug1")]
        {
            debug1_print!("\nStack:");
            // for h := head; h >= 0; h-- {
            let mut h = head;
            while h >= 0 {
                let j = stack[h as usize];
                debug1_print!(" {}", j);
                debug_assert!(j >= 0 && j < nn as isize);
                h -= 1;
            }
            debug1_print!("\n\n");
            debug_assert!(head < nn as isize);
        }
    }

    k
}
