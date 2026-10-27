use crate::internal::*;
use crate::post_tree::post_tree;

pub fn postorder(nn: usize, parent: &[isize], nv: &[isize], f_size: &[isize]) -> Vec<isize> {
    // output
    let mut order: Vec<isize> = vec![0; nn];

    // local workspace
    let mut child: Vec<isize> = vec![0; nn];
    let mut sibling: Vec<isize> = vec![0; nn];

    for j in 0..nn {
        child[j] = EMPTY;
        sibling[j] = EMPTY;
    }

    // Place the children in link lists - bigger elements tend to be last.
    let mut j: isize = nn as isize - 1;
    while j >= 0 {
        if nv[j as usize] > 0 {
            // This is an element.
            let p = parent[j as usize];
            if p != EMPTY {
                // Place the element in link list of the children its parent
                // bigger elements will tend to be at the end of the list.
                sibling[j as usize] = child[p as usize];
                child[p as usize] = j;
            }
        }
        j -= 1;
    }

    #[cfg(feature = "debug1")]
    {
        // var nels, ff int
        debug1_print!("\n\n================================ AMD_postorder:\n");

        let mut nels = 0;
        for j in 0..nn {
            if nv[j] > 0 {
                debug1_print!(
                    "{} :  nels {} npiv {} size {} parent {} maxfr {}\n",
                    j,
                    nels,
                    nv[j],
                    f_size[j],
                    parent[j],
                    f_size[j]
                );
                // This is an element. Dump the link list of children.
                let mut nchild = 0;
                debug1_print!("    Children:");
                let mut ff = child[j];
                while ff != EMPTY {
                    debug1_print!(" {}", ff);
                    debug_assert!(parent[ff as usize] == j as isize);
                    nchild += 1;
                    debug_assert!(nchild < nn);

                    ff = sibling[ff as usize];
                }
                debug1_println!();
                let p = parent[j];
                if p != EMPTY {
                    debug_assert!(nv[p as usize] > 0);
                }
                nels += 1;
            }
        }
        debug1_println!(
            "\n\nGo through the children of each node, and put
the biggest child last in each list:"
        );
    }

    // Place the largest child last in the list of children for each node.
    for i in 0..nn {
        if nv[i] > 0 && child[i] != EMPTY {
            let mut nchild = 0;
            {
                debug1_print!("Before partial sort, element {}\n", i);

                let mut f = child[i];
                while f != EMPTY {
                    debug_assert!(f >= 0 && f < nn as isize);
                    debug1_print!("      f: {}  size: {}\n", f, f_size[f as usize]);
                    nchild += 1;
                    debug_assert!(nchild <= nn);

                    f = sibling[f as usize];
                }
            }

            // Find the biggest element in the child list.
            let mut fprev = EMPTY;
            let mut maxfrsize = EMPTY;
            let mut bigfprev = EMPTY;
            let mut bigf = EMPTY;

            let mut f = child[i];
            while f != EMPTY {
                debug_assert!(f >= 0 && f < nn as isize);

                let frsize = f_size[f as usize];
                if frsize >= maxfrsize {
                    // This is the biggest seen so far.
                    maxfrsize = frsize;
                    bigfprev = fprev;
                    bigf = f;
                }
                fprev = f;

                f = sibling[f as usize];
            }
            debug_assert!(bigf != EMPTY);

            let fnext = sibling[bigf as usize];

            debug1_print!(
                "bigf {} maxfrsize {} bigfprev {} fnext {} fprev {}\n",
                bigf,
                maxfrsize,
                bigfprev,
                fnext,
                fprev
            );

            if fnext != EMPTY {
                // If fnext is EMPTY then bigf is already at the end of list.

                if bigfprev == EMPTY {
                    // Delete bigf from the element of the list.
                    child[i] = fnext;
                } else {
                    // Delete bigf from the middle of the list.
                    sibling[bigfprev as usize] = fnext;
                }

                // Put bigf at the end of the list.
                sibling[bigf as usize] = EMPTY;

                debug_assert!(child[i] != EMPTY);
                debug_assert!(fprev != bigf);
                debug_assert!(fprev != EMPTY);

                sibling[fprev as usize] = bigf;
            }

            {
                debug1_print!("After partial sort, element {}\n", i);
                let mut f = child[i];
                while f != EMPTY {
                    debug_assert!(f >= 0 && f < nn as isize);
                    debug1_print!("        {}  {}\n", f, f_size[f as usize]);
                    debug_assert!(nv[f as usize] > 0);
                    nchild -= 1;

                    f = sibling[f as usize];
                }
                debug_assert!(nchild == 0);
            }
        }
    }

    // Postorder the assembly tree.
    for i in 0..nn {
        order[i] = EMPTY;
    }

    let mut k = 0;
    let mut stack: Vec<isize> = vec![0; nn];
    for i in 0..nn {
        if parent[i] == EMPTY && nv[i] > 0 {
            debug1_print!("Root of assembly tree {}\n", i);
            k = post_tree(i, k, &mut child, &sibling, &mut order, &mut stack, nn);
        }
    }

    order
}
