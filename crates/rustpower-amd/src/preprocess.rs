use crate::amd::*;
use crate::internal::*;
use crate::valid::*;
use num_traits::{NumAssignOps, PrimInt};
use std::cmp::max;

pub fn preprocess<I: PrimInt + NumAssignOps>(n: I, a_p: &[I], a_i: &[I]) -> (Vec<I>, Vec<I>) {
    let un = n.to_usize().unwrap();

    debug_assert!(valid(n, n, a_p, a_i) != Status::Invalid);

    let mut w: Vec<I> = vec![I::zero(); un];
    let mut flag: Vec<isize> = vec![0; un];

    // Count the entries in each row of A (excluding duplicates).

    for i in 0..un {
        w[i] = I::zero(); // # of nonzeros in row i (excl duplicates)
        flag[i] = EMPTY; // flag[i] = j if i appears in column j.
    }
    for j in 0..un {
        let p1 = a_p[j].to_usize().unwrap();
        let p2 = a_p[j + 1].to_usize().unwrap();
        for p in p1..p2 {
            let i = a_i[p].to_usize().unwrap();
            if flag[i] != j as isize {
                // Row index i has not yet appeared in column j.
                w[i] += I::one(); // One more entry in row i.
                flag[i] = j as isize; // Flag row index i as appearing in col j.
            }
        }
    }

    // Compute the row pointers for R.

    let nz: usize = a_p[un].to_usize().unwrap();
    let mut r_p: Vec<I> = vec![I::zero(); un + 1];
    let mut r_i: Vec<I> = vec![I::zero(); max(nz, 1)];

    r_p[0] = I::zero();
    for i in 0..un {
        r_p[i + 1] = r_p[i] + w[i];
    }
    for i in 0..un {
        w[i] = r_p[i];
        flag[i] = EMPTY
    }

    // Construct the row form matrix R.

    // R = row form of pattern of A.
    for j in 0..un {
        let p1 = a_p[j].to_usize().unwrap();
        let p2 = a_p[j + 1].to_usize().unwrap();
        for p in p1..p2 {
            let i = a_i[p].to_usize().unwrap();
            if flag[i] != j as isize {
                // Row index i has not yet appeared in column j.
                r_i[w[i].to_usize().unwrap()] = I::from(j).unwrap(); // Put col j in row i.
                w[i] += I::one();
                flag[i] = j as isize; // Flag row index i as appearing in col j.
            }
        }
    }

    debug_assert!(valid(n, n, &r_p, &r_i) == Status::OK);
    for j in 0..un {
        debug_assert!(w[j] == r_p[j + 1])
    }

    (r_p, r_i)
}
