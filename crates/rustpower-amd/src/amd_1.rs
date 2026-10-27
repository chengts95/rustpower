use crate::amd::*;
use crate::amd_2::amd_2;
use crate::internal::*;
use crate::valid::valid;
use num_traits::PrimInt;
use std::fmt::Display;

pub fn amd_1<I: PrimInt + Display, const FAST: bool>(
    n: I,
    a_p: &[I],
    a_i: &[I],
    mut len: &mut [usize],
    iwlen: usize,
    control: &Control,
    mut info: &mut Info,
) -> (Vec<I>, Vec<I>) {
    let un = n.to_usize().unwrap();

    // Construct the matrix for amd_2.

    debug_assert!(un > 0);

    let mut p_e: Vec<isize> = vec![0; un];
    let mut s_p: Vec<usize> = vec![0; un];
    let mut t_p: Vec<usize> = vec![0; un];
    let mut i_w: Vec<isize> = vec![0; iwlen];

    debug_assert!(valid(n, n, a_p, a_i) == Status::OK);

    // Construct the pointers for A+A'.

    let mut pfree: usize = 0;
    for j in 0..un {
        p_e[j] = pfree as isize;
        s_p[j] = pfree;
        pfree += len[j];
    }

    // Note that this restriction on iwlen is slightly more restrictive than
    // what is strictly required in amd_2. amd_2 can operate with no elbow
    // room at all, but it will be very slow. For better performance, at
    // least size-n elbow room is enforced.
    debug_assert!(iwlen >= pfree + un);

    #[cfg(feature = "debug1")]
    for p in 0..iwlen {
        i_w[p] = EMPTY;
    }

    for k in 0..un {
        debug1_print!("Construct row/column k = {} of A+A'\n", k);
        let p1 = a_p[k].to_usize().unwrap();
        let p2 = a_p[k + 1].to_usize().unwrap();

        // Construct A+A'.
        let mut p = p1;
        while p < p2 {
            // Scan the upper triangular part of A.
            let j = a_i[p].to_usize().unwrap();
            debug_assert!(/*j >= 0 &&*/ j < un);
            if j < k {
                // Entry A(j,k) in the strictly upper triangular part.
                #[cfg(feature = "debug1")]
                if j == (un - 1) {
                    debug_assert!(s_p[j] < pfree);
                } else {
                    debug_assert!((s_p[j] as isize) < p_e[j + 1]);
                }
                #[cfg(feature = "debug1")]
                if k == (un - 1) {
                    debug_assert!(s_p[k] < pfree);
                } else {
                    debug_assert!((s_p[k] as isize) < p_e[k + 1]);
                }

                i_w[s_p[j]] = k as isize;
                s_p[j] += 1;

                i_w[s_p[k]] = j as isize;
                s_p[k] += 1;

                p += 1;
            } else if j == k {
                // Skip the diagonal.
                p += 1;
                break;
            } else {
                // j > k
                // First entry below the diagonal.
                break;
            }

            // Scan lower triangular part of A, in column j until reaching
            // row k. Start where last scan left off.
            debug_assert!(
                a_p[j].to_usize().unwrap() <= t_p[j] && t_p[j] <= a_p[j + 1].to_usize().unwrap()
            );
            let mut pj = t_p[j];
            let pj2 = a_p[j + 1].to_usize().unwrap();
            while pj < pj2 {
                let i = a_i[pj].to_usize().unwrap();
                debug_assert!(/*i >= 0 &&*/ i < un);
                if i < k {
                    // A (i,j) is only in the lower part, not in upper.
                    #[cfg(feature = "debug1")]
                    if i == un - 1 {
                        debug_assert!(s_p[i] < pfree);
                    } else {
                        debug_assert!((s_p[i] as isize) < p_e[i + 1]);
                    }
                    #[cfg(feature = "debug1")]
                    if j == un - 1 {
                        debug_assert!(s_p[j] < pfree);
                    } else {
                        debug_assert!((s_p[j] as isize) < p_e[j + 1]);
                    }

                    i_w[s_p[i]] = j as isize;
                    s_p[i] += 1;

                    i_w[s_p[j]] = i as isize;
                    s_p[j] += 1;

                    pj += 1;
                } else if i == k {
                    // Entry A(k,j) in lower part and A(j,k) in upper.
                    pj += 1;
                    break;
                } else {
                    // i > k
                    // Consider this entry later, when k advances to i.
                    break;
                }
            }
            t_p[j] = pj;
        }
        t_p[k] = p;
    }

    // Clean up, for remaining mismatched entries.
    for j in 0..un {
        let p2 = a_p[j + 1].to_usize().unwrap();
        for pj in t_p[j]..p2 {
            let i = a_i[pj].to_usize().unwrap();

            debug_assert!(/*i >= 0 &&*/ i < un);
            // A(i,j) is only in the lower part, not in upper.
            #[cfg(feature = "debug1")]
            if i == un - 1 {
                debug_assert!(s_p[i] < pfree);
            } else {
                debug_assert!((s_p[i] as isize) < p_e[i + 1]);
            }
            #[cfg(feature = "debug1")]
            if j == un - 1 {
                debug_assert!(s_p[j] < pfree);
            } else {
                debug_assert!((s_p[j] as isize) < p_e[j + 1]);
            }

            i_w[s_p[i]] = j as isize;
            s_p[i] += 1;

            i_w[s_p[j]] = i as isize;
            s_p[j] += 1;
        }
    }

    #[cfg(feature = "debug1")]
    for j in 0..un - 1 {
        debug_assert!((s_p[j] as isize) == p_e[j + 1])
    }
    debug_assert!(s_p[un - 1] == pfree);

    // Tp and Sp no longer needed.

    // Order the matrix.
    let (_nv, p_inv, p, _e_len) = amd_2::<I, FAST, false>(
        n, &mut p_e, &mut i_w, &mut len, iwlen, pfree, control, &mut info, None, None,
    );

    (p, p_inv)
}
