#[cfg(feature = "debug1")]
use crate::internal::*;

#[cfg(feature = "debug1")]
pub fn dump(
    n: usize,
    pe: &[isize],
    iw: &[isize],
    len: &[usize],
    iwlen: usize,
    pfree: usize,
    nv: &[isize],
    next: &[isize],
    last: &[isize],
    head: &[isize],
    e_len: &[isize],
    degree: &[usize],
    w: &[usize],
    nel: isize,
) {
    debug_assert!(pfree <= iwlen);
    debug3_print!("\nAMD dump, pfree: {}\n", pfree);
    for i in 0..n {
        let pei = pe[i];
        let elen = e_len[i];
        let nvi = nv[i];
        let leni = len[i];
        let wi = w[i];

        if elen >= EMPTY {
            if nvi == 0 {
                debug3_print!("\nI {}: nonprincipal:    \n", i);
                debug_assert!(elen == EMPTY);
                if pei == EMPTY {
                    debug3_println!(" dense node");
                    debug_assert!(wi == 1);
                } else {
                    debug_assert!(pei < EMPTY);
                    debug3_print!(" i {} -> parent {}\n", i, flip(pe[i]));
                }
            } else {
                debug3_print!("\nI {}: active principal supervariable:\n", i);
                debug3_print!("   nv(i): {}  Flag: {}\n", nvi, nvi < 0);
                debug_assert!(elen >= 0);
                debug_assert!(nvi > 0 && pei >= 0);
                let mut p = pei;
                debug3_print!("   e/s: ");
                if elen == 0 {
                    debug3_print!(" : ");
                }
                debug_assert!(pei as usize + leni <= pfree);
                for k in 0..leni {
                    let j = iw[p as usize];
                    debug3_print!("  {}", j);
                    debug_assert!(j >= 0 && j < n as isize);
                    if k as isize == elen - 1 {
                        debug3_print!(" : ");
                    }
                    p += 1;
                }
                debug3_println!();
            }
        } else {
            #[cfg(feature = "debug3")]
            let e = i;
            if wi == 0 {
                debug3_print!("\nE {}: absorbed element: w {}\n", e, wi);
                debug_assert!(nvi > 0 && pei < 0);
                debug3_print!(" e {} -> parent {}\n", e, flip(pe[e]));
            } else {
                debug3_print!("\nE {}: unabsorbed element: w {}\n", e, wi);
                debug_assert!(nvi > 0 && pei >= 0);
                let mut p = pei;
                debug3_print!(" : ");
                debug_assert!(pei as usize + leni <= pfree);
                for _k in 0..leni {
                    let j = iw[p as usize];
                    debug3_print!("  {}", j);
                    debug_assert!(j >= 0 && j < n as isize);
                    p += 1;
                }
                debug3_println!();
            }
        }
    }

    // This routine cannot be called when the hash buckets are non-empty.
    debug3_println!("\nDegree lists:");
    if nel >= 0 {
        let mut cnt: isize = 0;
        for deg in 0..n {
            if head[deg] == EMPTY {
                continue;
            }
            let mut ilast = EMPTY;
            debug3_print!("{}: \n", deg);
            let mut i = head[deg];
            while i != EMPTY {
                debug3_print!(
                    "   {} : next {} last {} deg {}\n",
                    i,
                    next[i as usize],
                    last[i as usize],
                    degree[i as usize]
                );
                debug_assert!(
                    i >= 0
                        && i < n as isize
                        && ilast == last[i as usize]
                        && deg == degree[i as usize]
                );
                cnt += nv[i as usize];
                ilast = i;

                i = next[i as usize];
            }
            debug3_print!("\n");
        }
        debug_assert!(cnt == n as isize - nel);
    }
}
