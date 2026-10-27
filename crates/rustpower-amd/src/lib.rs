#![doc = include_str!("../POWERFLOW_API.md")]
/// Copyright (c) 1996-2015 Timothy A. Davis, Patrick R. Amestoy and Iain S. Duff.
/// Copyright (c) 2011-2021 Richard Lincoln.
/// All Rights Reserved.
extern crate num_traits;

mod aat;
mod amd;
mod amd_1;
mod amd_2;
mod control;
mod dump;
mod info;
mod internal;
mod post_tree;
mod postorder;
pub mod powerflow;
mod preprocess;
mod valid;
mod work;
mod ybus;
pub use ybus::{order_ybus, order_ybus_unchecked, ybus_is_eligible};

use num_traits::{NumAssignOps, PrimInt};

pub use crate::amd::*;
use aat::aat;
use amd_1::amd_1;
pub use control::control;
pub use info::info;
use internal::*;
use preprocess::preprocess;

use std::fmt::Display;
use valid::valid;

fn order_impl<I: PrimInt + NumAssignOps + Display, const FAST: bool>(
    n: I,
    a_p: &[I],
    a_i: &[I],
    control: &Control,
) -> Result<(Vec<I>, Vec<I>, Info), Status> {
    let un = n.to_usize().unwrap();

    let mut info = Info::new(un);

    if n == I::zero() {
        let p = Vec::new();
        let p_inv = Vec::new();
        return Ok((p, p_inv, info));
    }
    let nz = a_p[un].to_usize().unwrap();
    info.nz = nz;

    // Check the input matrix.
    let status = valid(n, n, a_p, a_i);
    if status == Status::Invalid {
        return Err(Status::Invalid);
    }

    let (c_p, c_i) = if status == Status::OkButJumbled {
        // Sort the input matrix and remove duplicate entries.
        debug1_println!("Matrix is jumbled");
        preprocess(n, a_p, a_i) // R = A'.
    } else {
        // Order the input matrix as-is. No need to compute R = A' first.
        (a_p.to_vec(), a_i.to_vec())
    };

    // Determine the symmetry and count off-diagonal nonzeros in A+A'.
    let (nzaat, mut len) = aat(n, &c_p, &c_i, &mut info);
    debug1_print!("nzaat: {}\n", nzaat);
    let clean_nz = c_p[un].to_usize().unwrap();
    debug_assert!((clean_nz.saturating_sub(un) <= nzaat) && (nzaat <= 2 * clean_nz));

    let iwlen = nzaat + (nzaat / 5) + un; // Space for matrix + elbow room.
    debug1_print!("iwlen {}\n", iwlen);

    // Order the matrix.
    let (p, p_inv) = amd_1::<I, FAST>(n, &c_p, &c_i, &mut len, iwlen, control, &mut info);

    info.status = status;

    Ok((p, p_inv, info))
}

/// Order a validated CSC matrix using bounds-checked native Rust AMD.
pub fn order(
    n: usize,
    p: &[usize],
    i: &[usize],
    control: &Control,
) -> Result<(Vec<usize>, Vec<usize>, Info), Status> {
    validate_csc(n, p, i)?;
    order_impl::<usize, false>(n, p, i, control)
}
/// Same algorithm, with unchecked indexing in the AMD elimination workspace.
/// # Safety
/// Experimental: caller must ensure valid CSC and all AMD workspace index invariants.
/// Use the checked variant to diagnose invariant violations before benchmarking.
pub unsafe fn order_unchecked(
    n: usize,
    p: &[usize],
    i: &[usize],
    control: &Control,
) -> Result<(Vec<usize>, Vec<usize>, Info), Status> {
    validate_csc(n, p, i)?;
    order_impl::<usize, true>(n, p, i, control)
}
fn validate_csc(n: usize, p: &[usize], i: &[usize]) -> Result<(), Status> {
    if n > isize::MAX as usize / 128
        || p.len() != n.checked_add(1).ok_or(Status::Invalid)?
        || p.first() != Some(&0)
        || p.last() != Some(&i.len())
        || p.windows(2).any(|x| x[0] > x[1])
        || i.iter().any(|&v| v >= n)
    {
        return Err(Status::Invalid);
    }
    Ok(())
}

#[cfg(feature = "trace")]
pub mod trace;

pub mod exact;

mod lazy_core;
pub mod on_demand;

#[cfg(feature = "klu")]
pub mod klu;
#[cfg(feature = "klu")]
mod klu_rust;
