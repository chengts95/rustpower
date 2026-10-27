use crate::amd::{Info, Status};

pub fn info(info: &Info) {
    println!("\nAMD results:");

    let n = info.n;
    let ndiv = info.n_div;
    let nmultsubs_ldl = info.n_mult_subs_ldl;
    let nmultsubs_lu = info.n_mult_subs_lu;
    let lnz = info.lnz;
    let lnzd = n + lnz;
    // let lnzd = if n >= 0 && lnz >= 0 {
    //     n + lnz
    // } else {
    //     0 /*-1*/
    // };

    // AMD return status.
    print!("    status:                                             ");
    if info.status == Status::OK {
        println!("OK");
    } else if info.status == Status::Invalid {
        println!("invalid matrix");
    } else if info.status == Status::OkButJumbled {
        println!("OK, but jumbled");
    } else {
        println!("unknown");
    }

    // Statistics about the input matrix.
    println!(
        "    n, dimension of A:                                  {}",
        n
    );
    println!(
        "    nz, number of nonzeros in A:                        {}",
        info.nz
    );
    println!(
        "    symmetry of A:                                      {}",
        info.symmetry
    );
    println!(
        "    number of nonzeros on diagonal:                     {}",
        info.nz_diag
    );
    println!(
        "    nonzeros in pattern of A+A' (excl. diagonal):       {}",
        info.nz_a_plus_at
    );
    println!(
        "    # dense rows/columns of A+A':                       {}",
        info.n_dense
    );

    // Statistics about AMD's behavior.
    // println!("    memory used, in bytes:                              {}", info.memory);
    println!(
        "    # of memory compactions:                            {}",
        info.n_cmp_a
    );

    // Statistics about the ordering quality.
    println!(
        "
    The following approximate statistics are for a subsequent
    factorization of A(P,P) + A(P,P)'.  They are slight upper
    bounds if there are no dense rows/columns in A+A', and become
    looser if dense rows/columns exist.\n"
    );

    println!(
        "    nonzeros in L (excluding diagonal):                 {}",
        lnz
    );
    println!(
        "    nonzeros in L (including diagonal):                 {}",
        lnzd
    );
    println!(
        "    # divide operations for LDL' or LU:                 {}",
        ndiv
    );
    println!(
        "    # multiply-subtract operations for LDL':            {}",
        nmultsubs_ldl
    );
    println!(
        "    # multiply-subtract operations for LU:              {}",
        nmultsubs_lu
    );
    println!(
        "    max nz. in any column of L (incl. diagonal):        {}",
        info.d_max
    );

    // Total flop counts for various factorizations.

    // if n >= 0 && ndiv >= 0 && nmultsubs_ldl >= 0 && nmultsubs_lu >= 0 {
    println!(
        "
    chol flop count for real A, sqrt counted as 1 flop: {}
    LDL' flop count for real A:                         {}
    LDL' flop count for complex A:                      {}
    LU flop count for real A (with no pivoting):        {}
    LU flop count for complex A (with no pivoting):     {}",
        n + ndiv + 2 * nmultsubs_ldl,
        ndiv + 2 * nmultsubs_ldl,
        9 * ndiv + 8 * nmultsubs_ldl,
        ndiv + 2 * nmultsubs_lu,
        9 * ndiv + 8 * nmultsubs_lu
    );
    println!();
    // }
}
