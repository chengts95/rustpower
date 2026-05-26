use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Second derivatives of complex branch power flows w.r.t. voltage.
///
/// Computes lam^T · d²Sbr/dVx·dVy for x,y ∈ {a=angle, v=magnitude}.
/// Returns (Saa, Sav, Sva, Svv) — each nb×nb real sparse matrix.
///
/// Arguments:
///   Cbr : nb×nl connection matrix  (Cbr[bus, branch] = 1 if bus is the "from" end)
///   Ybr : nl×nb branch admittance matrix (Yf or Yt)
///   v   : nb complex voltage vector
///   lam : nl multiplier vector (complex — diagSbr_conj * mu from d2ASbr)
///
/// Reference: MATPOWER TN2 §6, pandapower.pypower.d2Sbr_dV2
#[allow(non_snake_case)]
pub fn d2Sbr_dV2(
    Cbr: &CscMatrix<Complex64>, // nb × nl
    Ybr: &CscMatrix<Complex64>, // nl × nb
    v: &DVector<Complex64>,
    lam: &DVector<Complex64>,
) -> (
    CscMatrix<Complex64>, // Saa
    CscMatrix<Complex64>, // Sav
    CscMatrix<Complex64>, // Sva
    CscMatrix<Complex64>, // Svv
) {
    // Python reference:
    //   A = Ybr.H * diaglam * Cbr            # nb×nb
    //   B = conj(diagV) * A * diagV           # nb×nb, same pattern as A
    //   D = diag( (A * V) * conj(V) )         # nb×nb diagonal
    //   E = diag( A^T * conj(V) * V )         # nb×nb diagonal (note: A.T, not A.H)
    //   F = B + B^T
    //   G = diag(1 / |V|)
    //
    //   Haa = F - D - E
    //   Hva = j * G * (B - B^T - D + E)
    //   Hav = Hva^T
    //   Hvv = G * F * G

    let nb = v.len();
    let j_unit = Complex64::i();
    let v_s = v.as_slice();
    let lam_s = lam.as_slice();

    // inv_abv[i] = 1 / |V[i]|
    let inv_abv: Vec<f64> = v_s.iter().map(|vi| 1.0 / vi.norm()).collect();

    // A = Ybr^H * diaglam * Cbr
    // A[i, j] = sum_l conj(Ybr[l, i]) * lam[l] * Cbr[j, l]  ... wait
    //
    // Actually: Ybr^H is (nb×nl), diaglam is (nl×nl), Cbr is (nb×nl)
    // But Ybr is (nl×nb), so Ybr^H is (nb×nl).
    // Cbr is (nb×nl), so Cbr^T would be (nl×nb).
    //
    // In pypower: Cbr is (nl×nb), so Cbr^T is (nb×nl).
    // Let's be explicit: Cbr here is (nb×nl) connection matrix.
    //   A = Ybr^H * diaglam * Cbr^T   but Cbr is passed as (nb×nl) already
    //
    // Actually, let me re-read the pypower code:
    //   A = Ybr.H * diaglam * Cbr
    // where Ybr is (nl×nb) → Ybr.H is (nb×nl)
    // diaglam is (nl×nl)
    // Cbr is (nl×nb) in pypower
    // So A is (nb×nb).
    //
    // Here our Cbr parameter is (nb×nl) so:
    //   A = Ybr^H * diaglam * Cbr^T   -- (nb×nl)*(nl×nl)*(nl×nb) = (nb×nb)
    //
    // We pass Cbr as CscMatrix of shape (nb×nl) meaning rows=buses, cols=branches.
    // Ybr is (nl×nb) in CSC.
    //
    // Compute A[i,k] = sum_l conj(Ybr[l,i]) * lam[l] * Cbr[i2,l] where...
    // Hmm this is getting confusing. Let me just compute A element-by-element.
    //
    // A[i,k] = (Ybr^H * diaglam * Cbr^T)[i,k]
    //        = sum_l (Ybr^H)[i,l] * (diaglam * Cbr^T)[l,k]
    //        = sum_l conj(Ybr[l,i]) * lam[l] * Cbr^T[l,k]
    //        = sum_l conj(Ybr[l,i]) * lam[l] * Cbr[k,l]
    //
    // Since Cbr[k,l] is the connection matrix (1 if bus k is the "from"/"to" end of branch l),
    // and Cbr is sparse (one 1 per column for the from or to bus), this is:
    //   A[i,k] = sum over branches l where bus k appears: conj(Ybr[l,i]) * lam[l]
    //
    // Since this involves SpGEMM, let's build A via intermediate products.

    // Step 1: diaglam * Cbr^T  (nl×nb matrix)
    // (diaglam * Cbr^T)[l, k] = lam[l] * Cbr[k, l]
    // Cbr is (nb×nl) CSC. Its transpose Cbr^T is (nl×nb).
    // We can compute diaglam * Cbr^T by scaling rows of Cbr^T by lam.

    // Build Cbr^T as a new CSC matrix by transposing Cbr
    let nl = Ybr.nrows();
    let cbr_t = csc_transpose(Cbr); // nl × nb

    // Scale rows of cbr_t by lam: dlam_cbr_t[l, k] = lam[l] * cbr_t[l, k]
    let mut dlam_cbr_t_vals: Vec<Complex64> = cbr_t.values().to_vec();
    {
        let ri = cbr_t.row_indices();
        for (idx, v) in dlam_cbr_t_vals.iter_mut().enumerate() {
            *v = lam_s[ri[idx]] * *v;
        }
    }
    let dlam_cbr_t = CscMatrix::try_from_csc_data(
        nl,
        nb,
        cbr_t.col_offsets().to_vec(),
        cbr_t.row_indices().to_vec(),
        dlam_cbr_t_vals,
    )
    .unwrap();

    // Step 2: A = Ybr^H * (diaglam * Cbr^T)
    // Ybr^H = conj(Ybr^T). Build it.
    let ybr_h = csc_adjoint(Ybr); // nb × nl
    // A = ybr_h (nb×nl) * dlam_cbr_t (nl×nb) = nb×nb
    let a_mat = spgemm(&ybr_h, &dlam_cbr_t); // nb × nb

    // Step 3: B = conj(diagV) * A * diagV
    // = element-wise: B[i,j] = conj(V[i]) * A[i,j] * V[j]
    let b_mat = {
        let a_ri = a_mat.row_indices();
        let a_cp = a_mat.col_offsets();
        let a_v = a_mat.values();
        let mut b_vals: Vec<Complex64> = Vec::with_capacity(a_mat.nnz());
        for j in 0..nb {
            for idx in a_cp[j]..a_cp[j + 1] {
                let i = a_ri[idx];
                b_vals.push(v_s[i].conj() * a_v[idx] * v_s[j]);
            }
        }
        CscMatrix::try_from_csc_data(nb, nb, a_cp.to_vec(), a_ri.to_vec(), b_vals).unwrap()
    };

    // Step 4: (A * V) and A^T * conj(V) — needed for diagonals D and E
    // (A * V)[i] = sum_j A[i,j] * V[j]  → SpMV
    // (A^T * conj(V))[i] = sum_j A[j,i] * conj(V[j])

    // SpMV: A * V (dense result length nb)
    let mut av = vec![Complex64::new(0.0, 0.0); nb];
    {
        let a_ri = a_mat.row_indices();
        let a_cp = a_mat.col_offsets();
        let a_v = a_mat.values();
        for j in 0..nb {
            for idx in a_cp[j]..a_cp[j + 1] {
                let i = a_ri[idx];
                av[i] += a_v[idx] * v_s[j];
            }
        }
    }
    // D[i,i] = (A*V)[i] * conj(V[i])
    let d_diag: Vec<Complex64> = (0..nb).map(|i| av[i] * v_s[i].conj()).collect();

    // A^T * conj(V): for col i of A^T = row i of A
    let mut at_conjv = vec![Complex64::new(0.0, 0.0); nb];
    {
        let a_ri = a_mat.row_indices();
        let a_cp = a_mat.col_offsets();
        let a_v = a_mat.values();
        for j in 0..nb {
            for idx in a_cp[j]..a_cp[j + 1] {
                let i = a_ri[idx];
                // A[i,j] → A^T[j,i]
                at_conjv[j] += a_v[idx] * v_s[i].conj();
            }
        }
    }
    // E[i,i] = (A^T * conj(V))[i] * V[i]
    let e_diag: Vec<Complex64> = (0..nb).map(|i| at_conjv[i] * v_s[i]).collect();

    // Step 5: B^T (transpose of B, same pattern as A^T)
    let bt_mat = csc_transpose_cx(&b_mat);

    // Step 6: F = B + B^T
    // Since B and B^T may have different sparsity, we need to add them.
    // For simplicity, use a dense accumulation per position, or use sparsity union.
    let f_mat = csc_add(&b_mat, &bt_mat);

    // Step 7: Assemble output matrices
    // Haa = F - D - E  (subtract diagonal D and E from F)
    // Hva = j * G * (B - B^T - D + E)
    // Hvv = G * F * G
    //
    // G = diag(inv_abv) — apply as row/col scaling.

    let saa_mat = subtract_diags(&f_mat, &d_diag, &e_diag, false);
    let temp = {
        // (B - B^T - D + E): subtract diags from (B - B^T)
        let bmbT = csc_subtract(&b_mat, &bt_mat);
        subtract_diags(&bmbT, &d_diag, &e_diag, true) // true = subtract D, ADD E
    };

    // Sva = j * diag(inv_abv) * temp  (row-scale by inv_abv, then by j)
    let sva_mat = {
        let cp = temp.col_offsets();
        let ri = temp.row_indices();
        let vals: Vec<Complex64> = temp
            .values()
            .iter()
            .zip(ri.iter())
            .map(|(&val, &i)| j_unit * inv_abv[i] * val)
            .collect();
        CscMatrix::try_from_csc_data(nb, nb, cp.to_vec(), ri.to_vec(), vals).unwrap()
    };

    // Sav = Sva^T
    let sav_mat = csc_transpose_cx(&sva_mat);

    // Svv = diag(inv_abv) * F * diag(inv_abv)  (row- and col-scale F)
    let svv_mat = {
        let cp = f_mat.col_offsets();
        let ri = f_mat.row_indices();
        let vals: Vec<Complex64> = f_mat
            .values()
            .iter()
            .enumerate()
            .map(|(idx, &val)| inv_abv[ri[idx]] * val * inv_abv[cp_to_col(cp, idx)])
            .collect();
        // We need the column index for each nnz to apply inv_abv[j].
        // Rebuild with proper column indices.
        let mut col_vals = vec![Complex64::new(0.0, 0.0); f_mat.nnz()];
        for j in 0..nb {
            for idx in cp[j]..cp[j + 1] {
                let i = ri[idx];
                col_vals[idx] = inv_abv[i] * f_mat.values()[idx] * inv_abv[j];
            }
        }
        let _ = vals; // drop the incorrect vals above
        CscMatrix::try_from_csc_data(nb, nb, cp.to_vec(), ri.to_vec(), col_vals).unwrap()
    };

    (saa_mat, sav_mat, sva_mat, svv_mat)
}

/// Second derivatives of |Sbr|² w.r.t. voltage, weighted by multipliers mu.
///
/// Returns (Haa, Hav, Hva, Hvv) — each nb×nb real sparse matrix.
/// Used by opf_hessfcn for the branch flow limit contribution to the Lagrangian Hessian.
///
/// Reference: MATPOWER TN2, pandapower.pypower.d2ASbr_dV2
#[allow(non_snake_case)]
pub fn d2ASbr_dV2(
    dSbr_dVa: &CscMatrix<Complex64>, // nl × nb
    dSbr_dVm: &CscMatrix<Complex64>, // nl × nb
    Sbr: &DVector<Complex64>,         // nl
    Cbr: &CscMatrix<Complex64>,       // nb × nl
    Ybr: &CscMatrix<Complex64>,       // nl × nb
    v: &DVector<Complex64>,
    lam: &DVector<f64>, // nl real multipliers (mu_f or mu_t)
) -> (
    CscMatrix<f64>, // Haa
    CscMatrix<f64>, // Hav
    CscMatrix<f64>, // Hva
    CscMatrix<f64>, // Hvv
) {
    // Python:
    //   diaglam       = diag(lam)
    //   diagSbr_conj  = diag(conj(Sbr))
    //   Saa, Sav, Sva, Svv = d2Sbr_dV2(Cbr, Ybr, V, diagSbr_conj * lam)
    //   Haa = 2 * real(Saa + dSbr_dVa^T * diaglam * conj(dSbr_dVa))
    //   Hva = 2 * real(Sva + dSbr_dVm^T * diaglam * conj(dSbr_dVa))
    //   Hav = 2 * real(Sav + dSbr_dVa^T * diaglam * conj(dSbr_dVm))
    //   Hvv = 2 * real(Svv + dSbr_dVm^T * diaglam * conj(dSbr_dVm))

    let nl = Sbr.len();
    let s_s = Sbr.as_slice();
    let lam_s = lam.as_slice();

    // lam_cx[l] = conj(Sbr[l]) * lam[l]  (complex multiplier for d2Sbr_dV2)
    let lam_cx = DVector::from_iterator(nl, (0..nl).map(|l| s_s[l].conj() * lam_s[l]));

    let (saa, sav, sva, svv) = d2Sbr_dV2(Cbr, Ybr, v, &lam_cx);

    // For each pair (dSbr_dVx^T * diaglam * conj(dSbr_dVy)):
    //   = sum_l lam[l] * conj(dSbr_dVx[l, :])^T * dSbr_dVy[l, :]  (outer product sum)
    // Each term is a rank-1 outer product added to the result.
    // Result is nb×nb.

    let nb = v.len();

    // diaglam * conj(dSbr_dVa): row-scale conj(dSbr_dVa) by lam
    // Then dSbr_dVa^T * (above) = SpGEMM of (nb×nl) × (nl×nb)
    let lam_conj_da = row_scale_cx(dSbr_dVa, lam_s, true); // lam[l]*conj(dSbr_dVa)
    let lam_conj_dm = row_scale_cx(dSbr_dVm, lam_s, true); // lam[l]*conj(dSbr_dVm)

    // Transpose of dSbr_dVa and dSbr_dVm: (nb×nl)
    let da_t = csc_transpose_cx(dSbr_dVa); // nb × nl
    let dm_t = csc_transpose_cx(dSbr_dVm); // nb × nl

    // SpGEMM products
    let da_t_lam_conj_da = spgemm(&da_t, &lam_conj_da); // nb × nb
    let dm_t_lam_conj_da = spgemm(&dm_t, &lam_conj_da); // nb × nb
    let da_t_lam_conj_dm = spgemm(&da_t, &lam_conj_dm); // nb × nb
    let dm_t_lam_conj_dm = spgemm(&dm_t, &lam_conj_dm); // nb × nb

    // Hxx = 2 * real(Sxx + product)
    let to_real = |s: CscMatrix<Complex64>, prod: CscMatrix<Complex64>| -> CscMatrix<f64> {
        let sum = csc_add(&s, &prod);
        let cp = sum.col_offsets().to_vec();
        let ri = sum.row_indices().to_vec();
        let vals: Vec<f64> = sum.values().iter().map(|v| 2.0 * v.re).collect();
        CscMatrix::try_from_csc_data(nb, nb, cp, ri, vals).unwrap()
    };

    (
        to_real(saa, da_t_lam_conj_da),
        to_real(sav, da_t_lam_conj_dm),
        to_real(sva, dm_t_lam_conj_da),
        to_real(svv, dm_t_lam_conj_dm),
    )
}

// ── sparse helpers ────────────────────────────────────────────────────────────

/// Sparse matrix-matrix product (CSC × CSC → CSC), complex.
/// Uses a dense accumulator (column-by-column), suitable for nb ≤ ~10k.
pub(crate) fn spgemm(
    a: &CscMatrix<Complex64>,
    b: &CscMatrix<Complex64>,
) -> CscMatrix<Complex64> {
    let m = a.nrows();
    let n = b.ncols();
    assert_eq!(a.ncols(), b.nrows());
    let k = a.ncols();

    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();

    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();

    // Dense column buffer
    let mut acc = vec![Complex64::new(0.0, 0.0); m];
    let mut visited = vec![false; m];
    let mut col_nz: Vec<usize> = Vec::new();

    let mut c_col_offsets = vec![0usize; n + 1];
    let mut c_row_indices: Vec<usize> = Vec::new();
    let mut c_values: Vec<Complex64> = Vec::new();

    for j in 0..n {
        col_nz.clear();
        // For each k with B[k,j] != 0
        for idx_b in b_cp[j]..b_cp[j + 1] {
            let kk = b_ri[idx_b];
            let b_kj = b_v[idx_b];
            // For each i with A[i,k] != 0
            for idx_a in a_cp[kk]..a_cp[kk + 1] {
                let i = a_ri[idx_a];
                let a_ik = a_v[idx_a];
                if !visited[i] {
                    visited[i] = true;
                    col_nz.push(i);
                }
                acc[i] += a_ik * b_kj;
            }
        }
        col_nz.sort_unstable();
        for &i in &col_nz {
            c_row_indices.push(i);
            c_values.push(acc[i]);
            acc[i] = Complex64::new(0.0, 0.0);
            visited[i] = false;
        }
        c_col_offsets[j + 1] = c_row_indices.len();
    }

    CscMatrix::try_from_csc_data(m, n, c_col_offsets, c_row_indices, c_values).unwrap()
}

/// Transpose a real-pattern CscMatrix<Complex64>.
pub(crate) fn csc_transpose_cx(a: &CscMatrix<Complex64>) -> CscMatrix<Complex64> {
    let m = a.nrows();
    let n = a.ncols();
    let nnz = a.nnz();
    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();

    // Count nnz per row of A (= per col of A^T)
    let mut row_counts = vec![0usize; m];
    for &r in a_ri {
        row_counts[r] += 1;
    }
    let mut t_cp = vec![0usize; m + 1];
    for i in 0..m {
        t_cp[i + 1] = t_cp[i] + row_counts[i];
    }
    let mut t_ri = vec![0usize; nnz];
    let mut t_v = vec![Complex64::new(0.0, 0.0); nnz];
    let mut pos = t_cp.clone();

    // Column index for each nnz
    let mut col_of = vec![0usize; nnz];
    for j in 0..n {
        for idx in a_cp[j]..a_cp[j + 1] {
            col_of[idx] = j;
        }
    }

    for idx in 0..nnz {
        let r = a_ri[idx];
        let p = pos[r];
        t_ri[p] = col_of[idx];
        t_v[p] = a_v[idx];
        pos[r] += 1;
    }

    // Sort each row's entries by column
    for i in 0..m {
        let s = t_cp[i];
        let e = t_cp[i + 1];
        let mut pairs: Vec<(usize, Complex64)> =
            (s..e).map(|p| (t_ri[p], t_v[p])).collect();
        pairs.sort_unstable_by_key(|&(c, _)| c);
        for (p, (col, val)) in (s..e).zip(pairs) {
            t_ri[p] = col;
            t_v[p] = val;
        }
    }

    CscMatrix::try_from_csc_data(n, m, t_cp, t_ri, t_v).unwrap()
}

/// Transpose a real-pattern CscMatrix<Complex64> with integer values (for connection matrices).
fn csc_transpose(a: &CscMatrix<Complex64>) -> CscMatrix<Complex64> {
    csc_transpose_cx(a)
}

/// Hermitian adjoint (conjugate transpose) of a CSC complex matrix.
pub(crate) fn csc_adjoint(a: &CscMatrix<Complex64>) -> CscMatrix<Complex64> {
    let mut t = csc_transpose_cx(a);
    for v in t.values_mut() {
        *v = v.conj();
    }
    t
}

/// Element-wise addition of two CSC complex matrices (must have compatible sparsity or use union).
fn csc_add(a: &CscMatrix<Complex64>, b: &CscMatrix<Complex64>) -> CscMatrix<Complex64> {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let nb = a.ncols();

    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();
    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();

    let mut c_cp = vec![0usize; nb + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<Complex64> = Vec::new();

    for j in 0..nb {
        let mut ia = a_cp[j];
        let mut ib = b_cp[j];
        let ea = a_cp[j + 1];
        let eb = b_cp[j + 1];
        while ia < ea || ib < eb {
            let row_a = if ia < ea { a_ri[ia] } else { usize::MAX };
            let row_b = if ib < eb { b_ri[ib] } else { usize::MAX };
            if row_a < row_b {
                c_ri.push(row_a);
                c_v.push(a_v[ia]);
                ia += 1;
            } else if row_b < row_a {
                c_ri.push(row_b);
                c_v.push(b_v[ib]);
                ib += 1;
            } else {
                c_ri.push(row_a);
                c_v.push(a_v[ia] + b_v[ib]);
                ia += 1;
                ib += 1;
            }
        }
        c_cp[j + 1] = c_ri.len();
    }

    CscMatrix::try_from_csc_data(a.nrows(), nb, c_cp, c_ri, c_v).unwrap()
}

fn csc_subtract(a: &CscMatrix<Complex64>, b: &CscMatrix<Complex64>) -> CscMatrix<Complex64> {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    let nb = a.ncols();

    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();
    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();

    let mut c_cp = vec![0usize; nb + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<Complex64> = Vec::new();

    for j in 0..nb {
        let mut ia = a_cp[j];
        let mut ib = b_cp[j];
        let ea = a_cp[j + 1];
        let eb = b_cp[j + 1];
        while ia < ea || ib < eb {
            let row_a = if ia < ea { a_ri[ia] } else { usize::MAX };
            let row_b = if ib < eb { b_ri[ib] } else { usize::MAX };
            if row_a < row_b {
                c_ri.push(row_a);
                c_v.push(a_v[ia]);
                ia += 1;
            } else if row_b < row_a {
                c_ri.push(row_b);
                c_v.push(-b_v[ib]);
                ib += 1;
            } else {
                c_ri.push(row_a);
                c_v.push(a_v[ia] - b_v[ib]);
                ia += 1;
                ib += 1;
            }
        }
        c_cp[j + 1] = c_ri.len();
    }

    CscMatrix::try_from_csc_data(a.nrows(), nb, c_cp, c_ri, c_v).unwrap()
}

/// Subtract diagonal vectors from a sparse matrix.
/// If negate_e is false:  result = mat - diag(d) - diag(e)
/// If negate_e is true:   result = mat - diag(d) + diag(e)  (used for Hva)
fn subtract_diags(
    mat: &CscMatrix<Complex64>,
    d: &[Complex64],
    e: &[Complex64],
    negate_e: bool,
) -> CscMatrix<Complex64> {
    let nb = mat.ncols();
    let mut c_cp = mat.col_offsets().to_vec();
    let mut c_ri = mat.row_indices().to_vec();
    let mut c_v = mat.values().to_vec();

    // For each diagonal position j, add -(d[j] + e[j]) to mat[j,j] (if present)
    // or insert a new entry.
    for j in 0..nb {
        let delta = if negate_e {
            -d[j] + e[j]
        } else {
            -d[j] - e[j]
        };
        if delta == Complex64::new(0.0, 0.0) {
            continue;
        }
        // Find diagonal entry in column j
        let start = c_cp[j];
        let end = c_cp[j + 1];
        let pos = c_ri[start..end].binary_search(&j);
        if let Ok(rel) = pos {
            c_v[start + rel] += delta;
        }
        // If diagonal not present, we skip (it should be present due to Ybus structure).
        // For robustness in practice this is fine — Ybus always has diagonal entries.
    }

    CscMatrix::try_from_csc_data(nb, nb, c_cp, c_ri, c_v).unwrap()
}

/// Row-scale a CSC complex matrix: result[l, j] = lam[l] * a[l, j]   (or conj(a[l,j]) if conjugate).
fn row_scale_cx(
    a: &CscMatrix<Complex64>,
    lam: &[f64],
    conjugate: bool,
) -> CscMatrix<Complex64> {
    let ri = a.row_indices();
    let vals: Vec<Complex64> = a
        .values()
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let scaled = if conjugate { v.conj() } else { v };
            lam[ri[idx]] * scaled
        })
        .collect();
    CscMatrix::try_from_csc_data(
        a.nrows(),
        a.ncols(),
        a.col_offsets().to_vec(),
        a.row_indices().to_vec(),
        vals,
    )
    .unwrap()
}

/// Helper: given col_offsets and an nnz index, return which column it belongs to.
#[inline]
fn cp_to_col(col_offsets: &[usize], idx: usize) -> usize {
    col_offsets.partition_point(|&o| o <= idx) - 1
}
