use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

/// Second derivatives of complex bus power injections w.r.t. voltage.
///
/// Returns (Gaa, Gav, Gva, Gvv) — each nb×nb complex sparse matrix — such that
///   lam^T · d²S/dVx·dVy  =  Gxy   for x,y ∈ {a=angle, v=magnitude}
///
/// Reference: MATPOWER TN2 §5.
/// Python equivalent: pandapower.pypower.d2Sbus_dV2
#[allow(non_snake_case)]
pub fn d2Sbus_dV2(
    Ybus: &CscMatrix<Complex64>,
    v: &DVector<Complex64>,
    lam: &DVector<Complex64>,
) -> (
    CscMatrix<Complex64>, // Gaa
    CscMatrix<Complex64>, // Gav
    CscMatrix<Complex64>, // Gva
    CscMatrix<Complex64>, // Gvv
) {
    let nb = v.len();
    let j_unit = Complex64::i();

    let v_s = v.as_slice();
    let lam_s = lam.as_slice();

    // Ibus = Ybus * V
    let ibus: DVector<Complex64> = Ybus * v;
    let ibus_s = ibus.as_slice();

    // Precompute per-bus quantities
    // inv_abv[i] = 1 / |V[i]|
    let inv_abv: Vec<f64> = v_s.iter().map(|vi| 1.0 / vi.norm()).collect();

    // lam_v[i] = lam[i] * V[i]
    let lam_v: Vec<Complex64> = (0..nb).map(|i| lam_s[i] * v_s[i]).collect();

    // D·lam vector: (D·lam)[i] = sum_j conj(Ybus[j,i]) * V[j] * lam[j]
    //   = (Ybus^H · (lam .* V))[i]
    // Ybus^H = conj(Ybus^T); we compute it as conj(Ybus)^T · lam_v
    // Equivalent to: for each col i of Ybus, sum over row j: conj(Ybus[j,i]) * lam_v[j]
    let mut d_lam = vec![Complex64::new(0.0, 0.0); nb];
    {
        let col_offsets = Ybus.col_offsets();
        let row_indices = Ybus.row_indices();
        let ybus_vals = Ybus.values();
        for i in 0..nb {
            for idx in col_offsets[i]..col_offsets[i + 1] {
                let j = row_indices[idx];
                // Ybus[j, i] → Ybus^H [i, j]: conj(Ybus[j,i])
                // (Ybus^H · lam_v)[i] += conj(Ybus[j,i]) * lam_v[j]
                d_lam[i] += ybus_vals[idx].conj() * lam_v[j];
            }
        }
    }

    // Now build the four output matrices by a single pass over Ybus NNZ.
    // All four have the same sparsity as Ybus (possibly with extra diagonal terms).
    //
    // Using the element-wise derivation of MATPOWER TN2:
    //   C[i,j] = lam[i]*V[i] * conj(Ybus[i,j]*V[j])           (row-scale of conj(B), B=Ybus*diagV)
    //   E_off[i,j] = conj(V[i]) * conj(Ybus[j,i]) * V[j]*lam[j]  for i≠j
    //   E_diag[i]  = conj(V[i]) * (conj(Ybus[i,i])*V[i]*lam[i] - d_lam[i])
    //   F_off[i,j] = C[i,j]
    //   F_diag[i]  = C[i,i] - lam[i]*V[i]*conj(Ibus[i])
    //
    //   Gaa[i,j] = E[i,j] + F[i,j]
    //   Gva[i,j] = j * inv_abv[i] * (E[i,j] - F[i,j])
    //   Gav = Gva^T   (handled by swapping roles of row/col)
    //   Gvv[i,j] = inv_abv[i] * (C[i,j] + C[j,i]) * inv_abv[j]
    //
    // We store Gva and Gav with the Ybus sparsity.  Gav = Gva^T is built by
    // accumulating into the transpose positions.

    let nnz = Ybus.nnz();
    let col_offsets = Ybus.col_offsets();
    let row_indices = Ybus.row_indices();
    let ybus_vals = Ybus.values();

    let mut gaa_v = vec![Complex64::new(0.0, 0.0); nnz];
    let mut gva_v = vec![Complex64::new(0.0, 0.0); nnz];
    let mut gvv_v = vec![Complex64::new(0.0, 0.0); nnz];

    // We'll build Gvv using C + C^T, which needs access to the transpose element.
    // Build a column-index → NNZ-index map for transposed access.
    // For each (j,i) in Ybus, find the index of (i,j).
    // Since Ybus is symmetric for standard power networks, C^T[i,j] = C[j,i].
    // We build a helper: for non-zero at position (j, i), find its index.

    // transpose_idx[idx] = index in the CSC arrays corresponding to (col, row) = (row[idx], col_of[idx])
    // i.e., the index of the transpose entry.
    let mut col_of = vec![0usize; nnz]; // which column each nnz belongs to
    for j in 0..nb {
        for idx in col_offsets[j]..col_offsets[j + 1] {
            col_of[idx] = j;
        }
    }

    // For each nnz at (i=row, j=col), find the nnz at (i=j, j=i) = (row=j, col=i).
    // Build a hash map: (row, col) → nnz_index
    // For efficiency, use a sorted search since Ybus is sorted.
    let find_nnz = |row: usize, col: usize| -> Option<usize> {
        // Binary search in col `col`'s range for row `row`
        let range = col_offsets[col]..col_offsets[col + 1];
        let ri_slice = &row_indices[range.clone()];
        ri_slice
            .binary_search(&row)
            .ok()
            .map(|pos| range.start + pos)
    };

    for j in 0..nb {
        for idx in col_offsets[j]..col_offsets[j + 1] {
            let i = row_indices[idx];
            let y_ij = ybus_vals[idx];

            // C[i,j] = lam_v[i] * conj(y_ij * V[j])
            let c_ij = lam_v[i] * (y_ij * v_s[j]).conj();

            // E[i,j]:
            let e_ij = if i == j {
                // diagonal: conj(V[i]) * (conj(Ybus[i,i])*V[i]*lam[i] - d_lam[i])
                v_s[i].conj() * (y_ij.conj() * v_s[i] * lam_s[i] - d_lam[i])
            } else {
                // off-diagonal: conj(V[i]) * conj(Ybus[j,i]) * V[j] * lam[j]
                // conj(Ybus[j,i]): need element (j,i) of Ybus; if symmetric = conj(y_ij)
                // For the general case, search for (row=j, col=i):
                let y_ji_conj = if let Some(ji_idx) = find_nnz(j, i) {
                    ybus_vals[ji_idx].conj()
                } else {
                    Complex64::new(0.0, 0.0)
                };
                v_s[i].conj() * y_ji_conj * v_s[j] * lam_s[j]
            };

            // F[i,j]:
            let f_ij = if i == j {
                c_ij - lam_v[i] * ibus_s[i].conj()
            } else {
                c_ij
            };

            gaa_v[idx] = e_ij + f_ij;
            gva_v[idx] = j_unit * inv_abv[i] * (e_ij - f_ij);

            // Gvv[i,j] = inv_abv[i] * (C[i,j] + C[j,i]) * inv_abv[j]
            // C[j,i] is at the transpose position
            let c_ji = if let Some(ji_idx) = find_nnz(j, i) {
                lam_v[j] * (ybus_vals[ji_idx] * v_s[i]).conj()
            } else {
                Complex64::new(0.0, 0.0)
            };
            gvv_v[idx] = inv_abv[i] * (c_ij + c_ji) * inv_abv[j];
        }
    }

    // Gva^T = Gav: build by transposing Gva
    // Gva[i,j] is at nnz idx; Gav[j,i] should get the same value.
    // We build a Gav values array in the Ybus^T sparsity = Ybus sparsity (for symmetric),
    // but to be safe, build Gav as the transpose of Gva.
    let mut gav_v = vec![Complex64::new(0.0, 0.0); nnz];
    for j in 0..nb {
        for idx in col_offsets[j]..col_offsets[j + 1] {
            let i = row_indices[idx];
            // Gva[i,j] is gva_v[idx]; Gav = Gva^T means Gav[j,i] = Gva[i,j]
            // Find position (row=j, col=i) in the CSC arrays
            if let Some(ji_idx) = find_nnz(j, i) {
                gav_v[ji_idx] = gva_v[idx];
            }
        }
    }

    let mk = |vals: Vec<Complex64>| {
        CscMatrix::try_from_csc_data(nb, nb, col_offsets.to_vec(), row_indices.to_vec(), vals)
            .unwrap()
    };

    (mk(gaa_v), mk(gav_v), mk(gva_v), mk(gvv_v))
}
