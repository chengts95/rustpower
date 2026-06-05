use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use crate::basic::dsbr_dv::{dAbr_dV, dSbr_dV};
use crate::basic::dsbus_dv::dSbus_dV;

use super::problem::OPFData;

/// Evaluate OPF constraints and their Jacobians.
///
/// Returns:
///   g  : 2·nb equality constraint values (P and Q mismatch)
///   h  : 2·nl2 inequality constraint values (|Sf|²-Smax², |St|²-Smax²)
///   dg : (nx × 2·nb) equality constraint Jacobian (transposed for PIPS)
///   dh : (nx × 2·nl2) inequality Jacobian (transposed)
///
/// g[i]     = Re(mis[i])  for i in 0..nb   (P balance)
/// g[nb+i]  = Im(mis[i])  for i in 0..nb   (Q balance)
/// h[l]     = |Sf[l]|² - Smax[l]²
/// h[nl2+l] = |St[l]|² - Smax[l]²
#[allow(non_snake_case)]
pub fn opf_consfcn(
    data: &OPFData,
    x: &[f64],
) -> (
    Vec<f64>,          // g
    Vec<f64>,          // h
    CscMatrix<f64>,    // dg (nx × 2nb, transposed Jacobian)
    CscMatrix<f64>,    // dh (nx × 2nl2, transposed Jacobian)
) {
    let nb = data.nb;
    let nl = data.nl;
    let ng = data.ng;
    let nx = data.nx();

    let v = data.v_from_x(x);
    let sbus = data.sbus_from_x(x);
    let v_norm: DVector<Complex64> = v.map(|vi| vi / vi.norm());

    // Power mismatch: mis = V·conj(Ybus·V) - Sbus
    let mis: DVector<Complex64> = {
        let ibus: DVector<Complex64> = &data.ybus * &v;
        v.component_mul(&ibus.map(|c| c.conj())) - &sbus
    };

    // ── equality constraint values ──────────────────────────────────────────
    let mut g = vec![0.0f64; 2 * nb];
    for i in 0..nb {
        g[i] = mis[i].re;
        g[nb + i] = mis[i].im;
    }

    // ── inequality constraint values + branch jacobian (single dSbr_dV pass) ──
    let flow_max = data.flow_max_sq();
    let nl2 = nl;

    let (d_sf_d_va, d_sf_d_vm, d_st_d_va, d_st_d_vm, sf, st) = dSbr_dV(
        &data.yf, &data.yt, &data.f_buses, &data.t_buses, &v, &v_norm,
    );

    let mut h = vec![0.0f64; 2 * nl2];
    for l in 0..nl2 {
        h[l] = sf[l].norm_sqr() - flow_max[l];
        h[nl2 + l] = st[l].norm_sqr() - flow_max[l];
    }

    // ── Jacobians ────────────────────────────────────────────────────────────
    let (d_sbus_d_vm, d_sbus_d_va) = dSbus_dV(&data.ybus, &v, &v_norm);

    let dg_t = build_dg_transposed(
        nb, ng, nx,
        &d_sbus_d_va, &d_sbus_d_vm,
        &data.cg,
    );

    let dh_t = {
        let (d_af_d_va, d_af_d_vm, d_at_d_va, d_at_d_vm) =
            dAbr_dV(&d_sf_d_va, &d_sf_d_vm, &d_st_d_va, &d_st_d_vm, &sf, &st);
        build_dh_transposed(nb, nx, nl2, &d_af_d_va, &d_af_d_vm, &d_at_d_va, &d_at_d_vm)
    };

    (g, h, dg_t, dh_t)
}

// ── Jacobian builders ─────────────────────────────────────────────────────────

/// Build dg^T (nx × 2nb) from dSbus/dV blocks and Cg.
/// Layout: rows = [iVa | iVm | iPg | iQg], cols = [P_eq | Q_eq]
fn build_dg_transposed(
    nb: usize,
    ng: usize,
    nx: usize,
    dSbus_dVa: &CscMatrix<Complex64>,
    dSbus_dVm: &CscMatrix<Complex64>,
    cg: &CscMatrix<f64>,
) -> CscMatrix<f64> {
    // dg^T is (nx × 2nb).
    // Column j of dg^T corresponds to constraint j:
    //   j < nb  → P equation at bus j
    //   j >= nb → Q equation at bus (j-nb)
    //
    // For equation j, the row entries in dg^T (variables) are:
    //   Va_k (row k):      Re(dSbus_dVa[j, k])
    //   Vm_k (row nb+k):   Re(dSbus_dVm[j, k])
    //   Pg_g (row 2nb+g):  -Cg[j, g]
    //
    // dSbus_dV matrices are (nb × nb).  We need row j of these matrices.
    let va_rows = csc_to_row_lists_complex(dSbus_dVa);
    let vm_rows = csc_to_row_lists_complex(dSbus_dVm);
    let cg_rows = csc_to_row_lists(cg);

    let mut col_offsets = vec![0usize; 2 * nb + 1];
    let mut row_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    // P equations (j = 0..nb)
    for j in 0..nb {
        let start = row_indices.len();
        // d(P_j)/d(Va_k) = Re(dSbus_dVa[j, k])
        for &(k, val) in &va_rows[j] {
            row_indices.push(k);
            values.push(val.re);
        }
        // d(P_j)/d(Vm_k) = Re(dSbus_dVm[j, k])
        for &(k, val) in &vm_rows[j] {
            row_indices.push(nb + k);
            values.push(val.re);
        }
        // d(P_j)/d(Pg_g) = -Cg[j, g]
        for &(g, val) in &cg_rows[j] {
            row_indices.push(2 * nb + g);
            values.push(-val);
        }
        let end = row_indices.len();
        sort_col(&mut row_indices[start..end], &mut values[start..end]);
        col_offsets[j + 1] = end;
    }

    // Q equations (j = 0..nb -> eq nb..2nb)
    for j in 0..nb {
        let start = row_indices.len();
        // d(Q_j)/d(Va_k) = Im(dSbus_dVa[j, k])
        for &(k, val) in &va_rows[j] {
            row_indices.push(k);
            values.push(val.im);
        }
        // d(Q_j)/d(Vm_k) = Im(dSbus_dVm[j, k])
        for &(k, val) in &vm_rows[j] {
            row_indices.push(nb + k);
            values.push(val.im);
        }
        // d(Q_j)/d(Qg_g) = -Cg[j, g]
        for &(g, val) in &cg_rows[j] {
            row_indices.push(2 * nb + ng + g);
            values.push(-val);
        }
        let end = row_indices.len();
        sort_col(&mut row_indices[start..end], &mut values[start..end]);
        col_offsets[nb + j + 1] = end;
    }

    CscMatrix::try_from_csc_data(nx, 2 * nb, col_offsets, row_indices, values).unwrap()
}

/// Build dh^T (nx × 2nl2) from dAbr matrices.
/// Layout: rows = [iVa(0..nb) | iVm(nb..2nb) | iPg | iQg],
///         cols = [from-flow (0..nl2) | to-flow (nl2..2nl2)]
fn build_dh_transposed(
    nb: usize,
    nx: usize,
    nl2: usize,
    dAf_dVa: &CscMatrix<f64>,
    dAf_dVm: &CscMatrix<f64>,
    dAt_dVa: &CscMatrix<f64>,
    dAt_dVm: &CscMatrix<f64>,
) -> CscMatrix<f64> {
    // dh^T is (nx × 2nl2).
    // Cols 0..nl2: from-flow limits, rows in [Va, Vm] only (no Pg/Qg coupling)
    // Cols nl2..2nl2: to-flow limits
    //
    // For col l (from-flow eq l):
    //   Va rows: dAf_dVa[l, :]  (row l of dAf_dVa)
    //   Vm rows: dAf_dVm[l, :]
    //
    // dAf_dVa is (nl × nb).  Row l gives derivatives of flow l w.r.t. all buses.
    // Row l entries are (bus, val).

    let mut col_offsets = vec![0usize; 2 * nl2 + 1];
    let mut row_indices: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    // From-flow constraints (cols 0..nl2)
    append_flow_cols(
        &mut col_offsets, &mut row_indices, &mut values,
        dAf_dVa, dAf_dVm, nb, nl2, 0,
    );

    // To-flow constraints (cols nl2..2nl2)
    append_flow_cols(
        &mut col_offsets, &mut row_indices, &mut values,
        dAt_dVa, dAt_dVm, nb, nl2, nl2,
    );

    CscMatrix::try_from_csc_data(nx, 2 * nl2, col_offsets, row_indices, values).unwrap()
}

/// For each branch l, collect dA_dVa[l,:] and dA_dVm[l,:] and insert them
/// as column (col_offset + l) in the output dh^T.
fn append_flow_cols(
    col_offsets: &mut Vec<usize>,
    row_indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    dA_dVa: &CscMatrix<f64>, // nl × nb
    dA_dVm: &CscMatrix<f64>, // nl × nb
    nb: usize,
    nl: usize,
    col_offset: usize,
) {
    let va_rows = csc_to_row_lists(dA_dVa);
    let vm_rows = csc_to_row_lists(dA_dVm);

    for l in 0..nl {
        let start = row_indices.len();
        // Va contributions: row index = bus (0..nb)
        for &(bus, val) in &va_rows[l] {
            row_indices.push(bus);
            values.push(val);
        }
        // Vm contributions: row index = nb + bus
        for &(bus, val) in &vm_rows[l] {
            row_indices.push(nb + bus);
            values.push(val);
        }
        let end = row_indices.len();
        sort_col(&mut row_indices[start..end], &mut values[start..end]);
        col_offsets[col_offset + l + 1] = end;
    }
}

/// Convert a CSC matrix into a vector of per-row entry lists: `[(col, val)]`.
fn csc_to_row_lists(m: &CscMatrix<f64>) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); m.nrows()];
    let cp = m.col_offsets();
    let ri = m.row_indices();
    let v = m.values();
    for j in 0..m.ncols() {
        for idx in cp[j]..cp[j + 1] {
            rows[ri[idx]].push((j, v[idx]));
        }
    }
    rows
}

fn csc_to_row_lists_complex(m: &CscMatrix<Complex64>) -> Vec<Vec<(usize, Complex64)>> {
    let mut rows = vec![Vec::new(); m.nrows()];
    let cp = m.col_offsets();
    let ri = m.row_indices();
    let v = m.values();
    for j in 0..m.ncols() {
        for idx in cp[j]..cp[j + 1] {
            rows[ri[idx]].push((j, v[idx]));
        }
    }
    rows
}


#[inline]
fn sort_col(ri: &mut [usize], v: &mut [f64]) {
    // Insertion sort — columns are short (≤ degree of node)
    for i in 1..ri.len() {
        let mut j = i;
        while j > 0 && ri[j - 1] > ri[j] {
            ri.swap(j - 1, j);
            v.swap(j - 1, j);
            j -= 1;
        }
    }
}
