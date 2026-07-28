//! Phase 0 gate (architecture doc §6):
//!
//! * starts tables match hand-computed values on a 3-bus fixture;
//! * the shared graph pattern is **identical** to `JacobianPattern2`
//!   (col starts, row indices, nnz) — the proven Jacobian layout;
//! * `base + col_starts[k] + diag_off[k]` matches a direct search of the
//!   diagonal quadrant slots in every block;
//! * 14-bus fixture: the pattern matches an independent naive reference,
//!   column rows are sorted and unique (write-once coverage).

use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

use super::pattern::KktPattern;
use super::YbusAnalysisCache;
use crate::basic::new_dsdvbus2::JacobianPattern2;

/// Symmetric test Ybus with explicit structural diagonal, values irrelevant.
fn ybus_from_edges(nb: usize, edges: &[(usize, usize)]) -> CscMatrix<Complex64> {
    let mut coo = CooMatrix::new(nb, nb);
    for k in 0..nb {
        coo.push(k, k, Complex64::new(2.0, -7.0));
    }
    for &(i, j) in edges {
        let y = Complex64::new(-1.0, 4.0);
        coo.push(i, j, y);
        coo.push(j, i, y);
    }
    CscMatrix::from(&coo)
}

/// 4 buses: 0, 1 PQ; 2 PV; 3 slack. Columns:
///   col0 [0,1,2]  col1 [0,1,2]  col2 [0,1,2,3]  col3 [2,3]
fn fixture_3bus() -> (CscMatrix<Complex64>, usize, usize) {
    (ybus_from_edges(4, &[(0, 1), (0, 2), (1, 2), (2, 3)]), 1, 2)
}

/// 14 buses: 0..9 PQ, 9..13 PV, 13 slack. Ring + chords (ext_ref style).
fn fixture_14bus() -> (CscMatrix<Complex64>, usize, usize) {
    let nb = 14;
    let mut edges: Vec<(usize, usize)> = (0..nb).map(|i| (i, (i + 1) % nb)).collect();
    for i in (0..nb).step_by(2) {
        edges.push((i, (i + 3) % nb));
    }
    (ybus_from_edges(nb, &edges), 4, 9)
}

#[test]
fn cache_hand_computed_3bus() {
    let (ybus, n_pv, n_pq) = fixture_3bus();
    let cache = YbusAnalysisCache::build(&ybus, n_pv, n_pq);

    assert_eq!(cache.n_active(), 3);
    assert_eq!(cache.pq_ends(), &[2, 2, 2]);
    assert_eq!(cache.active_ends(), &[3, 3, 3]);
    assert_eq!(cache.diag_ptrs(), &[0, 4, 8]);
    assert_eq!(cache.diag_off(), &[0, 1, 2]);
    // Mirror edges: (1,0)→3, (2,0)→6, diagonal (2,2)→8, (3,2)→10.
    assert_eq!(cache.y_trans()[1], 3);
    assert_eq!(cache.y_trans()[2], 6);
    assert_eq!(cache.y_trans()[8], 8);
    assert_eq!(cache.y_trans()[9], 10);
}

#[test]
fn graph_hand_computed_3bus() {
    let (ybus, n_pv, n_pq) = fixture_3bus();
    let pat = KktPattern::build(&ybus, n_pv, n_pq);

    // 5 state/eq columns [θ0,θ1,θ2,|V|0,|V|1]; each is [active | pq + n_act]
    // = [0,1,2,3,4] since every active column here has rows [0,1,2].
    assert_eq!(pat.graph.n_cols, 5);
    assert_eq!(pat.graph.col_starts, [0, 5, 10, 15, 20]);
    assert_eq!(pat.graph.nnz, 25);
    assert!(pat.graph.row_indices.chunks(5).all(|c| c == [0, 1, 2, 3, 4]));

    // Bases: [J | H | Jᵀ | −I]; 5 equations for the −I block.
    assert_eq!(pat.j_base, 0);
    assert_eq!(pat.h_base, 25);
    assert_eq!(pat.jt_base, 50);
    assert_eq!(pat.d_base, 75);
    assert_eq!(pat.nnz_total, 80);
}

#[test]
fn graph_equals_jacobian_pattern2() {
    for fixture in [fixture_3bus, fixture_14bus] {
        let (ybus, n_pv, n_pq) = fixture();
        let pat = KktPattern::build(&ybus, n_pv, n_pq);
        let jp2 =
            JacobianPattern2::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), n_pv, n_pq);

        assert_eq!(pat.graph.nnz, jp2.nnz_j);
        assert_eq!(pat.graph.n_cols + 1, jp2.j_col_ptrs.len());
        assert_eq!(pat.graph.col_starts, &jp2.j_col_ptrs[..pat.graph.n_cols]);
        assert_eq!(pat.graph.row_indices, jp2.j_row_indices);
    }
}

/// Global position of `row` inside the graph's matrix column `col`, placed
/// at `block_base`, by linear search.
fn find_in_block(pat: &KktPattern, block_base: usize, col: usize, row: usize) -> usize {
    let pos = pat
        .graph
        .col_rows(col)
        .iter()
        .position(|&r| r == row)
        .unwrap_or_else(|| panic!("row {row} missing in graph column {col}"));
    block_base + pat.graph.col_range(col).start + pos
}

fn check_diag_addressing(pat: &KktPattern) {
    let cache = &pat.cache;
    let n_act = cache.n_active();

    for k in 0..n_act {
        // aa diagonal (θ_k, θ_k): leading segment of the θ column.
        assert_eq!(
            pat.col_diag(pat.h_base, k, k),
            find_in_block(pat, pat.h_base, k, k)
        );
        if k < cache.n_pq() {
            // va diagonal (|V|_k, θ_k): leading offset + active segment length.
            let va = pat.col_diag(pat.h_base, k, k) + cache.active_ends()[k];
            assert_eq!(va, find_in_block(pat, pat.h_base, k, n_act + k));
            // av diagonal (θ_k, |V|_k): leading segment of the |V| column.
            assert_eq!(
                pat.col_diag(pat.h_base, n_act + k, k),
                find_in_block(pat, pat.h_base, n_act + k, k)
            );
            // vv diagonal (|V|_k, |V|_k): the μ-shift slot for PQ buses.
            let vv = pat.col_diag(pat.h_base, n_act + k, k) + cache.active_ends()[k];
            assert_eq!(vv, find_in_block(pat, pat.h_base, n_act + k, n_act + k));
        }
    }
}

#[test]
fn diag_addressing_matches_direct_search_3bus() {
    let (ybus, n_pv, n_pq) = fixture_3bus();
    let pat = KktPattern::build(&ybus, n_pv, n_pq);
    check_diag_addressing(&pat);
}

/// Independent naive reference: rebuild the graph rows directly from the
/// Ybus CSC with plain scans, and compare entry by entry.
#[test]
fn graph_matches_naive_reference_14bus() {
    let (ybus, n_pv, n_pq) = fixture_14bus();
    let pat = KktPattern::build(&ybus, n_pv, n_pq);
    let n_act = pat.cache.n_active();
    let (y_cp, y_ri) = (ybus.col_offsets(), ybus.row_indices());

    let mut expected_rows: Vec<usize> = Vec::new();
    let mut expected_starts: Vec<usize> = Vec::new();
    for c in 0..n_act + n_pq {
        expected_starts.push(expected_rows.len());
        let k = c % n_act;
        let col = &y_ri[y_cp[k]..y_cp[k + 1]];
        expected_rows.extend(col.iter().copied().filter(|&r| r < n_act));
        expected_rows.extend(col.iter().copied().filter(|&r| r < n_pq).map(|r| n_act + r));
    }

    assert_eq!(pat.graph.col_starts, expected_starts);
    assert_eq!(pat.graph.row_indices, expected_rows);
    check_diag_addressing(&pat);
}

#[test]
fn rows_sorted_unique_and_reduced_14bus() {
    let (ybus, n_pv, n_pq) = fixture_14bus();
    let pat = KktPattern::build(&ybus, n_pv, n_pq);
    let n_state = pat.cache.n_active() + n_pq;

    for c in 0..pat.graph.n_cols {
        let rows = pat.graph.col_rows(c);
        assert!(rows.windows(2).all(|w| w[0] < w[1]), "column {c} not sorted/unique");
        assert!(rows.iter().all(|&r| r < n_state), "slack leaked into column {c}");
    }

    // Partition check: blocks tile [0, nnz_total) with no gaps or overlaps.
    assert_eq!(pat.j_base, 0);
    assert_eq!(pat.h_base, pat.graph.nnz);
    assert_eq!(pat.jt_base, 2 * pat.graph.nnz);
    assert_eq!(pat.d_base, 3 * pat.graph.nnz);
    assert_eq!(pat.nnz_total, pat.d_base + n_state);
}

// ─── Phase 1 gate (doc §6): fill_h + fill_jt ────────────────────────────────
//
// * J vs finite differences ≤ 1e-8 (the existing v3 kernel on the graph layout);
// * H(r) vs residual-weighted J-differences ≤ 1e-8;
// * H exactly symmetric;
// * Jᵀ reconstruction == Jᵀ exactly (bitwise transpose);
// * apply_mu_delta touches only the aa/vv diagonal slots.

use nalgebra::DVector;

use super::kernels::{apply_mu_delta, fill_h, fill_jt};
use super::BlockDesc;
use crate::basic::new_dsdvbus3::fill_jacobian_v3;

struct LmCase {
    ybus: CscMatrix<Complex64>,
    pat: KktPattern,
    v: DVector<Complex64>,
    sbus: Vec<Complex64>,
    n_pv: usize,
    n_pq: usize,
    n_act: usize,
    n_state: usize,
}

fn lm_case(fixture: fn() -> (CscMatrix<Complex64>, usize, usize)) -> LmCase {
    let (ybus, n_pv, n_pq) = fixture();
    let n_act = n_pv + n_pq;
    let n_state = n_act + n_pq;
    let pat = KktPattern::build(&ybus, n_pv, n_pq);

    let nb = ybus.ncols();
    // Reference point (defines the injections) and evaluation point.
    let v_ref = DVector::from_vec(vec![Complex64::new(1.0, 0.0); nb]);
    let ibus_ref = &ybus * &v_ref;
    let sbus: Vec<Complex64> = (0..nb).map(|i| v_ref[i] * ibus_ref[i].conj()).collect();

    let v = DVector::from_vec(
        (0..nb)
            .map(|k| {
                let ang = 0.03 * (1.3 * k as f64).sin() - 0.01 * k as f64;
                let mag = 1.0 + 0.004 * (2.1 * k as f64).cos();
                Complex64::from_polar(mag, ang)
            })
            .collect(),
    );

    LmCase { ybus, pat, v, sbus, n_pv, n_pq, n_act, n_state }
}

/// Reduced residual r = [P mis (n_act); Q mis (n_pq)].
fn mismatch(case: &LmCase, v: &DVector<Complex64>) -> Vec<f64> {
    let ibus = &case.ybus * v;
    let mut r = vec![0.0; case.n_state];
    for i in 0..case.n_act {
        let s = v[i] * ibus[i].conj() - case.sbus[i];
        r[i] = s.re;
        if i < case.n_pq {
            r[case.n_act + i] = s.im;
        }
    }
    r
}

/// `Vnorm` and `scalc = V·conj(I)` at `v` (the v3 kernel's per-bus inputs).
fn vnorm_scalc(case: &LmCase, v: &DVector<Complex64>) -> (Vec<Complex64>, Vec<Complex64>) {
    let ibus = &case.ybus * v;
    let scalc: Vec<Complex64> = (0..v.len()).map(|i| v[i] * ibus[i].conj()).collect();
    let vnorm: Vec<Complex64> = (0..v.len())
        .map(|i| {
            let m = v[i].norm();
            if m > 1e-12 { v[i] / m } else { Complex64::new(1.0, 0.0) }
        })
        .collect();
    (vnorm, scalc)
}

/// J block values via the production v3 kernel (its layout is the graph pattern).
fn jacobian_at(case: &LmCase, v: &DVector<Complex64>) -> Vec<f64> {
    let (vnorm, scalc) = vnorm_scalc(case, v);
    let mut j_vals = vec![0.0; case.pat.graph.nnz];
    fill_jacobian_v3::<false>(
        &case.ybus,
        v.as_slice(),
        &vnorm,
        &scalc,
        &case.pat.graph.col_starts,
        case.pat.cache.pq_ends(),
        case.pat.cache.active_ends(),
        case.pat.cache.diag_ptrs(),
        case.n_pv,
        case.n_pq,
        &mut j_vals,
    );
    j_vals
}

/// Dense `n × n` expansion of a graph-layout block.
fn dense_block(graph: &BlockDesc, vals: &[f64], n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n * n];
    for c in 0..graph.n_cols {
        for (pos, &row) in graph.col_rows(c).iter().enumerate() {
            d[row * n + c] = vals[graph.col_range(c).start + pos];
        }
    }
    d
}

/// Perturb state `c` by exactly `d`: θ states rotate the bus voltage by `d`
/// radians, |V| states add `d` to the magnitude. Slack is never a state.
fn perturbed(case: &LmCase, c: usize, d: f64) -> DVector<Complex64> {
    let mut vp = case.v.clone();
    if c < case.n_act {
        vp[c] *= Complex64::from_polar(1.0, d);
    } else {
        let k = c - case.n_act;
        let m = vp[k].norm();
        vp[k] *= (m + d) / m;
    }
    vp
}

fn phase1_fd_gate(fixture: fn() -> (CscMatrix<Complex64>, usize, usize)) {
    let case = lm_case(fixture);
    let n = case.n_state;
    let r = mismatch(&case, &case.v);

    let j_vals = jacobian_at(&case, &case.v);
    let mut jt_vals = vec![0.0; case.pat.graph.nnz];
    fill_jt::<false>(&case.ybus, &case.pat, j_vals.as_ptr(), jt_vals.as_mut_ptr());
    let mut h_vals = vec![0.0; case.pat.graph.nnz];
    fill_h::<false>(&case.ybus, &case.pat, case.v.as_slice(), &r, &mut h_vals);

    let dj = dense_block(&case.pat.graph, &j_vals, n);
    let djt = dense_block(&case.pat.graph, &jt_vals, n);
    let dh = dense_block(&case.pat.graph, &h_vals, n);

    let eps = 1e-6;
    let mut max_j = 0.0f64;
    let mut max_h = 0.0f64;
    for c in 0..n {
        let vp = perturbed(&case, c, eps);
        let vm = perturbed(&case, c, -eps);
        let fp = mismatch(&case, &vp);
        let fm = mismatch(&case, &vm);
        let jp = dense_block(&case.pat.graph, &jacobian_at(&case, &vp), n);
        let jm = dense_block(&case.pat.graph, &jacobian_at(&case, &vm), n);
        for row in 0..n {
            max_j = max_j.max(((fp[row] - fm[row]) / (2.0 * eps) - dj[row * n + c]).abs());
        }
        // H(m,c) = Σ_row r_row · ∂J[row,m]/∂x_c
        for m in 0..n {
            let mut hn = 0.0;
            for row in 0..n {
                hn += r[row] * (jp[row * n + m] - jm[row * n + m]);
            }
            max_h = max_h.max((hn / (2.0 * eps) - dh[m * n + c]).abs());
        }
    }
    assert!(max_j < 1e-8, "J vs FD: {max_j:e}");
    assert!(max_h < 1e-8, "H vs r-weighted J-difference: {max_h:e}");

    // H exactly symmetric.
    let mut max_sym = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            max_sym = max_sym.max((dh[a * n + b] - dh[b * n + a]).abs());
        }
    }
    assert!(max_sym < 1e-12, "H not symmetric: {max_sym:e}");

    // Jᵀ is a bitwise transpose of J (values are copied, not recomputed).
    let mut n_bad = 0;
    for a in 0..n {
        for b in 0..n {
            if djt[a * n + b] != dj[b * n + a] {
                if n_bad < 10 {
                    println!("Jᵀ mismatch at ({a},{b}): jt={:e} j={:e}", djt[a * n + b], dj[b * n + a]);
                }
                n_bad += 1;
            }
        }
    }
    assert_eq!(n_bad, 0, "{n_bad} Jᵀ mismatches");
}

#[test]
fn phase1_fd_gate_3bus() {
    phase1_fd_gate(fixture_3bus);
}

#[test]
fn phase1_fd_gate_14bus() {
    phase1_fd_gate(fixture_14bus);
}

#[test]
fn apply_mu_delta_touches_only_diag_slots() {
    let case = lm_case(fixture_3bus);
    let r = mismatch(&case, &case.v);
    let mut h_vals = vec![0.0; case.pat.graph.nnz];
    fill_h::<false>(&case.ybus, &case.pat, case.v.as_slice(), &r, &mut h_vals);

    let before = h_vals.clone();
    let dmu = 0.125;
    apply_mu_delta::<false>(&case.pat, &mut h_vals, dmu);

    let cache = &case.pat.cache;
    let cs = &case.pat.graph.col_starts;
    let mut mu_slots = std::collections::HashSet::new();
    for k in 0..case.n_act {
        mu_slots.insert(cs[k] + cache.diag_off()[k]);
        if k < case.n_pq {
            mu_slots.insert(cs[case.n_act + k] + cache.active_ends()[k] + cache.diag_off()[k]);
        }
    }
    for idx in 0..h_vals.len() {
        if mu_slots.contains(&idx) {
            assert_eq!(h_vals[idx], before[idx] + dmu, "slot {idx}");
        } else {
            assert_eq!(h_vals[idx], before[idx], "slot {idx} must be untouched");
        }
    }
}

// ─── Phase 2 gate (doc §6): flat view ───────────────────────────────────────
//
// * the global CSC passes solver-format validation (nalgebra constructor);
// * block view and flat view agree entry-by-entry on the assembled global
//   matrix, before and after an apply_mu_delta step;
// * the constant −I block, stamped once, is never rewritten by any fill;
// * a μ update moves exactly the n_state diagonal slots, nothing else.

use super::flat::{fill_kkt, fill_kkt_flat, FlatLayout};

/// Dense `2n × 2n` expansion of the flat global CSC.
fn dense_flat(flat: &FlatLayout, vals: &[f64]) -> Vec<f64> {
    let n2 = 2 * flat.n_state;
    let mut d = vec![0.0; n2 * n2];
    for c in 0..n2 {
        for p in flat.col_offsets[c]..flat.col_offsets[c + 1] {
            d[flat.row_indices[p] * n2 + c] = vals[p];
        }
    }
    d
}

/// Dense `2n × 2n` assembly of the block view `[J | H | Jᵀ | −I]`.
fn dense_block_view(case: &LmCase, vals: &[f64]) -> Vec<f64> {
    let n = case.n_state;
    let n2 = 2 * n;
    let nnz = case.pat.graph.nnz;
    let mut d = vec![0.0; n2 * n2];
    let dj = dense_block(&case.pat.graph, &vals[0..nnz], n);
    let dh = dense_block(&case.pat.graph, &vals[nnz..2 * nnz], n);
    let djt = dense_block(&case.pat.graph, &vals[2 * nnz..3 * nnz], n);
    for a in 0..n {
        for b in 0..n {
            d[a * n2 + b] = dh[a * n + b]; // top-left: H
            d[a * n2 + (n + b)] = djt[a * n + b]; // top-right: Jᵀ
            d[(n + a) * n2 + b] = dj[a * n + b]; // bottom-left: J
        }
        d[(n + a) * n2 + (n + a)] = vals[3 * nnz + a]; // bottom-right: −I
    }
    d
}

fn phase2_gate(fixture: fn() -> (CscMatrix<Complex64>, usize, usize)) {
    let case = lm_case(fixture);
    let flat = FlatLayout::build(&case.pat);
    let n2 = 2 * case.n_state;
    let r = mismatch(&case, &case.v);
    let (vnorm, scalc) = vnorm_scalc(&case, &case.v);

    // Solver-format validation by the nalgebra constructor (offsets monotone,
    // row indices sorted per column and in range).
    let csc = CscMatrix::try_from_csc_data(
        n2,
        n2,
        flat.col_offsets.clone(),
        flat.row_indices.clone(),
        vec![0.0; flat.nnz_flat],
    )
    .expect("flat CSC fails nalgebra format validation");
    assert_eq!(csc.nnz(), flat.nnz_flat);

    // Fill both views at the same point.
    let mut block_vals = vec![0.0; case.pat.nnz_total];
    fill_kkt(&case.ybus, &case.pat, case.v.as_slice(), &vnorm, &scalc, &r, &mut block_vals);

    let mut flat_vals = vec![0.0; flat.nnz_flat];
    flat.stamp_neg_i(&mut flat_vals);
    fill_kkt_flat(&case.ybus, &case.pat, &flat, case.v.as_slice(), &vnorm, &scalc, &r, &mut flat_vals);

    // The −I slots survive the three fills untouched.
    for c in 0..case.n_state {
        let p = flat.col_offsets[case.n_state + c + 1] - 1;
        assert_eq!(flat_vals[p], -1.0, "−I slot of s-column {c} rewritten by a fill");
    }

    // Entry-by-entry agreement.
    let db = dense_block_view(&case, &block_vals);
    let df = dense_flat(&flat, &flat_vals);
    assert_eq!(db, df, "block/flat views disagree");

    // A μ step in both views still agrees and moves only the diagonal.
    let dmu = 0.0625;
    apply_mu_delta::<false>(&case.pat, &mut block_vals[case.pat.h_base..case.pat.jt_base], dmu);
    apply_mu_delta::<true>(&case.pat, &mut flat_vals, dmu);
    let db2 = dense_block_view(&case, &block_vals);
    let df2 = dense_flat(&flat, &flat_vals);
    assert_eq!(db2, df2, "block/flat views disagree after μ update");

    let mut n_diff = 0;
    for a in 0..n2 {
        for b in 0..n2 {
            if db2[a * n2 + b] != db[a * n2 + b] {
                assert_eq!(a, b, "μ moved an off-diagonal entry ({a},{b})");
                n_diff += 1;
            }
        }
    }
    assert_eq!(n_diff, case.n_state, "μ should move exactly the n_state diagonal slots");
}

#[test]
fn phase2_flat_gate_3bus() {
    phase2_gate(fixture_3bus);
}

#[test]
fn phase2_flat_gate_14bus() {
    phase2_gate(fixture_14bus);
}
