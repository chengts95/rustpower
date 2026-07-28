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
