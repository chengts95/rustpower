use super::new_dsdvbus2::JacobianPattern2;

#[test]
fn v4_cache_matches_v3_structure_and_cuts() {
    for n in 1..=6 {
        for npq in 0..=n {
            for npv in 0..=n - npq {
                for mask in 0..1usize << n {
                    let mut rows = Vec::new();
                    let mut offsets = vec![0];
                    for k in 0..n {
                        rows.extend((0..n).filter(|&i| i == k || mask & (1 << i) != 0));
                        offsets.push(rows.len());
                    }
                    let old = JacobianPattern2::build_full_range(&offsets, &rows, npv, npq);
                    let new = super::jacobian_cache::JacobianCache::build_from_permuted(
                        &offsets, &rows, npv, npq,
                    );
                    assert_eq!(old.j_col_ptrs, new.j_col_ptrs);
                    assert_eq!(old.j_row_indices, new.j_row_indices);
                    assert_eq!(old.pq_ends, new.pq_ends);
                    assert_eq!(old.active_ends, new.active_ends);
                    assert_eq!(old.diag_ptrs, new.diag_ptrs);
                }
            }
        }
    }
}

pub(crate) fn assert_patterns_equal(a: &JacobianPattern2, b: &JacobianPattern2) {
    assert_eq!(a.nnz_j, b.nnz_j);
    assert_eq!(a.j_col_ptrs, b.j_col_ptrs);
    assert_eq!(a.j_row_indices, b.j_row_indices);
    assert_eq!(a.pq_ends, b.pq_ends);
    assert_eq!(a.active_ends, b.active_ends);
    assert_eq!(a.j11_starts, b.j11_starts);
    assert_eq!(a.j21_starts, b.j21_starts);
    assert_eq!(a.j12_starts, b.j12_starts);
    assert_eq!(a.j22_starts, b.j22_starts);
    assert_eq!(a.diag_ptrs, b.diag_ptrs);
}

#[test]
fn symbolic_ranges_match_full_search_for_all_small_row_patterns() {
    // Enumerate every sorted row subset with and without the diagonal, at
    // every PQ/PV/slack split. This covers empty row blocks, boundary
    // diagonals, no PQ/PV buses and multiple slack buses.
    for n in 1..=7 {
        for npq in 0..=n {
            for npv in 0..=n - npq {
                for mask in 0..1usize << n {
                    let rows: Vec<_> = (0..n).filter(|r| mask & (1 << r) != 0).collect();
                    let mut indices = Vec::new();
                    let mut offsets = vec![0];
                    for _ in 0..n {
                        indices.extend_from_slice(&rows);
                        offsets.push(indices.len());
                    }
                    let old = JacobianPattern2::build_full_range(&offsets, &indices, npv, npq);
                    let new = JacobianPattern2::build_from_permuted(&offsets, &indices, npv, npq);
                    assert_patterns_equal(&old, &new);
                }
            }
        }
    }
}

#[test]
fn symbolic_ranges_match_full_search_for_long_mixed_columns() {
    let n = 513;
    let npq = 211;
    let npv = 199;
    let mut rows = Vec::new();
    let mut offsets = vec![0];
    for k in 0..n {
        rows.extend((0..n).filter(|&r| r == k || (r + 3 * k) % 7 == 0));
        offsets.push(rows.len());
    }
    let old = JacobianPattern2::build_full_range(&offsets, &rows, npv, npq);
    let new = JacobianPattern2::build_from_permuted(&offsets, &rows, npv, npq);
    assert_patterns_equal(&old, &new);
}
