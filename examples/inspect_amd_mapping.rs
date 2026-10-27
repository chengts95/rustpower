//! Structural experiment only: no production solver or assembly changes.
use nalgebra_sparse::{CooMatrix, CscMatrix};
use rustpower::{
    io::pandapower::{Network, load_csv_zip},
    prelude::*,
    prelude::{ecs::powerflow::systems::PowerFlowMat, jacobian_cache::JacobianCache},
    testcases::case_ieee39::IEEE_39,
};
use rustpower_amd::{
    Control,
    powerflow::{OrderingWorkspace, YbusPattern},
};
use serde_json::json;
use std::{fs, path::PathBuf};
fn main() {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "docs/experiments/amd-ybus/mapping-results".into()),
    );
    fs::create_dir_all(&out).unwrap();
    let mut summaries = vec![];
    for (case, net) in [
        ("39", serde_json::from_str::<Network>(IEEE_39).unwrap()),
        ("118", load_csv_zip("cases/IEEE118/data.zip").unwrap()),
        ("9241", load_csv_zip("cases/pegase9241/data.zip").unwrap()),
    ] {
        let mut g = PowerGrid::default();
        g.world_mut().insert_resource(PPNetwork(net));
        g.init_pf_net();
        let m = g.world().resource::<PowerFlowMat>();
        let a = m.npv + m.npq;
        let q = m.npq;
        let n = a + q;
        let pattern =
            YbusPattern::analyze(m.y_bus.col_offsets(), m.y_bus.row_indices(), m.npv, q).unwrap();
        let j = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            q,
        );
        let lengths: Vec<_> = pattern
            .columns()
            .iter()
            .map(|c| c.active_count + c.pq_count)
            .collect();
        let old_lengths: Vec<_> = (0..n)
            .map(|c| lengths[if c < a { c } else { c - a }])
            .collect();
        let old_offsets: Vec<_> = std::iter::once(0)
            .chain(old_lengths.iter().scan(0usize, |s, &v| {
                *s += v;
                Some(*s)
            }))
            .collect();
        assert_eq!(old_offsets, j.j_col_ptrs);
        let mut workspace = OrderingWorkspace::new();
        for kind in ["scalar", "seeded"] {
            let (p, inv) = if kind == "seeded" {
                let o = workspace.order(&pattern, &Control::default()).unwrap();
                (o.permutation().to_vec(), o.inverse_permutation().to_vec())
            } else {
                let (p, i, _) =
                    rustpower_amd::order(n, &j.j_col_ptrs, &j.j_row_indices, &Control::default())
                        .unwrap();
                (p, i)
            };
            let new_lengths: Vec<_> = p.iter().map(|&old| old_lengths[old]).collect();
            let predicted_offsets: Vec<_> = std::iter::once(0)
                .chain(new_lengths.iter().scan(0usize, |s, &v| {
                    *s += v;
                    Some(*s)
                }))
                .collect();
            // Independent reference: materialize existing J pattern, tag each stored
            // entry by its original position, then apply both scalar permutations via COO.
            let mut coo = CooMatrix::new(n, n);
            for c in 0..n {
                for pos in j.j_col_ptrs[c]..j.j_col_ptrs[c + 1] {
                    coo.push(inv[j.j_row_indices[pos]], inv[c], pos + 1);
                }
            }
            let actual = CscMatrix::from(&coo);
            // Column-only permutation preserves every within-column stored offset.
            let mut column_only_coo = CooMatrix::new(n, n);
            for c in 0..n {
                for pos in j.j_col_ptrs[c]..j.j_col_ptrs[c + 1] {
                    column_only_coo.push(j.j_row_indices[pos], inv[c], pos + 1);
                }
            }
            let column_only = CscMatrix::from(&column_only_coo);
            assert_eq!(column_only.col_offsets(), predicted_offsets);
            for (newcol, &oldcol) in p.iter().enumerate() {
                for local in 0..old_lengths[oldcol] {
                    assert_eq!(
                        column_only.values()[predicted_offsets[newcol] + local] - 1,
                        old_offsets[oldcol] + local
                    );
                }
            }

            assert_eq!(predicted_offsets, actual.col_offsets());
            let mut reference_map = vec![usize::MAX; j.j_row_indices.len()];
            for (newpos, &tag) in actual.values().iter().enumerate() {
                reference_map[tag - 1] = newpos;
            }
            // Direct route: read only Ybus plus shared column layout. Never read J rows.
            let mut direct_rows = vec![];
            let mut direct_map = vec![usize::MAX; j.j_row_indices.len()];
            let mut preserved_columns = 0;
            let mut unchanged_local_positions = 0;
            for (newcol, &oldcol) in p.iter().enumerate() {
                let bus = if oldcol < a { oldcol } else { oldcol - a };
                let layout = pattern.columns()[bus];
                let start = m.y_bus.col_offsets()[bus];
                let rows = &m.y_bus.row_indices()[start..start + layout.active_count];
                let mut entries: Vec<_> = rows
                    .iter()
                    .enumerate()
                    .map(|(local, &row)| (inv[row], old_offsets[oldcol] + local))
                    .chain(
                        rows[..layout.pq_count]
                            .iter()
                            .enumerate()
                            .map(|(local, &row)| {
                                (
                                    inv[a + row],
                                    old_offsets[oldcol] + layout.active_count + local,
                                )
                            }),
                    )
                    .collect();
                if entries.windows(2).all(|w| w[0].0 < w[1].0) {
                    preserved_columns += 1;
                }
                entries.sort_unstable_by_key(|e| e.0);
                for (local, (row, oldpos)) in entries.into_iter().enumerate() {
                    let newpos = predicted_offsets[newcol] + local;
                    direct_rows.push(row);
                    direct_map[oldpos] = newpos;
                    if oldpos - old_offsets[oldcol] == local {
                        unchanged_local_positions += 1;
                    }
                }
            }
            assert_eq!(direct_rows, actual.row_indices());
            assert_eq!(direct_map, reference_map);
            let pair_distances: Vec<_> = (0..q).map(|b| inv[b].abs_diff(inv[a + b])).collect();
            let adjacent = pair_distances.iter().filter(|&&d| d == 1).count();
            let mut seen = vec![false; a];
            let mut bus_order = vec![];
            for &v in &p {
                let b = if v < a { v } else { v - a };
                if !seen[b] {
                    seen[b] = true;
                    bus_order.push(b);
                }
            }
            let expanded_bus_order: Vec<_> = bus_order
                .iter()
                .flat_map(|&b| {
                    let mut v = vec![b];
                    if b < q {
                        v.push(a + b);
                    }
                    v.sort_unstable_by_key(|&v| inv[v]);
                    v
                })
                .collect();
            let collapses = expanded_bus_order == p;
            assert_eq!(collapses, adjacent == q);
            let summary = json!({"case":case,"ordering":kind,"npq":q,"npv":m.npv,"active_buses":a,"scalar_dimension":n,"j_nnz":direct_rows.len(),"offset_prediction_exact":true,"column_only_local_offsets_exact":true,"direct_rows_exact":true,"direct_position_map_exact":true,"adjacent_pq_pairs":adjacent,"pq_pairs":q,"max_pair_distance":pair_distances.iter().max(),"collapses_to_bus_permutation":collapses,"columns_preserving_local_row_order":preserved_columns,"total_columns":n,"unchanged_local_positions":unchanged_local_positions,"total_positions":direct_map.len()});
            println!("{summary}");
            summaries.push(summary.clone());
            let layouts:Vec<_>=pattern.columns().iter().enumerate().map(|(b,c)|json!({"bus":b,"original_bus":m.from_perm[b],"active_nnz":c.active_count,"pq_nnz":c.pq_count,"weight":if b<q{2}else{1},"j_column_length":lengths[b],"theta_new_position":inv[b],"magnitude_new_position":if b<q{Some(inv[a+b])}else{None}})).collect();
            fs::write(out.join(format!("case{case}-{kind}.json")),serde_json::to_vec(&json!({"summary":summary,"bus_layout":layouts,"permutation_new_to_old":p,"inverse_old_to_new":inv,"old_j_column_offsets":old_offsets,"predicted_new_j_column_offsets":predicted_offsets,"actual_new_j_column_offsets":actual.col_offsets(),"new_j_row_indices":direct_rows,"old_to_new_value_position":direct_map,"induced_bus_order":bus_order,"pq_pair_distances":pair_distances})).unwrap()).unwrap();
        }
    }
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
}
