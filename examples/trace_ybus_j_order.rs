//! Trace ordering decisions; never changes assembly, matrix layout, or algorithms.
use rustpower::{
    io::pandapower::{Network, load_csv_zip},
    prelude::*,
    prelude::{ecs::powerflow::systems::PowerFlowMat, jacobian_cache::JacobianCache},
    testcases::case_ieee39::IEEE_39,
};
use rustpower_amd::{
    Control,
    powerflow::{OrderingWorkspace, YbusPattern},
    trace::{Pivot, capture},
};
use serde_json::json;
use std::{fs, path::PathBuf};
fn pivots(x: &[Pivot]) -> serde_json::Value {
    json!(x.iter().map(|p|json!({"bus_or_variable":p.representative,"approx_external_degree":p.degree,"group_size":p.weight,"eliminated_scalars":p.eliminated,"candidate_states":p.candidates})).collect::<Vec<_>>())
}
fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "docs/experiments/amd-ybus/ordering-trace".into()),
    );
    fs::create_dir_all(&dir).unwrap();
    let mut summaries = vec![];
    for (case, net) in [
        ("39", serde_json::from_str::<Network>(IEEE_39).unwrap()),
        ("118", load_csv_zip("cases/IEEE118/data.zip").unwrap()),
        ("9241", load_csv_zip("cases/pegase9241/data.zip").unwrap()),
    ] {
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let m = grid.world().resource::<PowerFlowMat>();
        let a = m.npv + m.npq;
        let q = m.npq;
        let n = a + q;
        let pattern =
            YbusPattern::analyze(m.y_bus.col_offsets(), m.y_bus.row_indices(), m.npv, q).unwrap();
        let mut bp = vec![0];
        let mut bi = vec![];
        for b in 0..a {
            bi.extend(
                m.y_bus.row_indices()[m.y_bus.col_offsets()[b]..m.y_bus.col_offsets()[b + 1]]
                    .iter()
                    .copied()
                    .take_while(|&i| i < a),
            );
            bp.push(bi.len());
        }
        let ctl = Control::default();
        let (bus, bt) = capture(|| rustpower_amd::order(a, &bp, &bi, &ctl).unwrap());
        let j = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            q,
        );
        let (scalar, st) =
            capture(|| rustpower_amd::order(n, &j.j_col_ptrs, &j.j_row_indices, &ctl).unwrap());
        let (seeded, wt) = capture(|| OrderingWorkspace::new().order(&pattern, &ctl).unwrap());
        // Trace only: final permutations must remain identical to the already captured original references.
        let prior = PathBuf::from("docs/experiments/amd-ybus/ordering-correspondence");
        for (kind, actual) in [
            ("scalar_J", &scalar.0),
            ("seeded_J", &seeded.permutation().to_vec()),
        ] {
            let r: serde_json::Value = serde_json::from_slice(
                &fs::read(prior.join(format!("case{case}-{kind}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(json!(actual), r["j_amd_scalar"]);
            assert_eq!(json!(&bus.0), r["ybus_amd"]);
        }
        let pivot_prefix = bt
            .iter()
            .zip(&wt)
            .take_while(|(b, w)| b.representative == w.representative)
            .count();
        let first_divergence=bt.get(pivot_prefix).zip(wt.get(pivot_prefix)).map(|(b,w)|json!({"pivot_step":pivot_prefix,"bus_choice":b.representative,"bus_degree":b.degree,"bus_weight":b.weight,"seeded_choice":w.representative,"seeded_degree":w.degree,"seeded_weight":w.weight}));
        let mut degree_rule_count = 0;
        let mut changed_degrees = 0;
        let mut changed_scalar = 0;
        let mut buckets =
            std::collections::BTreeMap::<usize, std::collections::BTreeSet<usize>>::new();
        for b in 0..a {
            let rows = &bi[bp[b]..bp[b + 1]];
            let d = rows.len() - 1;
            let pv_neighbors = rows.iter().filter(|&&i| i != b && i >= q).count();
            let weighted = 2 * d - pv_neighbors;
            let own = if b < q { 2 } else { 1 };
            let js = j.j_row_indices[j.j_col_ptrs[b]..j.j_col_ptrs[b + 1]].len() - 1;
            assert_eq!(js, weighted + own - 1);
            degree_rule_count += 1;
            if weighted != d {
                changed_degrees += 1;
            }
            if js != d {
                changed_scalar += 1;
            }
            buckets.entry(d).or_default().insert(weighted);
        }
        let initial_min_bus = bt[0].candidates.iter().map(|x| x.1).min().unwrap();
        let initial_min_weighted = wt[0].candidates.iter().map(|x| x.1).min().unwrap();
        let min_bus: Vec<_> = bt[0]
            .candidates
            .iter()
            .filter(|x| x.1 == initial_min_bus)
            .map(|x| x.0)
            .collect();
        let min_weighted: Vec<_> = wt[0]
            .candidates
            .iter()
            .filter(|x| x.1 == initial_min_weighted)
            .map(|x| x.0)
            .collect();
        let report = json!({"case":case,"active_buses":a,"initial_degree_formula_verified":degree_rule_count,"buses_with_changed_external_degree":changed_degrees,"buses_with_changed_scalar_degree":changed_scalar,"initial_bus_min_degree":initial_min_bus,"initial_weighted_min_degree":initial_min_weighted,"initial_bus_min_candidates":min_bus,"initial_weighted_min_candidates":min_weighted,"initial_bucket_splits":buckets,"bus_pivot_events":bt.len(),"seeded_pivot_events":wt.len(),"same_pivot_prefix":pivot_prefix,"first_pivot_divergence":first_divergence,"bus_first8":bt.iter().take(8).map(|p|vec![p.representative,p.degree,p.weight]).collect::<Vec<_>>(),"seeded_first8":wt.iter().take(8).map(|p|vec![p.representative,p.degree,p.weight]).collect::<Vec<_>>()});
        println!("{report}");
        summaries.push(report.clone());
        fs::write(dir.join(format!("case{case}.json")),serde_json::to_vec(&json!({"summary":report,"bus_trace":pivots(&bt),"scalar_j_trace":pivots(&st),"seeded_j_trace":pivots(&wt)})).unwrap()).unwrap();
    }
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
}
