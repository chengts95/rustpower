//! Ordering correspondence only. Does not reorder any numerical matrix or change solving.
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
#[link(name = "amd")]
unsafe extern "C" {
    fn amd_l_order(
        n: i64,
        p: *const i64,
        i: *const i64,
        q: *mut i64,
        control: *const f64,
        info: *mut f64,
    ) -> i32;
}
fn c_order(n: usize, p: &[usize], i: &[usize]) -> Vec<usize> {
    let mut q = vec![0i64; n];
    let mut info = [0.; 20];
    assert_eq!(
        unsafe {
            amd_l_order(
                n as i64,
                p.as_ptr().cast(),
                i.as_ptr().cast(),
                q.as_mut_ptr(),
                std::ptr::null(),
                info.as_mut_ptr(),
            )
        },
        0
    );
    q.into_iter().map(|x| x as usize).collect()
}
fn rank(order: &[usize]) -> Vec<usize> {
    let mut r = vec![0; order.len()];
    for (k, &v) in order.iter().enumerate() {
        r[v] = k;
    }
    r
}
fn inversions(values: &[usize]) -> u64 {
    let n = values.len();
    let mut bit = vec![0u64; n + 1];
    let mut inv = 0;
    for (seen, &v) in values.iter().enumerate() {
        let mut k = v + 1;
        let mut prefix = 0;
        while k > 0 {
            prefix += bit[k];
            k &= k - 1;
        }
        inv += seen as u64 - prefix;
        let mut k = v + 1;
        while k <= n {
            bit[k] += 1;
            k += k & k.wrapping_neg();
        }
    }
    inv
}
fn main() {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "docs/experiments/amd-ybus/ordering-correspondence".into()),
    );
    fs::create_dir_all(&dir).unwrap();
    let mut reports = vec![];
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
        let n = a + m.npq;
        // Same active bus graph and diagonal as the exact J expansion, no slack nodes.
        let mut bp = vec![0];
        let mut bi = vec![];
        for b in 0..a {
            bi.extend(
                m.y_bus.row_indices()[m.y_bus.col_offsets()[b]..m.y_bus.col_offsets()[b + 1]]
                    .iter()
                    .copied()
                    .take_while(|&r| r < a),
            );
            bp.push(bi.len());
        }
        let bus = c_order(a, &bp, &bi);
        let bus_rust = rustpower_amd::order(a, &bp, &bi, &Control::default())
            .unwrap()
            .0;
        assert_eq!(bus, bus_rust);
        let br = rank(&bus);
        let j = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            m.npq,
        );
        let scalar = c_order(n, &j.j_col_ptrs, &j.j_row_indices);
        let pattern =
            YbusPattern::analyze(m.y_bus.col_offsets(), m.y_bus.row_indices(), m.npv, m.npq)
                .unwrap();
        let seeded = OrderingWorkspace::new()
            .order(&pattern, &Control::default())
            .unwrap()
            .permutation()
            .to_vec();
        for (kind, p) in [("scalar_J", scalar), ("seeded_J", seeded)] {
            let angles: Vec<_> = p.iter().copied().filter(|&v| v < a).collect();
            let ar = rank(&angles);
            let same = bus.iter().zip(&angles).filter(|(x, y)| x == y).count();
            let prefix = bus.iter().zip(&angles).take_while(|(x, y)| x == y).count();
            let relative: Vec<_> = angles.iter().map(|&b| br[b]).collect();
            let inversions = inversions(&relative);
            let pairs = a as u64 * (a as u64 - 1) / 2;
            let mut same_edges = 0;
            let mut edges = 0;
            for b in 0..a {
                for &i in &bi[bp[b]..bp[b + 1]] {
                    if i > b {
                        edges += 1;
                        if (br[b] < br[i]) == (ar[b] < ar[i]) {
                            same_edges += 1;
                        }
                    }
                }
            }
            let mut seen = vec![false; a];
            let mut first = vec![];
            for &v in &p {
                let b = if v < a { v } else { v - a };
                if !seen[b] {
                    seen[b] = true;
                    first.push(b);
                }
            }
            let fr = rank(&first);
            let first_relative: Vec<_> = first.iter().map(|&b| br[b]).collect();
            let summary = json!({"case":case,"comparison":kind,"active":a,"npq":m.npq,"npv":m.npv,"identical_angle_order":bus==angles,"same_angle_positions":same,"same_angle_prefix":prefix,"angle_pair_order_agreement_pct":100.*(1.-inversions as f64/pairs as f64),"angle_mean_rank_displacement":(0..a).map(|b|br[b].abs_diff(ar[b])).sum::<usize>() as f64/a as f64,"network_edges":edges,"edge_direction_agreement_pct":100.*same_edges as f64/edges as f64,"first_occurrence_identical":bus==first,"first_occurrence_pair_order_agreement_pct":100.*(1.-inversions_fn(&first_relative) as f64/pairs as f64),"first_occurrence_mean_rank_displacement":(0..a).map(|b|br[b].abs_diff(fr[b])).sum::<usize>() as f64/a as f64,"bus_rust_matches_c":true});
            println!("{summary}");
            reports.push(summary.clone());
            fs::write(dir.join(format!("case{case}-{kind}.json")),serde_json::to_vec(&json!({"summary":summary,"ybus_amd":bus,"j_amd_scalar":p,"j_angle_subsequence":angles,"j_bus_first_occurrence":first,"bus_pattern_offsets":bp,"bus_pattern_rows":bi})).unwrap()).unwrap();
        }
    }
    fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .unwrap();
}
fn inversions_fn(x: &[usize]) -> u64 {
    inversions(x)
}
