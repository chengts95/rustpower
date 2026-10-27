//! Exact first-touch prototype: original scalar algorithm, no matrix-layout changes.
unsafe extern "C" {
    fn sched_getcpu() -> i32;
}
use rustpower::{
    io::pandapower::{Network, load_csv_zip},
    prelude::*,
    prelude::{ecs::powerflow::systems::PowerFlowMat, jacobian_cache::JacobianCache},
    testcases::case_ieee39::IEEE_39,
};
use rustpower_amd::{Control, exact, powerflow::YbusPattern};
use std::{fs::File, io::Write, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let reps: usize = args.get(1).map(|x| x.parse().unwrap()).unwrap_or(50);
    let mut out = File::create(&args[2]).unwrap();
    let ctl = Control::default();
    for (case, net) in [
        ("39", serde_json::from_str::<Network>(IEEE_39).unwrap()),
        ("118", load_csv_zip("cases/IEEE118/data.zip").unwrap()),
        ("9241", load_csv_zip("cases/pegase9241/data.zip").unwrap()),
    ] {
        let mut grid = PowerGrid::default();
        grid.world_mut().insert_resource(PPNetwork(net));
        grid.init_pf_net();
        let m = grid.world().resource::<PowerFlowMat>();
        let view = YbusPattern::analyze(m.y_bus.col_offsets(), m.y_bus.row_indices(), m.npv, m.npq)
            .unwrap();
        let j = JacobianCache::build_from_permuted(
            m.y_bus.col_offsets(),
            m.y_bus.row_indices(),
            m.npv,
            m.npq,
        );
        let reference = unsafe {
            rustpower_amd::order_unchecked(
                view.scalar_dimension(),
                &j.j_col_ptrs,
                &j.j_row_indices,
                &ctl,
            )
        }
        .unwrap();
        #[cfg(feature = "experimental_amd_trace")]
        {
            let (_, before) = rustpower_amd::trace::capture(|| {
                unsafe {
                    rustpower_amd::order_unchecked(
                        view.scalar_dimension(),
                        &j.j_col_ptrs,
                        &j.j_row_indices,
                        &ctl,
                    )
                }
                .unwrap()
            });
            let (_, after) = rustpower_amd::trace::capture(|| exact::order(&view, &ctl, true));
            assert_eq!(before, after, "phase trace differs {case}");
            eprintln!("{case}: full recorded pivot/update/merge trace equal");
        }
        for repeat in 0..reps + 5 {
            for step in 0..3 {
                let variant = (repeat + step) % 3;
                let cpu = unsafe { sched_getcpu() };
                if let Ok(expected) = std::env::var("BENCH_CPU") {
                    assert_eq!(cpu, expected.parse::<i32>().unwrap());
                }
                let t = Instant::now();
                let (p, inv, info, counters, input_us, kernel_us) = if variant == 0 {
                    let (p, i, info) = unsafe {
                        rustpower_amd::order_unchecked(
                            view.scalar_dimension(),
                            &j.j_col_ptrs,
                            &j.j_row_indices,
                            &ctl,
                        )
                    }
                    .unwrap();
                    (p, i, info, None, None, None)
                } else {
                    let r = exact::order(&view, &ctl, variant == 2);
                    (
                        r.permutation,
                        r.inverse,
                        r.info,
                        Some(r.counters),
                        Some(r.input_us),
                        Some(r.kernel_us),
                    )
                };
                let us = t.elapsed().as_secs_f64() * 1e6;
                assert_eq!(cpu, unsafe { sched_getcpu() });
                assert_eq!(p, reference.0, "permutation {case} variant{variant}");
                assert_eq!(inv, reference.1);
                assert_eq!(info.lnz, reference.2.lnz);
                assert_eq!(info.n_mult_subs_lu, reference.2.n_mult_subs_lu);
                if repeat >= 5 {
                    let c = counters.unwrap_or_default();
                    writeln!(out,"{}",serde_json::json!({"case":case,"variant":(["original_scalar","direct_input","first_touch"][variant]),"repeat":repeat-5,"us":us,"input_us":input_us,"kernel_us":kernel_us,"cpu":cpu,"lnz_estimate":info.lnz,"pristine_updates":c.pristine_updates,"payloads":c.pair_payloads,"pair_reuses":c.pair_reuses,"own_twin_skips":c.own_twin_skips,"seed_scanned":c.seed_entries_scanned,"payload_written":c.entries_written,"suffix_scans_avoided":c.suffix_scans_avoided,"permutation_exact":true})).unwrap();
                }
            }
        }
        eprintln!("{case}: all comparisons passed");
    }
}
