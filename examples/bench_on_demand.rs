//! On-demand initial neighbors; strict scalar control, no numerical/layout changes.
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
            let (_, after) = rustpower_amd::trace::capture(|| {
                rustpower_amd::on_demand::order(&view, &ctl).unwrap()
            });
            assert_eq!(before, after, "phase trace differs {case}");
            eprintln!("{case}: full recorded pivot/update/merge trace equal");
        }
        let counted = rustpower_amd::on_demand::Workspace::default()
            .order_counted(&view, &ctl)
            .unwrap()
            .counts;
        let mut normal_workspace = rustpower_amd::on_demand::FullTableControl::default();
        let mut lazy_workspace = rustpower_amd::on_demand::Workspace::default();
        for mode in ["fresh", "reuse"] {
            for repeat in 0..reps + 5 {
                for step in 0..2 {
                    let variant = (repeat + step) % 2;
                    let cpu = unsafe { sched_getcpu() };
                    if let Ok(expected) = std::env::var("BENCH_CPU") {
                        assert_eq!(cpu, expected.parse::<i32>().unwrap());
                    }
                    let t = Instant::now();
                    let r = {
                        let view = YbusPattern::analyze(
                            m.y_bus.col_offsets(),
                            m.y_bus.row_indices(),
                            m.npv,
                            m.npq,
                        )
                        .unwrap();
                        if variant == 0 {
                            if mode == "fresh" {
                                rustpower_amd::on_demand::FullTableControl::default()
                                    .order(&view, &ctl)
                            } else {
                                normal_workspace.order(&view, &ctl)
                            }
                        } else {
                            if mode == "fresh" {
                                rustpower_amd::on_demand::order(&view, &ctl).unwrap()
                            } else {
                                lazy_workspace.order(&view, &ctl).unwrap()
                            }
                        }
                    };
                    let us = t.elapsed().as_secs_f64() * 1e6;
                    if std::env::var_os("BENCH_CPU").is_some() {
                        assert_eq!(cpu, unsafe { sched_getcpu() });
                    }
                    assert_eq!(r.permutation, reference.0);
                    assert_eq!(r.inverse, reference.1);
                    assert_eq!(r.info.lnz, reference.2.lnz);
                    assert_eq!(r.info.n_mult_subs_lu, reference.2.n_mult_subs_lu);
                    if repeat >= 5 {
                        let c = if variant == 0 {
                            r.counts
                        } else {
                            let mut c = counted.clone();
                            c.allocated_value_slots = r.counts.allocated_value_slots;
                            c
                        };
                        writeln!(out,"{}",serde_json::json!({"case":case,"mode":mode,"variant":if variant==0{"full_initial_table"}else{"on_demand_neighbors"},"repeat":repeat-5,"us":us,"cpu":cpu,"scalar_variables":view.scalar_dimension(),"full_initial_entries":view.scalar_nnz()-view.scalar_dimension(),"initial_neighbor_reads":if variant==0{None}else{Some(c.initial_neighbor_reads)},"initial_neighbor_ids_saved":c.generated_neighbor_ids_saved,"dynamic_writes":if variant==0{None}else{Some(c.dynamic_writes)},"distinct_dynamic_slots":if variant==0{None}else{Some(c.distinct_dynamic_slots)},"initial_variables_read":if variant==0{None}else{Some(c.initial_variables_read)},"allocated_value_slots":c.allocated_value_slots,"arena_compactions":r.info.n_cmp_a,"permutation_exact":true,"counters_outside_timing":true})).unwrap();
                    }
                }
            }
        }
        eprintln!("{case}: all comparisons passed");
    }
}
