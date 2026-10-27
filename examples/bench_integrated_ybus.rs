#[cfg(target_os = "linux")]
mod bench {
    // Ordinary native solve entry; both policies in one release binary.
    use bevy_ecs::prelude::World;
    use rustpower::native::amd_solver::AmdNewtonSolver as NewtonSolver;
    use rustpower::prelude::ecs::{
        network::apply_permutation,
        powerflow::systems::{OriginalYBus, PowerFlowMat},
    };
    use rustpower::{
        io::pandapower::{Network, load_csv_zip},
        prelude::*,
        testcases::case_ieee39::IEEE_39,
    };
    use serde_json::json;
    use std::{fs::File, io::Write, time::Instant};
    #[repr(C)]
    #[derive(Default)]
    struct Trace {
        amd_us: f64,
        prep_us: f64,
        amd_calls: u64,
    }
    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlsym(
            handle: *mut std::ffi::c_void,
            name: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void;
        fn sched_getcpu() -> i32;
    }
    struct TraceApi {
        reset: unsafe extern "C" fn(),
        get: unsafe extern "C" fn(*mut Trace),
        analysis: unsafe extern "C" fn() -> f64,
    }
    fn trace_api() -> &'static TraceApi {
        static API: std::sync::OnceLock<TraceApi> = std::sync::OnceLock::new();
        API.get_or_init(||unsafe{
  let reset=dlsym(std::ptr::null_mut(),c"rp_trace_reset".as_ptr());let get=dlsym(std::ptr::null_mut(),c"rp_trace_get".as_ptr());let analysis=dlsym(std::ptr::null_mut(),c"rp_klu_analysis_us".as_ptr());
  assert!(!reset.is_null()&&!get.is_null()&&!analysis.is_null(),"Run docs/experiments/amd-ybus/run-integration-bench.sh to load the measurement helper");
  TraceApi{reset:std::mem::transmute(reset),get:std::mem::transmute(get),analysis:std::mem::transmute(analysis)}
 })
    }
    unsafe fn rp_trace_reset() {
        unsafe { (trace_api().reset)() }
    }
    unsafe fn rp_trace_get(t: *mut Trace) {
        unsafe { (trace_api().get)(t) }
    }
    unsafe fn rp_klu_analysis_us() -> f64 {
        unsafe { (trace_api().analysis)() }
    }

    fn native(m: &PowerFlowMat, optimized: bool, btf: i32) -> NewtonSolver {
        let mut s = NewtonSolver::new();
        s.mat = Some(m.clone());
        s.set_ybus_ordering(optimized);
        unsafe {
            (*s.linear_solver.0.common).btf = btf;
        }
        s
    }
    pub fn run() {
        let args: Vec<_> = std::env::args().collect();
        let reps = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100usize);
        let mut out = File::create(
            args.get(2)
                .map(String::as_str)
                .unwrap_or("integrated.jsonl"),
        )
        .unwrap();
        let btf = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
        let expected_cpu: Option<i32> = std::env::var("BENCH_CPU").ok().map(|s| s.parse().unwrap());
        for (case, net) in [
            ("IEEE39", serde_json::from_str::<Network>(IEEE_39).unwrap()),
            ("IEEE118", load_csv_zip("cases/IEEE118/data.zip").unwrap()),
            (
                "PEGASE9241",
                load_csv_zip("cases/pegase9241/data.zip").unwrap(),
            ),
        ] {
            let mut g = PowerGrid::default();
            g.world_mut().insert_resource(PPNetwork(net));
            g.init_pf_net();
            let m = g.world().resource::<PowerFlowMat>().clone();
            let mut original = m.clone();
            original.y_bus = g.world().resource::<OriginalYBus>().0.clone();
            for (new, &old) in m.from_perm.iter().enumerate() {
                original.s_bus[old] = m.s_bus[new];
                original.v_bus_init[old] = m.v_bus_init[new];
            }
            let mut world = World::new();
            let mut baseline = native(&m, false, btf);
            assert!(baseline.solve(100, 1e-10).unwrap().0);
            let target = baseline.voltage().unwrap();
            let expected_iterations = baseline.iterations().unwrap();
            for mode in ["cold", "reuse"] {
                let mut solvers = [native(&m, false, btf), native(&m, true, btf)];
                for pair in 0..(reps + 6) {
                    for path in if pair % 2 == 0 {
                        [0usize, 1]
                    } else {
                        [1usize, 0]
                    } {
                        if mode == "cold" {
                            solvers[path] = native(&m, path == 1, btf);
                        }
                        world.insert_resource(original.clone());
                        let t = Instant::now();
                        world.run_system_cached(apply_permutation).unwrap();
                        let permutation_us = t.elapsed().as_secs_f64() * 1e6;
                        assert_eq!(world.resource::<PowerFlowMat>().y_bus, m.y_bus);
                        let s = &mut solvers[path];
                        let before = s.ybus_ordering_stats().preparations;

                        unsafe {
                            rp_trace_reset();
                        }
                        let cpu_before = unsafe { sched_getcpu() };
                        if let Some(cpu) = expected_cpu {
                            assert_eq!(cpu_before, cpu, "benchmark affinity changed");
                        }
                        let t = Instant::now();
                        let (ok, residual) = s.solve(100, 1e-10).unwrap();
                        let full_us = t.elapsed().as_secs_f64() * 1e6;
                        let cpu_after = unsafe { sched_getcpu() };
                        if let Some(cpu) = expected_cpu {
                            assert_eq!(cpu_after, cpu, "benchmark affinity changed");
                        }
                        let mut trace = Trace::default();
                        unsafe {
                            rp_trace_get(&mut trace);
                        }
                        let ordering = s.ybus_ordering_stats();
                        let prepared = ordering.preparations != before;
                        let ordering_us = if path == 1 && prepared {
                            ordering.last_preparation.as_secs_f64() * 1e6
                        } else {
                            unsafe { rp_klu_analysis_us() }
                        };
                        // J-cache timing is not instrumented in the legacy NR path.
                        let j_us: Option<f64> = None;
                        let error = s
                            .voltage()
                            .unwrap()
                            .iter()
                            .zip(&target)
                            .map(|(a, b)| (*a - *b).norm())
                            .fold(0., f64::max);
                        assert!(ok && residual < 1e-10 && error < 1e-8);
                        assert_eq!(s.iterations().unwrap(), expected_iterations);
                        if path == 1 {
                            assert!(ordering.optimized);
                            assert_eq!(trace.amd_calls, 0);
                        } else if mode == "cold" {
                            assert_eq!(trace.amd_calls, 1);
                        }
                        if pair >= 6 {
                            unsafe {
                                let k = &s.linear_solver.0;
                                let sy = &*k.symbolic;
                                let num = &*k.numeric;
                                writeln!(out,"{}",json!({"cpu_before":cpu_before,"cpu_after":cpu_after,"case":case,"mode":mode,"pair":pair-6,"path":path,"btf":btf,"permutation_us":permutation_us,"ordering_us":ordering_us,"j_symbolic_us":j_us,"net_preparation_us":permutation_us+ordering_us,"net_preparation_includes_j":false,"full_nr_us":full_us,"full_with_permutation_us":full_us+permutation_us,"amd_call_us":trace.amd_us,"amd_calls":trace.amd_calls,"optimized":ordering.optimized,"ordering_generation":ordering.preparations,"pattern_key_bytes":ordering.pattern_key_bytes,"iterations":s.iterations().unwrap(),"converged":ok,"residual":residual,"max_voltage_error":error,"nblocks":sy.nblocks,"do_btf":sy.do_btf,"lnz":num.lnz,"unz":num.unz})).unwrap();
                            }
                        }
                    }
                }
            }
            eprintln!(
                "{case}: actual native solve verified; {expected_iterations} NR steps, BTF={btf}"
            );
            out.flush().unwrap();
        }
    }
}
#[cfg(target_os = "linux")]
fn main() {
    bench::run()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The benchmark measurement helper requires Linux.")
}
