//! Tests for the dumb normal-equations baseline (`NeDriver`).
//!
//! Run: `cargo test --release --features "klu ldl" normal_eq -- --nocapture`

use super::*;
use crate::basic::solver::LDLSolver;
use crate::lm::residual::fixtures::load_ieee39_mat;

fn max_dv(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
}

/// IEEE39: the dumb NE path must converge to the same point as the
/// optimized augmented path (same LM math, same μ rules).
#[cfg(feature = "ldl")]
#[test]
fn ne_ieee39_matches_augmented() {
    let mat = load_ieee39_mat();
    let ybus = &mat.y_bus;
    let sbus_vec: Vec<Complex64> = mat.s_bus.iter().copied().collect();

    // Reference: optimized augmented path, same LDL backend.
    let mut ref_driver = crate::lm::gn_flat::GnDriver::build(ybus, mat.npv, mat.npq, sbus_vec.clone());
    let mut v_ref: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
    let mut s_ref = LDLSolver::default();
    let r_ref = ref_driver.solve_gn(ybus, &mut s_ref, &mut v_ref, 1e-10, 100);
    assert!(r_ref.converged);

    // Dumb NE path.
    let mut ne = NeDriver::build(ybus, mat.npv, mat.npq, sbus_vec);
    let mut v_ne: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
    let mut s_ne = LDLSolver::default();
    let r_ne = ne.solve_ne(ybus, &mut s_ne, &mut v_ne, 1e-10, 100);

    let dv = max_dv(&v_ref, &v_ne);
    println!(
        "IEEE39 NE(笨) vs 增广(优): it {} vs {} | max|ΔV|={dv:.3e} | conv={}",
        r_ne.iterations, r_ref.iterations, r_ne.converged
    );
    assert!(r_ne.converged);
    assert!(dv < 1e-8, "dumb NE path lands on a different point");
}

/// 病态算例：无解区两家必须都走最小二乘轨迹且终点一致；
/// 顺带观察 κ² 是否把 NE 的迭代轨迹打歪（它可能步数更多）。
#[cfg(feature = "ldl")]
#[test]
fn ne_ill_conditioned_behaves() {
    use crate::lm::residual::fixtures::ill_conditioned_case;
    let (ybus, npv, npq, v_star, s_spec) = ill_conditioned_case();

    for (alpha, label) in [(1.0f64, "可解"), (1.2f64, "无解区")] {
        let sbus: Vec<Complex64> = s_spec.iter().map(|s| s * alpha).collect();

        let mut ref_driver = crate::lm::gn_flat::GnDriver::build(&ybus, npv, npq, sbus.clone());
        let mut v_ref = v_star.clone();
        let mut s_ref = LDLSolver::default();
        let r_ref = ref_driver.solve_gn(&ybus, &mut s_ref, &mut v_ref, 1e-8, 200);

        let mut ne = NeDriver::build(&ybus, npv, npq, sbus);
        let mut v_ne = v_star.clone();
        let mut s_ne = LDLSolver::default();
        let r_ne = ne.solve_ne(&ybus, &mut s_ne, &mut v_ne, 1e-8, 200);

        println!(
            "病态14 α={alpha:.2} ({label}): 增广 conv={} it={} | NE conv={} it={} | max|ΔV|={:.3e}",
            r_ref.converged, r_ref.iterations, r_ne.converged, r_ne.iterations,
            max_dv(&v_ref, &v_ne)
        );
        assert_eq!(r_ref.converged, r_ne.converged, "convergence verdict mismatch at α={alpha}");
        assert!(max_dv(&v_ref, &v_ne) < 1e-5, "least-squares point mismatch at α={alpha}");
    }
}

/// PEGASE9241 性能：NE 路径两种模式（笨=每轮重做符号 / 聪明=缓存 pattern
/// 纯数值双指针乘法）的 wall time 与拆解；对照增广路径已测的
/// LDL ≈147ms、KLU ≈460ms。
#[cfg(all(feature = "klu", feature = "ldl"))]
#[test]
fn ne_perf_pegase9241() {
    use crate::basic::ecs::elements::PPNetwork;
    use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
    use crate::basic::ecs::powerflow::systems::PowerFlowMat;
    #[cfg(feature = "probe")]
    use crate::basic::solver::ldl_probe;
    use crate::io::pandapower::{Network, load_csv_zip};
    use std::time::Instant;

    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net: Network = load_csv_zip(&format!("{dir}/cases/pegase9241/data.zip")).unwrap();
    let mut pf = PowerGrid::default();
    pf.world_mut().insert_resource(PPNetwork(net));
    pf.init_pf_net();
    let mat = pf.world().get_resource::<PowerFlowMat>().unwrap().clone();

    let ybus = &mat.y_bus;
    let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();

    println!("=== PEGASE9241 NE 路径（LDL 后端）：笨 vs 聪明 ===");
    for dumb in [true, false] {
        for warm in [false, true] {
            let mut ne = NeDriver::build(ybus, mat.npv, mat.npq, sbus.clone());
            ne.dumb_mode = dumb;
            let mut v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
            let mut s = LDLSolver::default();
            #[cfg(feature = "probe")]
            ldl_probe::reset();
            ne.reset_prof();
            let t = Instant::now();
            let res = ne.solve_ne(ybus, &mut s, &mut v, 1e-8, 100);
            let wall = t.elapsed();
            println!(
                "NE {} {}: conv={} it={} res={:.2e} wall={:.1}ms",
                if dumb { "笨  " } else { "聪明" },
                if warm { "warm" } else { "cold" },
                res.converged, res.iterations, res.res_inf,
                wall.as_secs_f64() * 1e3
            );
            println!(
                "  driver: J fill={:.1}ms spgemm符号={:.1}ms JᵀJ数值={:.1}ms μ={:.1}ms",
                ne.prof_fill_ns as f64 / 1e6,
                ne.prof_spgemm_ns as f64 / 1e6,
                ne.prof_numeric_ns as f64 / 1e6,
                ne.prof_mu_ns as f64 / 1e6,
            );
            #[cfg(feature = "probe")]
            println!("  {}", ldl_probe::report());
            assert!(res.converged);
        }
    }
}
