//! 正规方程收敛与电压结果的回归测试。
//!
//! Run: `cargo test --release --features "klu ldl" normal_eq -- --nocapture`

use super::*;
use crate::basic::solver::LDLSolver;
use crate::lm::residual::fixtures::load_ieee39_mat;

fn max_dv(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
}

/// IEEE39：正规方程和增广方程应收敛到相同电压。
#[cfg(feature = "ldl")]
#[test]
fn ne_ieee39_matches_augmented() {
    let mat = load_ieee39_mat();
    let ybus = &mat.y_bus;
    let sbus_vec: Vec<Complex64> = mat.s_bus.iter().copied().collect();

    // 对照：相同 LDL 后端的增广方程。
    let mut ref_driver = crate::lm::gn_flat::GnDriver::build(ybus, mat.npv, mat.npq, sbus_vec.clone());
    let mut v_ref: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
    let mut s_ref = LDLSolver::default();
    let r_ref = ref_driver.solve_gn(ybus, &mut s_ref, &mut v_ref, 1e-10, 100);
    assert!(r_ref.converged);

    // 默认缓存乘积结构的正规方程。
    let mut ne = NeDriver::build(ybus, mat.npv, mat.npq, sbus_vec);
    let mut v_ne: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
    let mut s_ne = LDLSolver::default();
    let r_ne = ne.solve_ne(ybus, &mut s_ne, &mut v_ne, 1e-10, 100);

    let dv = max_dv(&v_ref, &v_ne);
    println!(
        "IEEE39 正规方程 vs 增广方程: it {} vs {} | max|ΔV|={dv:.3e} | conv={}",
        r_ne.iterations, r_ref.iterations, r_ne.converged
    );
    assert!(r_ne.converged);
    assert!(dv < 1e-8, "normal equations converge to a different voltage");
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
