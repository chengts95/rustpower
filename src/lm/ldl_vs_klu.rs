//! LDL vs KLU backend comparison for the LM path.
//!
//! The LM augmented system `[μI Jᵀ; J −I]` is symmetric **quasi-definite**,
//! so the SuiteSparse LDL package (no-pivot LDLᵀ + AMD ordering) is the
//! theoretically correct tool, while KLU (unsymmetric LU) is what the
//! production NR path uses. These tests check that swapping the backend:
//!
//! 1. converges identically (same iterations, same solution) on IEEE39;
//! 2. keeps the same fold/infeasible behavior on the ill-conditioned case;
//! 3. reports the warm/cold wall-time breakdown on PEGASE9241.
//!
//! Run (release, all backends, with phase breakdowns):
//! `cargo test --release --features "klu ldl qdldl probe" ldl_vs_klu -- --nocapture`

#[cfg(all(test, feature = "klu", feature = "ldl"))]
mod tests {
    #[cfg(feature = "probe")]
    use crate::basic::solver::{klu_probe, ldl_probe, qdldl_probe};
    use crate::basic::solver::{KLUSolver, LDLSolver, Solve};
    use crate::lm::gn_flat::{GnDriver, newton_pf_gn};
    use crate::lm::residual::fixtures::load_ieee39_mat;
    use nalgebra::DVector;
    use num_complex::Complex64;
    use std::time::Instant;

    fn max_dv(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter().zip(b.iter()).fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
    }

    /// IEEE39: GN-LM with LDL must match KLU iteration-for-iteration and
    /// land on the same voltage vector (both solve the identical system).
    #[test]
    fn ldl_lm_ieee39_matches_klu() {
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let sbus = DVector::from_vec(mat.s_bus.iter().copied().collect::<Vec<_>>());
        let v_init = DVector::from_vec(mat.v_bus_init.iter().copied().collect::<Vec<_>>());

        let mut s_klu = KLUSolver::default();
        let (v_klu, it_klu) =
            newton_pf_gn(ybus, &sbus, &v_init, mat.npv, mat.npq, Some(1e-10), Some(100), &mut s_klu)
                .expect("GN-LM+KLU should converge");

        let mut s_ldl = LDLSolver::default();
        let (v_ldl, it_ldl) =
            newton_pf_gn(ybus, &sbus, &v_init, mat.npv, mat.npq, Some(1e-10), Some(100), &mut s_ldl)
                .expect("GN-LM+LDL should converge");

        let dv = max_dv(
            v_klu.as_slice(),
            v_ldl.as_slice(),
        );
        println!(
            "IEEE39 GN-LM: KLU it={it_klu} | LDL it={it_ldl} | max|ΔV|={dv:.3e} | LDL inertia check below"
        );
        // 惯性必须恰为 (n_δ, n_r)：准定性的直接证据。
        let mut probe = LDLSolver::default();
        {
            let mut driver = GnDriver::build(ybus, mat.npv, mat.npq, sbus.iter().copied().collect());
            let mut v = v_init.iter().copied().collect::<Vec<_>>();
            let _ = driver.solve_gn(ybus, &mut probe, &mut v, 1e-10, 100);
            let (pos, neg) = probe.inertia();
            let n_state = mat.npv + 2 * mat.npq;
            println!("LDL inertia after LM: (+{pos}, −{neg}) 期望 ({n_state}, {n_state})");
            assert_eq!(pos, n_state);
            assert_eq!(neg, n_state);
        }
        assert_eq!(it_klu, it_ldl, "backend swap must not change the μ trajectory");
        assert!(dv < 1e-9, "LDL and KLU disagree on the solution");
    }

    /// 病态算例：可解点两家必须同样收敛；无解点两家都必须走出
    /// 最小二乘轨迹（不发散、不挂零主元）。
    #[test]
    fn ldl_lm_ill_conditioned_behaves() {
        use crate::lm::residual::fixtures::ill_conditioned_case;
        let (ybus, npv, npq, v_star, s_spec) = ill_conditioned_case();
        let s_base = DVector::from_vec(s_spec);

        for (alpha, label) in [(1.0f64, "可解"), (1.2f64, "无解区")] {
            let sbus = DVector::from_vec(s_base.iter().map(|s| s * alpha).collect::<Vec<_>>());
            let v_init = DVector::from_vec(v_star.clone());

            let mut s_klu = KLUSolver::default();
            let r_klu = newton_pf_gn(
                &ybus, &sbus, &v_init, npv, npq, Some(1e-8), Some(200), &mut s_klu,
            );
            let mut s_ldl = LDLSolver::default();
            let r_ldl = newton_pf_gn(
                &ybus, &sbus, &v_init, npv, npq, Some(1e-8), Some(200), &mut s_ldl,
            );

            let (ck, ik, vk) = match &r_klu {
                Ok((v, i)) => (true, *i, v.clone()),
                Err((_, v, i)) => (false, *i, v.clone()),
            };
            let (cl, il, vl) = match &r_ldl {
                Ok((v, i)) => (true, *i, v.clone()),
                Err((_, v, i)) => (false, *i, v.clone()),
            };
            println!(
                "病态14 α={alpha:.2} ({label}): KLU conv={ck} it={ik} | LDL conv={cl} it={il} | max|ΔV|={:.3e}",
                max_dv(vk.as_slice(), vl.as_slice())
            );
            assert_eq!(ck, cl, "convergence verdict must match at α={alpha}");
            assert!(
                max_dv(vk.as_slice(), vl.as_slice()) < 1e-6,
                "least-squares point must match at α={alpha}"
            );
        }
    }

    /// PEGASE9241 性能对照：LM-GN 全迭代 wall time + 探针拆解。
    /// 口径与 perf_pf_app_api 一致：driver 级，cold + warm 各一次。
    /// 增广矩阵三个后端全登场：KLU（非对称 LU）、SuiteSparse LDL、
    /// Clarabel QDLDL（纯 Rust）。
    #[cfg(feature = "qdldl")]
    #[test]
    fn ldl_lm_perf_pegase9241() {
        use crate::basic::ecs::elements::PPNetwork;
        use crate::basic::ecs::network::{DataOps, PowerFlow, PowerGrid};
        use crate::basic::ecs::powerflow::systems::PowerFlowMat;
        use crate::io::pandapower::{Network, load_csv_zip};

        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net: Network = load_csv_zip(&format!("{dir}/cases/pegase9241/data.zip")).unwrap();
        let mut pf = PowerGrid::default();
        pf.world_mut().insert_resource(PPNetwork(net));
        pf.init_pf_net();
        let mat = pf.world().get_resource::<PowerFlowMat>().unwrap().clone();

        let ybus = &mat.y_bus;
        let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();

        println!("=== PEGASE9241 GN-LM: KLU vs LDL vs QDLDL ===");
        for name in ["KLU", "LDL", "QDLDL"] {
            for warm in [false, true] {
                let mut driver =
                    GnDriver::build(ybus, mat.npv, mat.npq, sbus.clone());
                let mut v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
                #[cfg(feature = "probe")]
                {
                    klu_probe::reset();
                    ldl_probe::reset();
                    qdldl_probe::reset();
                }
                let t = Instant::now();
                let res = match name {
                    "LDL" => {
                        let mut s = LDLSolver::default();
                        driver.solve_gn(ybus, &mut s, &mut v, 1e-8, 100)
                    }
                    "QDLDL" => {
                        let mut s = crate::basic::solver::QDLDLSolver::default();
                        driver.solve_gn(ybus, &mut s, &mut v, 1e-8, 100)
                    }
                    _ => {
                        let mut s = KLUSolver::default();
                        driver.solve_gn(ybus, &mut s, &mut v, 1e-8, 100)
                    }
                };
                let wall = t.elapsed();
                println!(
                    "{name} {}: conv={} it={} res={:.2e} wall={:.1}ms",
                    if warm { "warm" } else { "cold" },
                    res.converged,
                    res.iterations,
                    res.res_inf,
                    wall.as_secs_f64() * 1e3
                );
                #[cfg(feature = "probe")]
                match name {
                    "LDL" => println!("  {}", ldl_probe::report()),
                    "QDLDL" => println!("  {}", qdldl_probe::report()),
                    _ => println!("  {}", klu_probe::report()),
                }
                assert!(res.converged);
            }
        }
    }

    /// QDLDL 正确性：IEEE39 必须与 LDL 后端同迭代数、同解。
    #[cfg(feature = "qdldl")]
    #[test]
    fn qdldl_lm_ieee39_matches_ldl() {
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let sbus_vec: Vec<Complex64> = mat.s_bus.iter().copied().collect();

        let mut d1 = GnDriver::build(ybus, mat.npv, mat.npq, sbus_vec.clone());
        let mut v1: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let mut s1 = LDLSolver::default();
        let r1 = d1.solve_gn(ybus, &mut s1, &mut v1, 1e-10, 100);

        let mut d2 = GnDriver::build(ybus, mat.npv, mat.npq, sbus_vec);
        let mut v2: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let mut s2 = crate::basic::solver::QDLDLSolver::default();
        let r2 = d2.solve_gn(ybus, &mut s2, &mut v2, 1e-10, 100);

        let dv = max_dv(&v1, &v2);
        println!(
            "IEEE39 GN-LM: LDL it={} | QDLDL it={} | max|ΔV|={dv:.3e} | QDLDL positive_inertia={:?}",
            r1.iterations, r2.iterations, s2.positive_inertia()
        );
        assert!(r1.converged && r2.converged);
        assert_eq!(r1.iterations, r2.iterations);
        assert!(dv < 1e-9);
        // 准定惯性：(n_δ 正, n_r 负)，n_state = npv + 2·npq。
        assert_eq!(s2.positive_inertia(), Some(mat.npv + 2 * mat.npq));
    }
}
