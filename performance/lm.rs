//! LM 对比入口：实现一致性、正规方程与增广方程、线性求解后端。
//! 各路径共用步长控制，在一次潮流内部复用求解器。
use crate::basic::solver::{KLUSolver, QDLDLSolver};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::lm::{gn_flat::GnDriver, gn_triu::GnTriuDriver};
use serde::Serialize;
use std::time::Instant;

pub(super) struct Case {
    name: String,
    y: CscMatrix<Complex64>,
    s: Vec<Complex64>,
    v: Vec<Complex64>,
    npv: usize,
    npq: usize,
}

pub(super) fn load_case(name: &str) -> Case {
    use crate::basic::ecs::{
        elements::PPNetwork,
        network::{DataOps, PowerFlow, PowerGrid},
        powerflow::systems::PowerFlowMat,
    };
    let path = format!("{}/cases/{name}/data.zip", env!("CARGO_MANIFEST_DIR"));
    let network = crate::io::pandapower::load_csv_zip(&path).unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(network));
    grid.init_pf_net();
    let mat = grid.world().get_resource::<PowerFlowMat>().unwrap();
    Case {
        name: name.into(),
        y: mat.y_bus.clone(),
        s: mat.s_bus.as_slice().to_vec(),
        v: mat.v_bus_init.as_slice().to_vec(),
        npv: mat.npv,
        npq: mat.npq,
    }
}

#[derive(serde::Deserialize)]
struct Export {
    nb: usize,
    npv: usize,
    npq: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    y_re: Vec<f64>,
    y_im: Vec<f64>,
    s_re: Vec<f64>,
    s_im: Vec<f64>,
    v_re: Vec<f64>,
    v_im: Vec<f64>,
}
pub(super) fn load_6515(init: &str) -> Case {
    let path = format!(
        "{}/target/research/lm_audit/6515rte_{init}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let data: Export = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let complex = |re: &[f64], im: &[f64]| {
        re.iter()
            .zip(im)
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect()
    };
    Case {
        name: format!("6515rte_{init}"),
        npv: data.npv,
        npq: data.npq,
        y: CscMatrix::try_from_csc_data(
            data.nb,
            data.nb,
            data.cp,
            data.ri,
            complex(&data.y_re, &data.y_im),
        )
        .unwrap(),
        s: complex(&data.s_re, &data.s_im),
        v: complex(&data.v_re, &data.v_im),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) enum Method {
    FullBaseline,
    FullOperator,
    UpperBaseline,
    UpperOperator,
}

#[derive(Serialize)]
pub(super) struct Measurement {
    pub build_ms: f64,
    pub solve_ms: f64,
    pub fill_ms: f64,
    pub mu_ms: f64,
    pub linear_solve_ms: f64,
    pub iterations: usize,
    pub linear_solves: u64,
    pub converged: bool,
    pub residual_inf: f64,
}

fn independent_residual(case: &Case, v: &[Complex64]) -> f64 {
    let current = &case.y * &DVector::from_column_slice(v);
    let mut result = 0.0_f64;
    for i in 0..case.npv + case.npq {
        let mismatch = v[i] * current[i].conj() - case.s[i];
        if !mismatch.re.is_finite() || !mismatch.im.is_finite() {
            return f64::INFINITY;
        }
        result = result.max(mismatch.re.abs());
        if i < case.npq {
            result = result.max(mismatch.im.abs());
        }
    }
    result
}

pub(super) fn run_with_options(
    case: &Case,
    method: Method,
    max_iter: usize,
    options: &rustpower::lm::LmOptions,
) -> (Measurement, Vec<Complex64>) {
    let mut v = case.v.clone();
    let start = Instant::now();
    let (build_ms, solve_ms, fill_ns, mu_ns, linear_ns, iterations, linear_solves, converged) =
        match method {
            Method::FullBaseline | Method::FullOperator => {
                let mut driver = if matches!(method, Method::FullOperator) {
                    GnDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone())
                } else {
                    GnDriver::build(&case.y, case.npv, case.npq, case.s.clone())
                };
                let mut solver = KLUSolver::default();
                let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                let solve_start = Instant::now();
                let result = driver.solve_gn_with_options(
                    &case.y,
                    &mut solver,
                    &mut v,
                    1e-8,
                    max_iter,
                    options,
                );
                (
                    build_ms,
                    solve_start.elapsed().as_secs_f64() * 1000.0,
                    driver.prof_fill_ns,
                    driver.prof_mu_ns,
                    driver.prof_solve_ns,
                    result.iterations,
                    driver.n_solves,
                    result.converged,
                )
            }
            Method::UpperBaseline | Method::UpperOperator => {
                let mut driver = if matches!(method, Method::UpperOperator) {
                    GnTriuDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone())
                } else {
                    GnTriuDriver::build(&case.y, case.npv, case.npq, case.s.clone())
                };
                let mut solver = QDLDLSolver::default();
                let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                let solve_start = Instant::now();
                let result = driver.solve_gn_with_options(
                    &case.y,
                    &mut solver,
                    &mut v,
                    1e-8,
                    max_iter,
                    options,
                );
                (
                    build_ms,
                    solve_start.elapsed().as_secs_f64() * 1000.0,
                    driver.prof_fill_ns,
                    driver.prof_mu_ns,
                    driver.prof_solve_ns,
                    result.iterations,
                    driver.n_solves,
                    result.converged,
                )
            }
        };
    let residual_inf = independent_residual(case, &v);
    if converged {
        assert!(
            residual_inf < 1e-8,
            "{} {method:?}: {residual_inf:e}",
            case.name
        );
    }
    (
        Measurement {
            build_ms,
            solve_ms,
            fill_ms: fill_ns as f64 / 1e6,
            mu_ms: mu_ns as f64 / 1e6,
            linear_solve_ms: linear_ns as f64 / 1e6,
            iterations,
            linear_solves,
            converged,
            residual_inf,
        },
        v,
    )
}

fn compare_pair(
    case: &Case,
    old: Method,
    new: Method,
    max_iter: usize,
    options: &rustpower::lm::LmOptions,
) {
    let (a, va) = run_with_options(case, old, max_iter, options);
    let (b, vb) = run_with_options(case, new, max_iter, options);
    assert_eq!(a.converged, b.converged, "{}: {old:?}/{new:?}", case.name);
    assert!(a.converged, "{} did not converge", case.name);
    assert_eq!(
        a.iterations, b.iterations,
        "{}: iteration counts",
        case.name
    );
    assert_eq!(
        a.linear_solves, b.linear_solves,
        "{}: LM trial counts",
        case.name
    );
    let error = va
        .iter()
        .zip(vb)
        .map(|(a, b)| (*a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(error < 1e-8, "{}: voltage difference {error:e}", case.name);
}

pub fn operator_lm_matches_original_drivers() {
    for name in ["IEEE39", "IEEE118"] {
        let case = load_case(name);
        for metric in [
            rustpower::lm::DampingMetric::Polar,
            rustpower::lm::DampingMetric::CartesianVoltage,
            rustpower::lm::DampingMetric::YbusDiagonal,
        ] {
            let options = rustpower::lm::LmOptions {
                damping_metric: metric,
                ..Default::default()
            };
            compare_pair(
                &case,
                Method::FullBaseline,
                Method::FullOperator,
                100,
                &options,
            );
            compare_pair(
                &case,
                Method::UpperBaseline,
                Method::UpperOperator,
                100,
                &options,
            );
            let (reference, voltage) = run_with_options(&case, Method::UpperOperator, 300, &options);
            for full_slice in [false, true] {
                for upper_only in [false, true] {
                    let (coo, coo_voltage, _) = run_coo_baseline(&case, full_slice, upper_only, &options);
                    assert!(coo.converged, "{name}: COO did not converge");
                    assert_eq!(coo.iterations, reference.iterations);
                    assert_eq!(coo.linear_solves, reference.linear_solves);
                    let error = voltage.iter().zip(&coo_voltage)
                        .map(|(a, b)| (*a - *b).norm()).fold(0.0_f64, f64::max);
                    assert!(error < 1e-8, "{name}: COO voltage difference {error}");
                }
            }
        }
    }
}

/// COO仍逐次组装和转换；与其他路径共用参数，并复用求解器。
fn run_coo_baseline(case: &Case, full_slice: bool, upper_only: bool, options: &rustpower::lm::LmOptions) -> (Measurement, Vec<Complex64>, f64) {
    use rustpower::lm::baseline::{aug_coo::AugCooDriver, full_slice::AugFsDriver};
    let mut v = case.v.clone();
    let t = Instant::now();
    let (build_ms, solve_ms, converged, iterations, fill_ns, coo_ns, mu_ns, linear_ns, solves) =
        if full_slice {
            let mut d = AugFsDriver::build(&case.y, case.npv, case.npq, case.s.clone());
            d.upper_only = upper_only;
            let build_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let r = d.solve_aug_fs_with_options(&case.y, &mut v, 1e-8, 300, options);
            (
                build_ms,
                t.elapsed().as_secs_f64() * 1000.0,
                r.converged,
                r.iterations,
                d.prof_full_j_ns,
                d.prof_slice_coo_ns,
                d.prof_mu_ns,
                d.prof_solve_ns,
                d.n_solves,
            )
        } else {
            let mut d = AugCooDriver::build(&case.y, case.npv, case.npq, case.s.clone());
            d.upper_only = upper_only;
            let build_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            let r = d.solve_aug_coo_with_options(&case.y, &mut v, 1e-8, 300, options);
            (
                build_ms,
                t.elapsed().as_secs_f64() * 1000.0,
                r.converged,
                r.iterations,
                d.prof_fill_ns,
                d.prof_coo_ns,
                d.prof_mu_ns,
                d.prof_solve_ns,
                d.n_solves,
            )
        };
    let residual_inf = independent_residual(case, &v);
    if converged {
        assert!(residual_inf < 1e-8);
    }
    (
        Measurement {
            build_ms,
            solve_ms,
            fill_ms: fill_ns as f64 / 1e6,
            mu_ms: mu_ns as f64 / 1e6,
            linear_solve_ms: linear_ns as f64 / 1e6,
            iterations,
            linear_solves: solves,
            converged,
            residual_inf,
        },
        v,
        coo_ns as f64 / 1e6,
    )
}

/// 比较同一个潮流问题的八条实现路径：
/// 1. 正规方程：首次建立 JᵀJ 结构，之后只计算数值。
/// 2. 正规方程：每轮重新计算 JᵀJ 的结构和数值。
/// 3. 增广方程：使用原上三角填充函数。
/// 4. 增广方程：使用新的 Jacobian 算子填充上三角。
/// 5. COO基线：V4填J，再用COO组装完整增广矩阵。
/// 6. COO基线：全J裁剪，再用COO组装完整增广矩阵。
/// 7–8. 两条COO基线开启upper_only，只写上三角。
///
/// 均使用QDLDL、同一初值及步长控制，并在一次潮流内复用求解器。
/// 分别记录初始化、矩阵准备、线性求解和完整潮流时间。
/// 每种实现先预热一次，再测七次。固定CPU，单线程运行此测试。
#[cfg(feature = "probe")]
pub fn benchmark_cached_normal_equations() {
    run_assembly_comparison(&[0, 1, 2, 3, 4, 5, 6, 7]);
}

/// 同一KKT结构与求解器：V4+上三角COO → 原triu直填 → 算子直填。
/// 直接读取已有Jacobian、COO和求解计时器，不增加纯内核微基准。
#[cfg(feature = "probe")]
pub fn benchmark_coo_ablation() {
    run_assembly_comparison(&[6, 2, 3]);
}

#[cfg(feature = "probe")]
fn run_assembly_comparison(methods: &[usize]) {
    use crate::basic::solver::qdldl_probe;
    use rustpower::lm::normal_eq::NeDriver;
    use std::sync::atomic::Ordering;

    let out =
        std::env::var("RUSTPOWER_NE_AUDIT_DIR").expect("请用 RUSTPOWER_NE_AUDIT_DIR 指定结果目录");
    std::fs::create_dir_all(&out).unwrap();
    let cases = [
        load_case("IEEE39"),
        load_case("IEEE118"),
        load_case("pegase9241"),
        load_6515("dc"),
        load_6515("flat"),
    ];
    let mut records = Vec::new();
    for case in &cases {
        let mut options = rustpower::lm::LmOptions::default();
        if case.name.contains("flat") {
            options.trust_region = Some(rustpower::lm::step_control::TrustRegionOptions::default());
        }
        // 以该算例第一条路径的结果为对照，检查后续各次运行。
        let mut reference = None::<(usize, u64, Vec<Complex64>)>;
        println!(
            "\n算例：{}。各实现使用相同初值和残差容差；全部使用当前步长控制并复用求解器。",
            case.name
        );
        // 第0轮只预热，不计入中位数。每次均重新创建驱动器和线性求解器。
        for round in 0..8 {
            for slot in 0..methods.len() {
                let index = if round % 2 == 0 { slot } else { methods.len() - 1 - slot };
                let method = methods[index];
                qdldl_probe::reset();
                let (measurement, v, product_symbolic_ms, product_numeric_ms, mu_ms, coo_ms) =
                    if method < 2 {
                        let mut v = case.v.clone();
                        let start = Instant::now();
                        let mut driver =
                            NeDriver::build(&case.y, case.npv, case.npq, case.s.clone());
                        driver.dumb_mode = method == 1;
                        let mut solver = QDLDLSolver::with_dsigns(vec![1; case.npv + 2 * case.npq]);
                        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let start = Instant::now();
                        let result = driver.solve_ne_with_options(
                            &case.y,
                            &mut solver,
                            &mut v,
                            1e-8,
                            300,
                            &options,
                        );
                        let solve_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let residual_inf = independent_residual(case, &v);
                        assert_eq!(result.converged, residual_inf < 1e-8);
                        (
                            Measurement {
                                build_ms,
                                solve_ms,
                                fill_ms: driver.prof_fill_ns as f64 / 1e6,
                                mu_ms: driver.prof_mu_ns as f64 / 1e6,
                                linear_solve_ms: driver.prof_solve_ns as f64 / 1e6,
                                iterations: result.iterations,
                                linear_solves: driver.n_solves,
                                converged: result.converged,
                                residual_inf,
                            },
                            v,
                            driver.prof_spgemm_ns as f64 / 1e6,
                            driver.prof_numeric_ns as f64 / 1e6,
                            driver.prof_mu_ns as f64 / 1e6,
                            0.0,
                        )
                    } else if method < 4 {
                        let mode = if method == 2 {
                            Method::UpperBaseline
                        } else {
                            Method::UpperOperator
                        };
                        let (m, v) = run_with_options(case, mode, 300, &options);
                        let mu_ms = m.mu_ms;
                        (m, v, 0.0, 0.0, mu_ms, 0.0)
                    } else {
                        let (m, v, coo_ms) = run_coo_baseline(case, method % 2 == 1, method >= 6, &options);
                        let m_mu_ms = m.mu_ms;
                        (m, v, 0.0, 0.0, m_mu_ms, coo_ms)
                    };
                // 英文标识用于已有JSON文件；终端输出使用完整中文名称。
                let label = [
                    "NE-cached",
                    "NE-rebuild",
                    "AUG-upper",
                    "AUG-operator",
                    "AUG-COO",
                    "AUG-FS",
                    "AUG-COO-upper",
                    "AUG-FS-upper",
                ][method];
                let description = [
                    "正规方程：复用乘积结构",
                    "正规方程：每轮重建乘积",
                    "增广方程：原上三角填充",
                    "增广方程：新算子填充",
                    "COO基线：V4填J再用COO组装",
                    "COO基线：全J裁剪后用完整COO组装",
                    "COO基线：V4填J后只组装上三角",
                    "COO基线：全J裁剪后只组装上三角",
                ][method];
                assert!(measurement.converged, "{} {label} failed", case.name);
                let max_dv = if measurement.converged {
                    if let Some((iterations, solves, voltage)) = &reference {
                        assert_eq!(*iterations, measurement.iterations, "{} {label}: iterations", case.name);
                        assert_eq!(*solves, measurement.linear_solves, "{} {label}: solves", case.name);
                        let dv = voltage
                            .iter()
                            .zip(&v)
                            .map(|(a, b)| (a - b).norm())
                            .fold(0.0f64, f64::max);
                        assert!(dv < 1e-6, "{} {label}: voltage difference {dv}", case.name);
                        Some(dv)
                    } else {
                        reference = Some((measurement.iterations, measurement.linear_solves, v));
                        Some(0.0)
                    }
                } else {
                    None
                };
                let read_ms = |counter: &std::sync::atomic::AtomicU64| {
                    counter.load(Ordering::Relaxed) as f64 / 1e6
                };
                let row = serde_json::json!({
                    "case": case.name, "method": label, "round": round,
                    "buses": case.y.ncols(), "states": case.npv + 2 * case.npq,
                    "options": options, "build_ms": measurement.build_ms,
                    "converged": measurement.converged,
                    "policy": "current",
                    "solver_reuse": true,
                    "jacobian_evaluations": measurement.iterations,
                    "coo_assemblies": if method >= 4 { measurement.linear_solves } else { 0 },
                    "coo_upper_only": if method >= 4 { Some(method >= 6) } else { None },
                    "tolerance_inf":1e-8,"max_iterations":300,
                    "coo_ms":coo_ms,
                    "matrix_preparation_ms":measurement.fill_ms+product_symbolic_ms+product_numeric_ms+mu_ms+coo_ms,
                    "total_execution_ms": measurement.build_ms + measurement.solve_ms,
                    "solve_ms": measurement.solve_ms, "iterations": measurement.iterations,
                    "linear_solves": measurement.linear_solves, "residual_inf": measurement.residual_inf,
                    "max_voltage_difference": max_dv, "j_or_aug_fill_ms": measurement.fill_ms,
                    "product_symbolic_ms": product_symbolic_ms, "product_numeric_ms": product_numeric_ms,
                    "mu_ms": mu_ms, "linear_total_ms": measurement.linear_solve_ms,
                    "solver_setup_ms": read_ms(&qdldl_probe::SYM_NS),
                    "solver_numeric_ms": read_ms(&qdldl_probe::NUMERIC_NS),
                    "solver_backsolve_ms": read_ms(&qdldl_probe::SOLVE_NS),
                });
                println!(
                    "  {description}，第{round}轮（0为预热）：{}，接受{}步，线性求解{}次，组装及右端准备{:.3} ms，总执行{:.3} ms（初始化{:.3} ms），残差∞={:.2e}",
                    if measurement.converged { "收敛" } else { "未收敛" },
                    measurement.iterations,
                    measurement.linear_solves,
                    measurement.fill_ms + product_symbolic_ms + product_numeric_ms + mu_ms + coo_ms,
                    measurement.build_ms + measurement.solve_ms,
                    measurement.build_ms,
                    measurement.residual_inf
                );
                records.push(row);
                std::fs::write(
                    format!("{out}/measurements.json"),
                    serde_json::to_vec_pretty(&records).unwrap(),
                )
                .unwrap();
            }
        }
    }
}

#[cfg(all(feature = "probe", target_os = "linux"))]
#[path = "linear_solvers.rs"]
pub mod linear_solvers;
