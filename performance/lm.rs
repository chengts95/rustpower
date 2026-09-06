//! LM 对比入口：实现一致性、正规方程与增广方程、线性求解后端。
//! 各路径共用步长控制，在一次潮流内部复用求解器。
use crate::basic::solver::{KLUSolver, QDLDLSolver};
use crate::bench::{self, Report, timeit};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::lm::{gn_flat::GnDriver, gn_triu::GnTriuDriver};
use serde::Serialize;

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

// 只转换已有计数器；不在宏中实现任何数值算法。
macro_rules! measurement {
    ($d:ident, $r:ident, $build:expr, $solve:expr, $fill:ident) => {
        Measurement {
            build_ms: $build,
            solve_ms: $solve,
            fill_ms: $d.$fill as f64 / 1e6,
            mu_ms: $d.prof_mu_ns as f64 / 1e6,
            linear_solve_ms: $d.prof_solve_ns as f64 / 1e6,
            iterations: $r.iterations,
            linear_solves: $d.n_solves,
            converged: $r.converged,
            residual_inf: $r.res_inf,
        }
    };
}

pub(super) fn run_with_options(
    case: &Case,
    method: Method,
    max_iter: usize,
    options: &rustpower::lm::LmOptions,
) -> (Measurement, Vec<Complex64>) {
    let mut v = case.v.clone();
    let operator = matches!(method, Method::FullOperator | Method::UpperOperator);
    // 两种驱动器有相同的接口和计时器，构造类型由下面的match明确选择。
    macro_rules! run {
        ($driver:ty, $solver:expr) => {{
            let ((mut d, mut solver), build_ms) = timeit!({
                let d = if operator {
                    <$driver>::build_operator(&case.y, case.npv, case.npq, case.s.clone())
                } else {
                    <$driver>::build(&case.y, case.npv, case.npq, case.s.clone())
                };
                (d, $solver)
            });
            let (r, solve_ms) = timeit!(d.solve_gn_with_options(
                &case.y,
                &mut solver,
                &mut v,
                1e-8,
                max_iter,
                options
            ));
            measurement!(d, r, build_ms, solve_ms, prof_fill_ns)
        }};
    }
    let mut m = match method {
        Method::FullBaseline | Method::FullOperator => run!(GnDriver, KLUSolver::default()),
        Method::UpperBaseline | Method::UpperOperator => run!(GnTriuDriver, QDLDLSolver::default()),
    };
    m.residual_inf = independent_residual(case, &v);
    assert!(
        !m.converged || m.residual_inf < 1e-8,
        "{} {method:?}: {}",
        case.name,
        m.residual_inf
    );
    (m, v)
}

/// 正规方程的两种乘积模式及所有线性后端共用此调用。
pub(super) fn run_normal<S: crate::basic::solver::Solve>(
    case: &Case,
    solver: &mut S,
    rebuild: bool,
    options: &rustpower::lm::LmOptions,
) -> (Measurement, Vec<Complex64>, f64, f64) {
    use rustpower::lm::normal_eq::NeDriver;
    let mut v = case.v.clone();
    let (mut d, build_ms) = timeit!({
        let mut d = NeDriver::build(&case.y, case.npv, case.npq, case.s.clone());
        d.dumb_mode = rebuild;
        d
    });
    let (r, solve_ms) =
        timeit!(d.solve_ne_with_options(&case.y, solver, &mut v, 1e-8, 300, options));
    let mut m = measurement!(d, r, build_ms, solve_ms, prof_fill_ns);
    m.residual_inf = independent_residual(case, &v);
    assert!(!m.converged || m.residual_inf < 1e-8);
    (
        m,
        v,
        d.prof_spgemm_ns as f64 / 1e6,
        d.prof_numeric_ns as f64 / 1e6,
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
            let (reference, voltage) =
                run_with_options(&case, Method::UpperOperator, 300, &options);
            for full_slice in [false, true] {
                for upper_only in [false, true] {
                    let (coo, coo_voltage, _) =
                        run_coo_baseline(&case, full_slice, upper_only, &options);
                    assert!(coo.converged, "{name}: COO did not converge");
                    assert_eq!(coo.iterations, reference.iterations);
                    assert_eq!(coo.linear_solves, reference.linear_solves);
                    let error = voltage
                        .iter()
                        .zip(&coo_voltage)
                        .map(|(a, b)| (*a - *b).norm())
                        .fold(0.0_f64, f64::max);
                    assert!(error < 1e-8, "{name}: COO voltage difference {error}");
                }
            }
        }
    }
}

/// COO仍逐次组装和转换；与其他路径共用参数，并复用求解器。
fn run_coo_baseline(
    case: &Case,
    full_slice: bool,
    upper_only: bool,
    options: &rustpower::lm::LmOptions,
) -> (Measurement, Vec<Complex64>, f64) {
    use rustpower::lm::baseline::{aug_coo::AugCooDriver, full_slice::AugFsDriver};
    let mut v = case.v.clone();
    macro_rules! run {
        ($driver:ty, $solve:ident, $fill:ident, $coo:ident) => {{
            let (mut d, build_ms) = timeit!({
                let mut d = <$driver>::build(&case.y, case.npv, case.npq, case.s.clone());
                d.upper_only = upper_only;
                d
            });
            let (r, solve_ms) = timeit!(d.$solve(&case.y, &mut v, 1e-8, 300, options));
            (
                measurement!(d, r, build_ms, solve_ms, $fill),
                d.$coo as f64 / 1e6,
            )
        }};
    }
    let (mut m, coo_ms) = if full_slice {
        run!(
            AugFsDriver,
            solve_aug_fs_with_options,
            prof_full_j_ns,
            prof_slice_coo_ns
        )
    } else {
        run!(
            AugCooDriver,
            solve_aug_coo_with_options,
            prof_fill_ns,
            prof_coo_ns
        )
    };
    m.residual_inf = independent_residual(case, &v);
    assert!(!m.converged || m.residual_inf < 1e-8);
    (m, v, coo_ms)
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
    use AssemblyMethod::*;
    run_assembly_comparison(&[
        NormalCached,
        NormalGeneric,
        Rows,
        Operator,
        Coo {
            full_j: false,
            upper: false,
        },
        Coo {
            full_j: true,
            upper: false,
        },
        Coo {
            full_j: false,
            upper: true,
        },
        Coo {
            full_j: true,
            upper: true,
        },
    ]);
}

/// 同一增广上三角与求解器：全J裁剪COO → V4+COO → 原triu → 算子triu。
/// 直接读取已有Jacobian、COO和求解计时器，不增加纯内核微基准。
#[cfg(feature = "probe")]
pub fn benchmark_coo_ablation() {
    use AssemblyMethod::*;
    run_assembly_comparison(&[
        Coo {
            full_j: true,
            upper: true,
        },
        Coo {
            full_j: false,
            upper: true,
        },
        Rows,
        Operator,
    ]);
}

#[derive(Clone, Copy)]
enum AssemblyMethod {
    NormalCached,
    NormalGeneric,
    Rows,
    Operator,
    Coo { full_j: bool, upper: bool },
}
impl AssemblyMethod {
    fn labels(self) -> (&'static str, &'static str) {
        use AssemblyMethod::*;
        match self {
            NormalCached => ("NE-cached", "V4 + 固定结构JᵀJ"),
            NormalGeneric => ("NE-rebuild", "V4 + 通用JᵀJ"),
            Rows => ("AUG-upper", "原triu直接填充"),
            Operator => ("AUG-operator", "算子triu直接填充"),
            Coo {
                full_j: false,
                upper: false,
            } => ("AUG-COO", "V4 + 完整COO"),
            Coo {
                full_j: true,
                upper: false,
            } => ("AUG-FS", "全J裁剪 + 完整COO（不用V4）"),
            Coo {
                full_j: false,
                upper: true,
            } => ("AUG-COO-upper", "V4 + 上三角COO"),
            Coo {
                full_j: true,
                upper: true,
            } => ("AUG-FS-upper", "全J裁剪 + 上三角COO（不用V4）"),
        }
    }
}

pub(super) fn benchmark_cases() -> Vec<Case> {
    let cases: Vec<_> = [
        "IEEE39",
        "IEEE118",
        "pegase9241",
        "6515rte_dc",
        "6515rte_flat",
    ]
    .into_iter()
    .filter(|name| bench::selected(name))
    .map(|name| match name {
        "6515rte_dc" => load_6515("dc"),
        "6515rte_flat" => load_6515("flat"),
        _ => load_case(name),
    })
    .collect();
    assert!(!cases.is_empty(), "--case没有匹配的LM算例");
    cases
}
pub(super) fn benchmark_options(case: &Case) -> rustpower::lm::LmOptions {
    let mut options = rustpower::lm::LmOptions::default();
    if case.name.ends_with("flat") {
        options.trust_region = Some(Default::default());
    }
    options
}
const COLUMNS: &[bench::Columns] = &[
    &[
        ("converged", "收敛"),
        ("iterations", "接受步"),
        ("linear_solves", "线性求解次数"),
        ("residual_inf", "残差∞"),
        ("max_voltage_difference", "最大电压差"),
    ],
    &[
        ("j_or_aug_fill_ms", "J/Jᵀ ms"),
        ("product_symbolic_ms", "乘积构建ms"),
        ("product_numeric_ms", "固定乘积数值ms"),
        ("coo_ms", "COO ms"),
        ("mu_ms", "μ/右端ms"),
    ],
    &[
        ("matrix_preparation_ms", "组装准备ms"),
        ("build_ms", "初始化ms"),
        ("solve_ms", "LM求解ms"),
        ("total_execution_ms", "总执行ms"),
    ],
    &[
        ("linear_total_ms", "线性求解ms"),
        ("solver_setup_ms", "符号ms"),
        ("solver_numeric_ms", "分解ms"),
        ("solver_backsolve_ms", "回代ms"),
        ("jacobian_evaluations", "J评估次数"),
        ("coo_assemblies", "COO组装次数"),
    ],
];

#[cfg(feature = "probe")]
fn run_assembly_comparison(methods: &[AssemblyMethod]) {
    use crate::basic::solver::qdldl_probe;
    use AssemblyMethod::*;
    use std::sync::atomic::Ordering;

    let out =
        std::env::var("RUSTPOWER_NE_AUDIT_DIR").expect("请用 RUSTPOWER_NE_AUDIT_DIR 指定结果目录");
    let repeats = bench::repeats();
    let mut report = Report::new(out, COLUMNS);
    for case in &benchmark_cases() {
        let options = benchmark_options(case);
        // 以该算例第一条路径的结果为对照，检查后续各次运行。
        let mut reference = None::<(usize, u64, Vec<Complex64>)>;
        println!(
            "\n算例：{}。各实现使用相同初值和残差容差；全部使用当前步长控制并复用求解器。",
            case.name
        );
        println!(
            "节点={}，状态变量={}，容差∞=1e-8，最大迭代=300，QDLDL，预热1次、测量{repeats}次。",
            case.y.ncols(),
            case.npv + 2 * case.npq
        );
        println!(
            "LM参数：{}",
            serde_json::to_string_pretty(&options).unwrap()
        );
        for (round, method) in bench::runs(methods, repeats) {
            qdldl_probe::reset();
            let (measurement, v, product_symbolic_ms, product_numeric_ms, mu_ms, coo_ms) =
                match method {
                    NormalCached | NormalGeneric => {
                        let (mut solver, solver_build_ms) =
                            timeit!(QDLDLSolver::with_dsigns(vec![1; case.npv + 2 * case.npq]));
                        let (mut m, v, symbolic, numeric) = run_normal(
                            case,
                            &mut solver,
                            matches!(method, NormalGeneric),
                            &options,
                        );
                        m.build_ms += solver_build_ms;
                        let mu_ms = m.mu_ms;
                        (m, v, symbolic, numeric, mu_ms, 0.0)
                    }
                    Rows | Operator => {
                        let mode = if matches!(method, Rows) {
                            Method::UpperBaseline
                        } else {
                            Method::UpperOperator
                        };
                        let (m, v) = run_with_options(case, mode, 300, &options);
                        let mu_ms = m.mu_ms;
                        (m, v, 0.0, 0.0, mu_ms, 0.0)
                    }
                    Coo { full_j, upper } => {
                        let (m, v, coo_ms) = run_coo_baseline(case, full_j, upper, &options);
                        let mu_ms = m.mu_ms;
                        (m, v, 0.0, 0.0, mu_ms, coo_ms)
                    }
                };
            let (label, description) = method.labels();
            assert!(measurement.converged, "{} {label} failed", case.name);
            let max_dv = if measurement.converged {
                if let Some((iterations, solves, voltage)) = &reference {
                    assert_eq!(
                        *iterations, measurement.iterations,
                        "{} {label}: iterations",
                        case.name
                    );
                    assert_eq!(
                        *solves, measurement.linear_solves,
                        "{} {label}: solves",
                        case.name
                    );
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
                "coo_assemblies": if matches!(method, Coo { .. }) { measurement.linear_solves } else { 0 },
                "coo_upper_only": match method { Coo { upper, .. } => Some(upper), _ => None },
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
            report.push(row, description);
        }
        report.summary(&case.name);
    }
}

#[cfg(all(feature = "probe", target_os = "linux"))]
#[path = "linear_solvers.rs"]
pub mod linear_solvers;
