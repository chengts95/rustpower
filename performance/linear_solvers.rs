//! 用通用稀疏乘法生成同一个正规方程，比较KLU、QDLDL和CHOLMOD LLᵀ。
//! 只用于测试；CHOLMOD的结构分析、分解存储和回代工作区均复用。
use super::{
    Case, Measurement, Method, benchmark_cases, benchmark_options, independent_residual,
    run_normal, run_with_options,
};
use crate::basic::solver::{KLUSolver, QDLDLSolver, Solve, klu_probe, qdldl_probe};
use crate::bench::{self, Report, timeit};
use crate::lm::{LmOptions, gn_flat::GnDriver};
use std::{
    ffi::{CString, c_char, c_int, c_void},
    sync::atomic::Ordering,
};

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(library: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(library: *mut c_void) -> c_int;
}
type Analyze =
    unsafe extern "C" fn(usize, *mut usize, *mut usize, *mut f64, c_int, c_int) -> *mut c_void;
type Factorize =
    unsafe extern "C" fn(*mut c_void, usize, *mut usize, *mut usize, *mut f64) -> c_int;
type Backsolve = unsafe extern "C" fn(*mut c_void, usize, *mut f64) -> c_int;
type Free = unsafe extern "C" fn(*mut c_void);

struct Cholesky {
    library: *mut c_void,
    handle: *mut c_void,
    analyze: Analyze,
    factorize: Factorize,
    backsolve: Backsolve,
    free: Free,
    supernodal: bool,
    threads: i32,
    pattern: Option<(Vec<usize>, Vec<usize>)>,
    analysis_count: usize,
    analysis_ms: f64,
    factor_ms: f64,
    backsolve_ms: f64,
}
impl Cholesky {
    fn new(supernodal: bool, threads: i32) -> Self {
        let path = CString::new(
            std::env::var("RUSTPOWER_CHOLMOD_LIBRARY")
                .expect("请指定测试接口动态库 RUSTPOWER_CHOLMOD_LIBRARY"),
        )
        .unwrap();
        unsafe {
            let library = dlopen(path.as_ptr(), 2);
            assert!(!library.is_null(), "无法加载CHOLMOD测试接口，请先编译C文件");
            let symbol = |name: &str| {
                let name = CString::new(name).unwrap();
                let ptr = dlsym(library, name.as_ptr());
                assert!(!ptr.is_null(), "CHOLMOD测试接口缺少函数");
                ptr
            };
            Self {
                analyze: std::mem::transmute::<*mut c_void, Analyze>(symbol("lm_cholmod_analyze")),
                factorize: std::mem::transmute::<*mut c_void, Factorize>(symbol(
                    "lm_cholmod_factorize",
                )),
                backsolve: std::mem::transmute::<*mut c_void, Backsolve>(symbol(
                    "lm_cholmod_solve",
                )),
                free: std::mem::transmute::<*mut c_void, Free>(symbol("lm_cholmod_free")),
                library,
                handle: std::ptr::null_mut(),
                supernodal,
                threads,
                pattern: None,
                analysis_count: 0,
                analysis_ms: 0.0,
                factor_ms: 0.0,
                backsolve_ms: 0.0,
            }
        }
    }
}
impl Solve for Cholesky {
    fn solve(
        &mut self,
        cp: &mut [usize],
        ri: &mut [usize],
        values: &mut [f64],
        rhs: &mut [f64],
        n: usize,
    ) -> Result<(), &'static str> {
        if let Some((old_cp, old_ri)) = &self.pattern {
            assert!(old_cp == cp && old_ri == ri, "求解期间矩阵结构改变");
        } else {
            let (_, elapsed_ms) = timeit!({
                self.pattern = Some((cp.to_vec(), ri.to_vec()));
                self.handle = unsafe {
                    (self.analyze)(
                        n,
                        cp.as_mut_ptr(),
                        ri.as_mut_ptr(),
                        values.as_mut_ptr(),
                        self.supernodal as i32,
                        self.threads,
                    )
                };
            });
            self.analysis_ms += elapsed_ms;
            if self.handle.is_null() {
                self.pattern = None;
                return Err("CHOLMOD分析失败");
            }
            self.analysis_count += 1;
        }
        let (ok, elapsed_ms) = timeit!(unsafe {
            (self.factorize)(
                self.handle,
                n,
                cp.as_mut_ptr(),
                ri.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        });
        self.factor_ms += elapsed_ms;
        if ok == 0 {
            return Err("CHOLMOD LLᵀ分解失败");
        }
        let (ok, elapsed_ms) =
            timeit!(unsafe { (self.backsolve)(self.handle, n, rhs.as_mut_ptr()) });
        self.backsolve_ms += elapsed_ms;
        if ok == 0 {
            return Err("CHOLMOD回代失败");
        }
        Ok(())
    }
    fn reset(&mut self) {
        unsafe {
            (self.free)(self.handle);
        }
        self.handle = std::ptr::null_mut();
        self.pattern = None;
    }
}
impl Drop for Cholesky {
    fn drop(&mut self) {
        self.reset();
        unsafe {
            dlclose(self.library);
        }
    }
}

/// 同一矩阵布局只换后端；每次完整潮流内部复用驱动器和求解器。
fn run_system<S: Solve>(
    case: &Case,
    options: &LmOptions,
    solver: &mut S,
    augmented: bool,
) -> (serde_json::Value, Vec<num_complex::Complex64>) {
    let (m, v, product_ms) = if augmented {
        let mut v = case.v.clone();
        let (mut d, build_ms) = timeit!(GnDriver::build_operator(
            &case.y,
            case.npv,
            case.npq,
            case.s.clone()
        ));
        let (r, solve_ms) =
            timeit!(d.solve_gn_with_options(&case.y, solver, &mut v, 1e-8, 300, options));
        let mut m = measurement!(d, r, build_ms, solve_ms, prof_fill_ns);
        m.residual_inf = independent_residual(case, &v);
        assert!(!m.converged || m.residual_inf < 1e-8);
        (m, v, 0.0)
    } else {
        let (m, v, product, _) = run_normal(case, solver, true, options);
        (m, v, product)
    };
    (backend_record(&m, product_ms), v)
}

fn backend_record(m: &Measurement, product_ms: f64) -> serde_json::Value {
    serde_json::json!({"converged":m.converged,"build_ms":m.build_ms,"solve_ms":m.solve_ms,
        "iterations":m.iterations,"linear_solves":m.linear_solves,"residual_inf":m.residual_inf,
        "product_ms":product_ms,"fill_ms":m.fill_ms,"mu_ms":m.mu_ms,"linear_ms":m.linear_solve_ms})
}

/// 将后端构造也计入初始化；一次潮流内仍使用同一个求解器。
fn run_with_backend<S: Solve>(
    case: &Case,
    options: &LmOptions,
    make_solver: impl FnOnce() -> S,
    augmented: bool,
) -> (serde_json::Value, Vec<num_complex::Complex64>) {
    let (mut solver, backend_build_ms) = timeit!(make_solver());
    let (mut row, v) = run_system(case, options, &mut solver, augmented);
    row["build_ms"] = (row["build_ms"].as_f64().unwrap() + backend_build_ms).into();
    (row, v)
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
        ("build_ms", "初始化ms"),
        ("fill_ms", "J/Jᵀ ms"),
        ("product_ms", "JᵀJ ms"),
        ("mu_ms", "μ/右端ms"),
        ("matrix_preparation_ms", "组装准备ms"),
    ],
    &[
        ("analysis_ms", "符号ms"),
        ("factor_ms", "分解含refactor ms"),
        ("backsolve_ms", "回代ms"),
        ("linear_ms", "线性求解ms"),
    ],
    &[
        ("solve_ms", "LM求解ms"),
        ("total_execution_ms", "总执行ms"),
        ("first_factor_count", "首次factor次数"),
        ("refactor_count", "refactor次数"),
        ("factor_fallback_count", "fallback次数"),
    ],
];

#[derive(Clone, Copy)]
enum BackendMethod {
    NormalQdldl,
    NormalCholesky,
    NormalSupernodal,
    UpperQdldl,
    NormalKlu,
    FullQdldl,
    FullKlu,
}
impl BackendMethod {
    fn label(self) -> &'static str {
        use BackendMethod::*;
        match self {
            NormalQdldl => "正规方程-QDLDL",
            NormalCholesky => "正规方程-普通Cholesky",
            NormalSupernodal => "正规方程-超节点Cholesky",
            UpperQdldl => "增广方程-上三角-QDLDL",
            NormalKlu => "正规方程-KLU",
            FullQdldl => "增广方程-完整-QDLDL",
            FullKlu => "增广方程-完整-KLU",
        }
    }
    fn backend(self) -> &'static str {
        use BackendMethod::*;
        match self {
            NormalKlu | FullKlu => "KLU",
            NormalCholesky | NormalSupernodal => "CHOLMOD",
            _ => "QDLDL",
        }
    }
}

pub fn benchmark_linear_solvers() {
    use BackendMethod::*;
    let threads: i32 = std::env::var("RUSTPOWER_CHOLMOD_THREADS")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    assert!(threads > 0);
    let out = std::env::var("RUSTPOWER_CHOLESKY_OUTPUT").expect("请指定输出目录");
    // --klu复用已有五条KLU/QDLDL路径，不要求安装CHOLMOD。
    let methods: &[BackendMethod] = if std::env::args().any(|arg| arg == "--klu") {
        &[NormalQdldl, UpperQdldl, NormalKlu, FullQdldl, FullKlu]
    } else {
        &[
            NormalQdldl,
            NormalCholesky,
            NormalSupernodal,
            UpperQdldl,
            NormalKlu,
            FullQdldl,
            FullKlu,
        ]
    };
    let mut report = Report::new(out, COLUMNS);
    let repeats = bench::repeats();
    for case in &benchmark_cases() {
        let options = benchmark_options(case);
        let mut reference = None::<(u64, u64, Vec<num_complex::Complex64>)>;
        println!(
            "\n{}：QDLDL/KLU后端比较，容差∞=1e-8，最大迭代300，预热1次、测量{repeats}次。\n参数：{}",
            case.name,
            serde_json::to_string_pretty(&options).unwrap()
        );
        for (round, method) in bench::runs(methods, repeats) {
            let label = method.label();
            qdldl_probe::reset();
            klu_probe::reset();
            let (mut row, v) = match method {
                NormalQdldl => run_with_backend(
                    case,
                    &options,
                    || QDLDLSolver::with_dsigns(vec![1; case.npv + 2 * case.npq]),
                    false,
                ),
                NormalCholesky | NormalSupernodal => {
                    let (mut solver, backend_build_ms) =
                        timeit!(Cholesky::new(matches!(method, NormalSupernodal), threads));
                    let (mut r, v) = run_system(case, &options, &mut solver, false);
                    r["build_ms"] = (r["build_ms"].as_f64().unwrap() + backend_build_ms).into();
                    assert_eq!(solver.analysis_count, 1, "CHOLMOD不应重复分析结构");
                    r["analysis_count"] = solver.analysis_count.into();
                    r["analysis_ms"] = solver.analysis_ms.into();
                    r["factor_ms"] = solver.factor_ms.into();
                    r["backsolve_ms"] = solver.backsolve_ms.into();
                    (r, v)
                }
                NormalKlu => run_with_backend(case, &options, KLUSolver::default, false),
                FullQdldl => {
                    let n = case.npv + 2 * case.npq;
                    run_with_backend(
                        case,
                        &options,
                        || QDLDLSolver::with_dsigns([vec![1; n], vec![-1; n]].concat()),
                        true,
                    )
                }
                FullKlu => run_with_backend(case, &options, KLUSolver::default, true),
                UpperQdldl => {
                    let (m, v) = run_with_options(case, Method::UpperOperator, 300, &options);
                    assert!(m.converged);
                    (backend_record(&m, 0.0), v)
                }
            };
            if method.backend() == "QDLDL" {
                let ms = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
                row["analysis_ms"] = ms(&qdldl_probe::SYM_NS).into();
                row["factor_ms"] = ms(&qdldl_probe::NUMERIC_NS).into();
                row["backsolve_ms"] = ms(&qdldl_probe::SOLVE_NS).into();
            }
            if method.backend() == "KLU" {
                let ms = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
                row["analysis_ms"] = ms(&klu_probe::SYM_NS).into();
                row["factor_ms"] = (ms(&klu_probe::FACTOR_NS) + ms(&klu_probe::REFACTOR_NS)).into();
                row["backsolve_ms"] = ms(&klu_probe::SOLVE_NS).into();
                row["first_factor_count"] =
                    klu_probe::N_FIRST_FACTOR.load(Ordering::Relaxed).into();
                row["refactor_count"] = klu_probe::N_REFACTOR.load(Ordering::Relaxed).into();
                row["factor_fallback_count"] =
                    klu_probe::N_FACTOR_FALLBACK.load(Ordering::Relaxed).into();
                assert_eq!(
                    klu_probe::N_FIRST_FACTOR.load(Ordering::Relaxed),
                    1,
                    "KLU应复用求解器"
                );
            }
            let iterations = row["iterations"].as_u64().unwrap();
            let solves = row["linear_solves"].as_u64().unwrap();
            let converged = row["converged"].as_bool().unwrap();
            // 不同分解的舍入及失败重试可能改变轨迹；记录计数差异，不强行视为同一步。
            // 只有收敛且电压一致的运行才可用于完整潮流耗时比较。
            if converged {
                if let Some((old_it, old_solves, old_v)) = &reference {
                    row["same_iterations_as_reference"] = (*old_it == iterations).into();
                    row["same_linear_solves_as_reference"] = (*old_solves == solves).into();
                    let dv = old_v
                        .iter()
                        .zip(&v)
                        .map(|(a, b)| (a - b).norm())
                        .fold(0.0f64, f64::max);
                    assert!(dv < 1e-6, "{} {label}电压差{dv}", case.name);
                    row["max_voltage_difference"] = dv.into();
                } else {
                    row["max_voltage_difference"] = 0.0.into();
                    row["same_iterations_as_reference"] = true.into();
                    row["same_linear_solves_as_reference"] = true.into();
                    reference = Some((iterations, solves, v));
                }
            }
            row["case"] = case.name.clone().into();
            row["method"] = label.into();
            row["round"] = round.into();
            row["cholmod_threads"] = threads.into();
            row["options"] = serde_json::to_value(&options).unwrap();
            row["backend"] = method.backend().into();
            row["matrix_storage"] = if matches!(method, UpperQdldl) {
                "upper"
            } else {
                "full"
            }
            .into();
            row["jacobian_method"] = if matches!(
                method,
                NormalQdldl | NormalCholesky | NormalSupernodal | NormalKlu
            ) {
                "V4"
            } else {
                "operator"
            }
            .into();
            row["solver_reuse"] = true.into();
            row["buses"] = case.y.ncols().into();
            row["states"] = (case.npv + 2 * case.npq).into();
            row["tolerance_inf"] = 1e-8.into();
            row["max_iterations"] = 300.into();
            row["total_execution_ms"] =
                (row["build_ms"].as_f64().unwrap() + row["solve_ms"].as_f64().unwrap()).into();
            row["matrix_preparation_ms"] = (row["fill_ms"].as_f64().unwrap()
                + row["product_ms"].as_f64().unwrap()
                + row["mu_ms"].as_f64().unwrap())
            .into();
            report.push(row, label);
        }
        report.summary(&case.name);
    }
}
