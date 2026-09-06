//! 用通用稀疏乘法生成同一个正规方程，比较KLU、QDLDL和CHOLMOD LLᵀ。
//! 只用于测试；CHOLMOD的结构分析、分解存储和回代工作区均复用。
use super::{Case, Method, independent_residual, load_6515, load_case, run_with_options};
use crate::basic::solver::{KLUSolver, QDLDLSolver, Solve, klu_probe, qdldl_probe};
use crate::lm::{
    LmOptions, gn_flat::GnDriver, normal_eq::NeDriver, step_control::TrustRegionOptions,
};
use std::{
    ffi::{CString, c_char, c_int, c_void},
    sync::atomic::Ordering,
    time::Instant,
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
            let start = Instant::now();
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
            self.analysis_ms += start.elapsed().as_secs_f64() * 1000.0;
            if self.handle.is_null() {
                self.pattern = None;
                return Err("CHOLMOD分析失败");
            }
            self.analysis_count += 1;
        }
        let start = Instant::now();
        let ok = unsafe {
            (self.factorize)(
                self.handle,
                n,
                cp.as_mut_ptr(),
                ri.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        self.factor_ms += start.elapsed().as_secs_f64() * 1000.0;
        if ok == 0 {
            return Err("CHOLMOD LLᵀ分解失败");
        }
        let start = Instant::now();
        let ok = unsafe { (self.backsolve)(self.handle, n, rhs.as_mut_ptr()) };
        self.backsolve_ms += start.elapsed().as_secs_f64() * 1000.0;
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
    let mut v = case.v.clone();
    let start = Instant::now();
    let (converged, iterations, build_ms, solve_ms, solves, product_ns, fill_ns, linear_ns) =
        if augmented {
            let mut driver = GnDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone());
            let build_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let result = driver.solve_gn_with_options(&case.y, solver, &mut v, 1e-8, 300, options);
            (
                result.converged,
                result.iterations,
                build_ms,
                start.elapsed().as_secs_f64() * 1000.0,
                driver.n_solves,
                0,
                driver.prof_fill_ns,
                driver.prof_solve_ns,
            )
        } else {
            let mut driver = NeDriver::build(&case.y, case.npv, case.npq, case.s.clone());
            driver.dumb_mode = true;
            let build_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let result = driver.solve_ne_with_options(&case.y, solver, &mut v, 1e-8, 300, options);
            (
                result.converged,
                result.iterations,
                build_ms,
                start.elapsed().as_secs_f64() * 1000.0,
                driver.n_solves,
                driver.prof_spgemm_ns,
                driver.prof_fill_ns,
                driver.prof_solve_ns,
            )
        };
    let residual = independent_residual(case, &v);
    if converged {
        assert!(residual < 1e-8, "{}收敛标记与残差不符", case.name);
    }
    (
        serde_json::json!({"converged":converged, "build_ms":build_ms, "solve_ms":solve_ms,
        "iterations":iterations, "linear_solves":solves,
        "residual_inf":residual, "product_ms":product_ns as f64/1e6,
        "fill_ms":fill_ns as f64/1e6, "linear_ms":linear_ns as f64/1e6}),
        v,
    )
}

pub fn benchmark_linear_solvers() {
    let threads: i32 = std::env::var("RUSTPOWER_CHOLMOD_THREADS")
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    assert!(threads > 0);
    let out = std::env::var("RUSTPOWER_CHOLESKY_OUTPUT").expect("请指定输出目录");
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
        let mut options = LmOptions::default();
        if case.name.contains("flat") {
            options.trust_region = Some(TrustRegionOptions::default());
        }
        let mut reference = None::<(u64, u64, Vec<num_complex::Complex64>)>;
        for round in 0..8 {
            for slot in 0..7 {
                let method = if round % 2 == 0 { slot } else { 6 - slot };
                let label = [
                    "正规方程-QDLDL",
                    "正规方程-普通Cholesky",
                    "正规方程-超节点Cholesky",
                    "增广方程-上三角-QDLDL",
                    "正规方程-KLU",
                    "增广方程-完整-QDLDL",
                    "增广方程-完整-KLU",
                ][method];
                qdldl_probe::reset();
                klu_probe::reset();
                let (mut row, v) = match method {
                    0 => run_system(
                        case,
                        &options,
                        &mut QDLDLSolver::with_dsigns(vec![1; case.npv + 2 * case.npq]),
                        false,
                    ),
                    1 | 2 => {
                        let mut solver = Cholesky::new(method == 2, threads);
                        let (mut r, v) = run_system(case, &options, &mut solver, false);
                        assert_eq!(solver.analysis_count, 1, "CHOLMOD不应重复分析结构");
                        r["analysis_count"] = solver.analysis_count.into();
                        r["analysis_ms"] = solver.analysis_ms.into();
                        r["factor_ms"] = solver.factor_ms.into();
                        r["backsolve_ms"] = solver.backsolve_ms.into();
                        (r, v)
                    }
                    4 => run_system(case, &options, &mut KLUSolver::default(), false),
                    5 => {
                        let n = case.npv + 2 * case.npq;
                        let signs = [vec![1; n], vec![-1; n]].concat();
                        run_system(case, &options, &mut QDLDLSolver::with_dsigns(signs), true)
                    }
                    6 => run_system(case, &options, &mut KLUSolver::default(), true),
                    3 => {
                        let (m, v) = run_with_options(case, Method::UpperOperator, 300, &options);
                        assert!(m.converged);
                        (
                            serde_json::json!({"converged":m.converged,"build_ms":m.build_ms,"solve_ms":m.solve_ms,
                            "iterations":m.iterations,"linear_solves":m.linear_solves,
                            "residual_inf":m.residual_inf,"product_ms":0.0,
                            "fill_ms":m.fill_ms,"linear_ms":m.linear_solve_ms}),
                            v,
                        )
                    }
                    _ => unreachable!(),
                };
                if method == 0 || method == 3 || method == 5 {
                    let ms =
                        |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
                    row["analysis_ms"] = ms(&qdldl_probe::SYM_NS).into();
                    row["factor_ms"] = ms(&qdldl_probe::NUMERIC_NS).into();
                    row["backsolve_ms"] = ms(&qdldl_probe::SOLVE_NS).into();
                }
                if method == 4 || method == 6 {
                    let ms =
                        |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
                    row["analysis_ms"] = ms(&klu_probe::SYM_NS).into();
                    row["factor_ms"] =
                        (ms(&klu_probe::FACTOR_NS) + ms(&klu_probe::REFACTOR_NS)).into();
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
                        reference = Some((iterations, solves, v));
                    }
                }
                row["case"] = case.name.clone().into();
                row["method"] = label.into();
                row["round"] = round.into();
                row["cholmod_threads"] = threads.into();
                row["options"] = serde_json::to_value(&options).unwrap();
                println!(
                    "{}，{label}，第{round}轮（0为预热）：收敛={converged}，接受{iterations}步，线性求解{solves}次，总时间{:.3} ms，分解{:.3} ms",
                    case.name,
                    row["solve_ms"].as_f64().unwrap(),
                    row["factor_ms"].as_f64().unwrap()
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
