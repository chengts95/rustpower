//! LM 专项实验的共享骨架：所有实验都是 load_case → 缩放注入 → 求解 → 独立统计。
//! 各实验文件（robustness / shed）只写自己的协议与输出，求解与统计一律走这里。

use super::comparison::Case;
use crate::bench;
use nalgebra::DVector;
use num_complex::Complex64;
use rustpower::basic::newtonpf::newton_pf;
use rustpower::basic::solver::{KLUSolver, QDLDLSolver, Solve};
use rustpower::lm::LmOptions;
use rustpower::lm::gn_triu::GnTriuDriver;
use std::time::Instant;

pub(super) const TOL: f64 = 1e-8;
pub(super) const MAXIT: usize = 300;
/// PQ 最小幅值低于此值视为可疑低压解支（原始数值同时记录，可后处理调整）。
const LOW_VMAG: f64 = 0.8;

/// 一格实验结果：一种启动 × 一种方法在某个注入下的一次求解。
pub(super) struct Cell {
    pub(super) start: &'static str,
    pub(super) method: &'static str,
    pub(super) converged: bool,
    pub(super) iterations: usize,
    pub(super) linear_solves: u64,
    pub(super) res_inf: f64,
    pub(super) merit: f64,
    pub(super) min_vmag_pq: f64,
    pub(super) wall_ms: f64,
}

impl Cell {
    pub(super) fn outcome(&self) -> &'static str {
        if !self.converged {
            "failed"
        } else if self.min_vmag_pq < LOW_VMAG {
            "low_voltage"
        } else {
            "solution"
        }
    }

    pub(super) fn mark(&self) -> String {
        match self.outcome() {
            "solution" => format!("✓{}", self.iterations),
            "low_voltage" => format!("⚠{}({:.2})", self.iterations, self.min_vmag_pq),
            _ => "✗".into(),
        }
    }
}

/// 独立的 (∞范数, ½‖r‖²) 统计，不依赖任何驱动的内部残差。
pub(super) fn residual_stats(case: &Case, s: &[Complex64], v: &[Complex64]) -> (f64, f64) {
    let current = &case.y * &DVector::from_column_slice(v);
    let (mut inf, mut ssq) = (0.0_f64, 0.0_f64);
    for i in 0..case.npv + case.npq {
        let r = v[i] * current[i].conj() - s[i];
        if !r.re.is_finite() || !r.im.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        inf = inf.max(r.re.abs());
        ssq += r.re * r.re;
        if i < case.npq {
            inf = inf.max(r.im.abs());
            ssq += r.im * r.im;
        }
    }
    (inf, 0.5 * ssq)
}

pub(super) fn finish(case: &Case, s: &[Complex64], v: &[Complex64], converged: bool, iterations: usize, linear_solves: u64, wall: Instant) -> Cell {
    let (res_inf, merit) = residual_stats(case, s, v);
    let min_vmag_pq = v[..case.npq]
        .iter()
        .map(|x| x.norm())
        .fold(f64::INFINITY, f64::min);
    // 声称收敛但独立残差不达标：按失败处理（生产残差与独立残差同源，仅防御）。
    let converged = converged && res_inf < TOL;
    Cell {
        start: "",
        method: "",
        converged,
        iterations,
        linear_solves,
        res_inf,
        merit,
        min_vmag_pq,
        wall_ms: wall.elapsed().as_secs_f64() * 1e3,
    }
}

/// 生产 NR（KLU），带回收敛/终止电压。
pub(super) fn run_nr(case: &Case, s: &DVector<Complex64>, v0: &[Complex64]) -> (Cell, Vec<Complex64>) {
    let t = Instant::now();
    let mut solver = KLUSolver::default();
    let v0 = DVector::from_column_slice(v0);
    let (cell, v) = match newton_pf(&case.y, s, &v0, case.npv, case.npq, Some(TOL), Some(MAXIT), &mut solver, None) {
        Ok((v, it)) => (finish(case, s.as_slice(), v.as_slice(), true, it, 0, t), v),
        Err((_, v, it)) => (finish(case, s.as_slice(), v.as_slice(), false, it, 0, t), v),
    };
    (cell, v.as_slice().to_vec())
}

/// 生产默认 GN 路径（triu-operator + QDLDL）。驱动可跨多次求解复用
/// （符号结构与注入无关；sbus 逐次覆盖）。
pub(super) fn run_gn(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64], opts: &LmOptions) -> Cell {
    driver.sbus.clear();
    driver.sbus.extend_from_slice(s);
    let mut v = v0.to_vec();
    let t = Instant::now();
    let solves0 = driver.n_solves;
    let r = driver.solve_gn_with_options(
        &case.y,
        solver,
        &mut v,
        TOL,
        MAXIT,
        opts,
    );
    // n_solves 是驱动的跨调用累计计数；每格记差值（TR 拒绝的试步也是分解，
    // 比较 TR vs TS 的分解成本靠这个数）。
    finish(case, s, &v, r.converged, r.iterations, driver.n_solves - solves0, t)
}

/// 与 run_gn 相同，但带回终止电压（LS 证书实验需要残差分布）。
pub(super) fn run_gn_v(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64], opts: &LmOptions) -> (Cell, Vec<Complex64>) {
    driver.sbus.clear();
    driver.sbus.extend_from_slice(s);
    let mut v = v0.to_vec();
    let t = Instant::now();
    let solves0 = driver.n_solves;
    let r = driver.solve_gn_with_options(
        &case.y,
        solver,
        &mut v,
        TOL,
        MAXIT,
        opts,
    );
    (finish(case, s, &v, r.converged, r.iterations, driver.n_solves - solves0, t), v)
}

/// 自缩放阻尼 LM（γ=‖F‖² 全步走，无步控；见 rustpower::lm::self_damped）。
pub(super) fn run_sd(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64]) -> Cell {
    driver.sbus.clear();
    driver.sbus.extend_from_slice(s);
    let mut v = v0.to_vec();
    let t = Instant::now();
    // `--sd-maxit N`：SD 是无步控的一阶化方法，病态系统上可能需要远超 300 次迭代，
    // 加长上限用于区分"慢速收敛"与"真失败"，不影响 NR/GN 两格。
    let maxit = bench::option("--sd-maxit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAXIT);
    let solves0 = driver.n_solves;
    let r = driver.solve_self_damped(&case.y, solver, &mut v, TOL, maxit);
    finish(case, s, &v, r.converged, r.iterations, driver.n_solves - solves0, t)
}

/// TSLM：三步复合自缩放阻尼（一次分解三枪；见 rustpower::lm::self_damped）。
pub(super) fn run_ts(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64]) -> Cell {
    driver.sbus.clear();
    driver.sbus.extend_from_slice(s);
    let mut v = v0.to_vec();
    let t = Instant::now();
    let maxit = bench::option("--sd-maxit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAXIT);
    let solves0 = driver.n_solves;
    let r = driver.solve_self_damped_ts(&case.y, solver, &mut v, TOL, maxit);
    finish(case, s, &v, r.converged, r.iterations, driver.n_solves - solves0, t)
}

/// Chord（冻结 J 分解）NR：v0 处组装一次 J、KLU 分解一次，后续全部
/// `resolve` 纯回代——"dc 初值 + 三步复合"的 γ=0 极限形态。
/// `iterations` 记回代次数；分解次数恒为 1（linear_solves=1）。
pub(super) fn run_chord(case: &Case, s: &[Complex64], v0: &[Complex64], solver: &mut KLUSolver) -> Cell {
    use rustpower::basic::benchmark_internals::{csc_matvec_and_scalc, fill_jacobian_v4};
    use rustpower::basic::jacobian_cache::JacobianCache;

    let t = Instant::now();
    let (npv, npq) = (case.npv, case.npq);
    let n_act = npv + npq;
    let n_state = npv + 2 * npq;
    let n_bus = case.v.len();
    let pat = JacobianCache::build_from_permuted(case.y.col_offsets(), case.y.row_indices(), npv, npq);
    let mut j_values = vec![0.0; pat.j_row_indices.len()];
    let mut ibus = vec![Complex64::new(0.0, 0.0); n_bus];
    let mut scalc = vec![Complex64::new(0.0, 0.0); n_bus];
    let mut f = vec![0.0; n_state];
    let mut v: Vec<Complex64> = v0.to_vec();

    let (ycp, yri, yv) = (case.y.col_offsets(), case.y.row_indices(), case.y.values());
    // 残差 F 与 ∞-范数（mis = scalc − sbus，与生产 newton_pf 同约定，
    // matvec 直接复用生产 csc_matvec_and_scalc）；
    // scalc 同时供冻结点的 fill_jacobian_v4 对角修正使用。
    let residual = |ibus: &mut [Complex64], scalc: &mut [Complex64], f: &mut [f64], v: &[Complex64]| -> f64 {
        csc_matvec_and_scalc(ycp, yri, yv, v, ibus, scalc);
        let mut norm = 0.0f64;
        for i in 0..n_act {
            let mis = scalc[i] - s[i];
            f[i] = mis.re;
            norm = norm.max(mis.re.abs());
        }
        for i in 0..npq {
            f[n_act + i] = scalc[i].im - s[i].im;
            norm = norm.max(f[n_act + i].abs());
        }
        norm
    };
    let apply_step = |v: &mut [Complex64], dx: &[f64]| {
        let mut v_m: Vec<f64> = v.iter().map(|e| e.norm()).collect();
        let mut v_a: Vec<f64> = v.iter().map(|e| e.arg()).collect();
        for k in 0..n_act {
            v_a[k] -= dx[k];
        }
        for k in 0..npq {
            v_m[k] -= dx[n_act + k];
        }
        for k in 0..n_bus {
            v[k] = Complex64::from_polar(v_m[k], v_a[k]);
        }
    };

    let mut iters = 0usize;
    let mut norm = residual(&mut ibus, &mut scalc, &mut f, &v);
    let mut failed = false;
    if norm >= TOL {
        // 冻结点：J(v0) 组装一次、分解一次。
        let v_norm: Vec<Complex64> = v
            .iter()
            .map(|e| {
                let m = e.norm();
                if m > 1e-12 { e / m } else { Complex64::new(1.0, 0.0) }
            })
            .collect();
        fill_jacobian_v4::<false>(
            &case.y,
            &v,
            &v_norm,
            &scalc,
            &pat.j_col_ptrs,
            &pat.pq_ends,
            &pat.active_ends,
            &pat.diag_ptrs,
            npv,
            npq,
            &mut j_values,
        );
        let mut ap = pat.j_col_ptrs.clone();
        let mut ai = pat.j_row_indices.clone();
        if solver.solve(&mut ap, &mut ai, &mut j_values, &mut f, n_state).is_err() {
            failed = true;
        } else {
            apply_step(&mut v, &f);
            iters += 1;
            norm = residual(&mut ibus, &mut scalc, &mut f, &v);
            while norm >= TOL && iters < MAXIT {
                if solver.resolve(&mut f).is_err() {
                    failed = true;
                    break;
                }
                apply_step(&mut v, &f);
                iters += 1;
                norm = residual(&mut ibus, &mut scalc, &mut f, &v);
            }
        }
    }
    let converged = !failed && norm < TOL;
    finish(case, s, &v, converged, iters, u64::from(converged), t)
}

/// `--lm-options '<json>'`：LmOptions 的 serde JSON，缺省字段用 Default。
pub(super) fn lm_options() -> (LmOptions, String) {
    match bench::option("--lm-options") {
        Some(json) => {
            let opts: LmOptions = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("--lm-options JSON 解析失败: {e}"));
            (opts, json)
        }
        None => (LmOptions::default(), "LmOptions::default()".into()),
    }
}
