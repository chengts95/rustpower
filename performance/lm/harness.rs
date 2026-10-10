//! LM 专项实验的共享骨架：所有实验都是 load_case → 缩放注入 → 求解 → 独立统计。
//! 各实验文件（robustness / shed）只写自己的协议与输出，求解与统计一律走这里。

use super::comparison::Case;
use crate::bench;
use nalgebra::DVector;
use num_complex::Complex64;
use rustpower::basic::newtonpf::newton_pf;
use rustpower::basic::solver::{KLUSolver, QDLDLSolver};
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
    let r = driver.solve_gn_with_options(
        &case.y,
        solver,
        &mut v,
        TOL,
        MAXIT,
        opts,
    );
    finish(case, s, &v, r.converged, r.iterations, driver.n_solves, t)
}

/// 与 run_gn 相同，但带回终止电压（LS 证书实验需要残差分布）。
pub(super) fn run_gn_v(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64], opts: &LmOptions) -> (Cell, Vec<Complex64>) {
    driver.sbus.clear();
    driver.sbus.extend_from_slice(s);
    let mut v = v0.to_vec();
    let t = Instant::now();
    let r = driver.solve_gn_with_options(
        &case.y,
        solver,
        &mut v,
        TOL,
        MAXIT,
        opts,
    );
    (finish(case, s, &v, r.converged, r.iterations, driver.n_solves, t), v)
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
