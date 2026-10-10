//! LM 鲁棒性扫描入口 `lm-robustness`：负荷因子 α 的点式扫描上比较
//! {flat, dc} 启动 × {NR, GN-LM} 四种组合的收敛域。
//!
//! 协议（每个算例、每个 α 四格完全同参）：
//! - 指定注入 S_spec × α（P、Q 同步缩放，功率因数不变）；
//! - flat 起点 = 算例自带初值；dc 起点 = 生产 DCPF 插件同一模型
//!   （`DcpfModel::from_ybus` + `dcpf_initial_v`，相移注入置零）；
//! - NR = 生产 `newton_pf`（KLU）；GN = 生产默认路径
//!   （triu-operator + QDLDL，`LmOptions::default()`）；
//! - tol∞ = 1e-8，外层上限 300。
//!
//! 每格记录：收敛/失败、迭代数（GN 另记线性求解次数）、独立残差∞、
//! ½‖r‖²（越墙后的最小二乘证书）、PQ 最小电压幅值（< 0.8 标 low_voltage，
//! 用于识别鼻点附近的非物理解支）。α 以 0.05 步进，四格全败即停，上限 3.0。
//! 输出 JSONL + 每算例四堵墙汇总；更密的网格留作后处理，不在此自适应。

use crate::bench;
use crate::lm_comparison::{Case, load_case};
use nalgebra::DVector;
use num_complex::Complex64;
use rustpower::basic::dcpf::{DcpfModel, dcpf_initial_v};
use rustpower::basic::newtonpf::newton_pf;
use rustpower::basic::solver::{KLUSolver, QDLDLSolver};
use rustpower::lm::LmOptions;
use rustpower::lm::gn_triu::GnTriuDriver;
use serde_json::json;
use std::time::Instant;

const TOL: f64 = 1e-8;
const MAXIT: usize = 300;
const ALPHA_STEP: f64 = 0.05;
const ALPHA_MAX: f64 = 3.0;
/// PQ 最小幅值低于此值视为可疑低压解支（原始数值同时记录，可后处理调整）。
const LOW_VMAG: f64 = 0.8;

struct Cell {
    start: &'static str,
    method: &'static str,
    converged: bool,
    iterations: usize,
    linear_solves: u64,
    res_inf: f64,
    merit: f64,
    min_vmag_pq: f64,
    wall_ms: f64,
}

impl Cell {
    fn outcome(&self) -> &'static str {
        if !self.converged {
            "failed"
        } else if self.min_vmag_pq < LOW_VMAG {
            "low_voltage"
        } else {
            "solution"
        }
    }

    fn mark(&self) -> String {
        match self.outcome() {
            "solution" => format!("✓{}", self.iterations),
            "low_voltage" => format!("⚠{}({:.2})", self.iterations, self.min_vmag_pq),
            _ => "✗".into(),
        }
    }
}

/// 独立的 (∞范数, ½‖r‖²) 统计，不依赖任何驱动的内部残差。
fn residual_stats(case: &Case, s: &[Complex64], v: &[Complex64]) -> (f64, f64) {
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

fn finish(case: &Case, s: &[Complex64], v: &[Complex64], converged: bool, iterations: usize, linear_solves: u64, wall: Instant) -> Cell {
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

fn run_nr(case: &Case, s: &DVector<Complex64>, v0: &[Complex64]) -> (Cell, Vec<Complex64>) {
    let t = Instant::now();
    let mut solver = KLUSolver::default();
    let v0 = DVector::from_column_slice(v0);
    let (cell, v) = match newton_pf(&case.y, s, &v0, case.npv, case.npq, Some(TOL), Some(MAXIT), &mut solver, None) {
        Ok((v, it)) => (finish(case, s.as_slice(), v.as_slice(), true, it, 0, t), v),
        Err((_, v, it)) => (finish(case, s.as_slice(), v.as_slice(), false, it, 0, t), v),
    };
    (cell, v.as_slice().to_vec())
}

/// GN 驱动在整个 α 扫描上复用（符号结构与 α 无关；sbus 逐格覆盖）。
fn run_gn(case: &Case, driver: &mut GnTriuDriver, solver: &mut QDLDLSolver, s: &[Complex64], v0: &[Complex64]) -> Cell {
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
        &LmOptions::default(),
    );
    finish(case, s, &v, r.converged, r.iterations, driver.n_solves, t)
}

fn sweep_case(case: &Case, out: &mut Vec<serde_json::Value>) {
    let n_act = case.npv + case.npq;
    let v_init = DVector::from_column_slice(&case.v);
    // 生产 DCPF 模型；相移注入置零（DC 起点忽略相移，与插件默认一致以外的偏差会注明）。
    let identity: Vec<usize> = (0..case.v.len()).collect();
    let zeros = vec![0.0; case.v.len()];
    let mut dc_model = DcpfModel::from_ybus(&case.y, &v_init, n_act, &identity, &zeros);
    let mut dc_solver = KLUSolver::default();
    let mut theta_ws = vec![0.0; n_act];
    let mut gn = GnTriuDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone());
    let mut gn_solver = QDLDLSolver::default();
    let options_note = "LmOptions::default()";

    // 四格各自的墙（最后收敛的 α）；分开记"干净解"与"含低压解"。
    let mut wall = vec![(-1.0f64, -1.0f64); 4]; // (last_converged, last_clean)
    let labels = ["flat+NR", "flat+GN", "dc+NR", "dc+GN"];

    let mut alpha = 1.0;
    println!(
        "\n=== {} (nb={}, n_act={}, npq={}) | tol={TOL} maxit={MAXIT} | GN: triu-operator+QDLDL {options_note} ===",
        case.name,
        case.y.ncols(),
        n_act,
        case.npq
    );
    loop {
        let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();
        let s_dvec = DVector::from_column_slice(&s_scaled);
        let v_dc = dcpf_initial_v(&mut dc_model, &s_dvec, &v_init, &mut theta_ws, &mut dc_solver).ok();

        let mut cells = Vec::with_capacity(4);
        let (mut c, _) = run_nr(case, &s_dvec, &case.v);
        c.start = "flat";
        c.method = "nr";
        cells.push(c);
        let mut c = run_gn(case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "gn";
        cells.push(c);
        if let Some(v0) = &v_dc {
            let (mut c, _) = run_nr(case, &s_dvec, v0.as_slice());
            c.start = "dc";
            c.method = "nr";
            cells.push(c);
            let mut c = run_gn(case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "gn";
            cells.push(c);
        }

        for (idx, cell) in cells.iter().enumerate() {
            if cell.converged {
                wall[idx].0 = alpha;
                if cell.outcome() == "solution" {
                    wall[idx].1 = alpha;
                }
            }
            out.push(json!({
                "case": case.name, "alpha": alpha,
                "start": cell.start, "method": cell.method,
                "outcome": cell.outcome(), "converged": cell.converged,
                "iterations": cell.iterations, "linear_solves": cell.linear_solves,
                "residual_inf": cell.res_inf, "merit_half_r2": cell.merit,
                "min_vmag_pq": cell.min_vmag_pq, "wall_ms": cell.wall_ms,
            }));
        }
        let show = |start: &str, method: &str| {
            cells
                .iter()
                .find(|c| c.start == start && c.method == method)
                .map(|c| c.mark())
                .unwrap_or_else(|| "dc失败".into())
        };
        println!(
            "  α={alpha:.2} | flat: NR {:>6} GN {:>6} | dc: NR {:>6} GN {:>6}",
            show("flat", "nr"),
            show("flat", "gn"),
            show("dc", "nr"),
            show("dc", "gn"),
        );

        let all_failed = cells.iter().all(|c| !c.converged);
        if (all_failed && alpha > 1.05) || alpha >= ALPHA_MAX {
            break;
        }
        alpha += ALPHA_STEP;
    }

    println!("  --- 墙（最后收敛 α / 最后干净解 α）---");
    for (l, (wc, ws)) in labels.iter().zip(&wall) {
        let fmt = |a: f64| {
            if a < 0.0 {
                "无".to_string()
            } else {
                format!("{a:.2}")
            }
        };
        println!("  {l:8} : {} / {}", fmt(*wc), fmt(*ws));
    }
}

pub fn run() {
    if let Some(name) = bench::option("--import") {
        run_import(&name);
        return;
    }
    let out_dir = std::env::var("RUSTPOWER_ROBUSTNESS_OUTPUT")
        .expect("main 应设置 RUSTPOWER_ROBUSTNESS_OUTPUT");
    let names: Vec<&str> = ["IEEE39", "IEEE118", "pegase9241"]
        .into_iter()
        .filter(|name| bench::selected(name))
        .collect();
    assert!(!names.is_empty(), "--case 没有匹配的算例");
    let mut rows = Vec::new();
    for name in names {
        sweep_case(&load_case(name), &mut rows);
    }
    let path = format!("{out_dir}/sweep.jsonl");
    let body: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
    std::fs::write(&path, body.join("\n") + "\n").unwrap();
    println!("\n已写出 {}（{} 行）", path, rows.len());
}

// ─── pandapower 同模型对照模式（--import <case>）──────────────────────────
// 输入由 performance/python/robustness_pp.py 生成：首行为基准输入
// （Ybus CSC、基准注入、flat 起点），后续每行一个 α 的 pandapower 结果。
// 我方四格在同一输入上重放；flat 起点与 pandapower 逐位一致（同模型对照），
// dc 起点双方各用自己的生产模型（惯例不同，记录角度差仅作参考）。

#[derive(serde::Deserialize)]
struct PpBase {
    nb: usize,
    npv: usize,
    npq: usize,
    cp: Vec<usize>,
    ri: Vec<usize>,
    y_re: Vec<f64>,
    y_im: Vec<f64>,
    s_re: Vec<f64>,
    s_im: Vec<f64>,
    v_flat_re: Vec<f64>,
    v_flat_im: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct PpResult {
    converged: bool,
    iterations: usize,
    v_re: Vec<Option<f64>>,
    v_im: Vec<Option<f64>>,
}

#[derive(serde::Deserialize)]
struct PpRow {
    alpha: f64,
    dc_start_re: Vec<f64>,
    dc_start_im: Vec<f64>,
    flat: PpResult,
    dc: PpResult,
}

fn pp_v(re: &[Option<f64>], im: &[Option<f64>]) -> Option<Vec<Complex64>> {
    re.iter()
        .zip(im)
        .map(|(r, i)| Some(Complex64::new((*r)?, (*i)?)))
        .collect()
}

fn max_dv(pp: &[Complex64], ours: &[Complex64]) -> f64 {
    pp.iter()
        .zip(ours)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max)
}

fn max_dangle(pp_re: &[f64], pp_im: &[f64], ours: &[Complex64], n_act: usize) -> f64 {
    (0..n_act)
        .map(|i| {
            let pp_ang = Complex64::new(pp_re[i], pp_im[i]).arg();
            let d = (ours[i].arg() - pp_ang + std::f64::consts::PI)
                .rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
            d.abs()
        })
        .fold(0.0_f64, f64::max)
}

fn run_import(name: &str) {
    let out_dir = std::env::var("RUSTPOWER_ROBUSTNESS_OUTPUT")
        .expect("main 应设置 RUSTPOWER_ROBUSTNESS_OUTPUT");
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = format!("{dir}/target/research/lm_robustness/{name}.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("找不到 {path}；先用 robustness_pp.py 生成"));
    let mut lines = text.lines();
    let base: PpBase = serde_json::from_str(lines.next().unwrap())
        .map(|v: serde_json::Value| v)
        .ok()
        .and_then(|v| serde_json::from_value(v["base"].clone()).ok())
        .expect("首行必须是 base 记录");
    let complex = |re: &[f64], im: &[f64]| {
        re.iter().zip(im).map(|(&r, &i)| Complex64::new(r, i)).collect::<Vec<_>>()
    };
    let case = Case {
        name: name.into(),
        y: nalgebra_sparse::CscMatrix::try_from_csc_data(
            base.nb,
            base.nb,
            base.cp,
            base.ri,
            complex(&base.y_re, &base.y_im),
        )
        .unwrap(),
        s: complex(&base.s_re, &base.s_im),
        v: complex(&base.v_flat_re, &base.v_flat_im),
        npv: base.npv,
        npq: base.npq,
    };
    let n_act = case.npv + case.npq;
    let v_init = DVector::from_column_slice(&case.v);
    let identity: Vec<usize> = (0..case.v.len()).collect();
    let zeros = vec![0.0; case.v.len()];
    let mut dc_model = DcpfModel::from_ybus(&case.y, &v_init, n_act, &identity, &zeros);
    let mut dc_solver = KLUSolver::default();
    let mut theta_ws = vec![0.0; n_act];
    let mut gn = GnTriuDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone());
    let mut gn_solver = QDLDLSolver::default();

    let mut out = Vec::new();
    let mut walls = vec![-1.0f64; 6]; // our flatNR flatGN dcNR dcGN | pp flat dc
    let wall_labels = ["our flat+NR", "our flat+GN", "our dc+NR", "our dc+GN", "pp flat", "pp dc"];

    println!("\n=== {name}（pandapower 同模型对照）===");
    for line in lines {
        let row: PpRow = serde_json::from_str(line).unwrap();
        let alpha = row.alpha;
        let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();
        let s_dvec = DVector::from_column_slice(&s_scaled);
        let v_dc_ours =
            dcpf_initial_v(&mut dc_model, &s_dvec, &v_init, &mut theta_ws, &mut dc_solver).ok();
        let d_start = v_dc_ours.as_ref().map(|v| {
            max_dangle(&row.dc_start_re, &row.dc_start_im, v.as_slice(), n_act)
        });

        let mut cells = Vec::with_capacity(4);
        let mut v_dc_nr: Option<Vec<Complex64>> = None;
        let (mut c, v_flat_nr) = run_nr(&case, &s_dvec, &case.v);
        let v_flat_nr = Some(v_flat_nr);
        c.start = "flat";
        c.method = "nr";
        cells.push(c);
        let mut c = run_gn(&case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "gn";
        cells.push(c);
        if let Some(v0) = &v_dc_ours {
            let (mut c, v) = run_nr(&case, &s_dvec, v0.as_slice());
            c.start = "dc";
            c.method = "nr";
            v_dc_nr = Some(v);
            cells.push(c);
            let mut c = run_gn(&case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "gn";
            cells.push(c);
        }
        for (idx, cell) in cells.iter().enumerate() {
            if cell.converged {
                walls[idx] = alpha;
            }
        }
        if row.flat.converged {
            walls[4] = alpha;
        }
        if row.dc.converged {
            walls[5] = alpha;
        }

        // 电压对照（双方均收敛时）：ours flat+NR vs pp flat，ours dc+NR vs pp dc。
        // run_nr 已带回电压，直接比较即可。
        let dv = |ours: &Option<Vec<Complex64>>, pp: &PpResult| -> Option<f64> {
            if !pp.converged {
                return None;
            }
            let ours = ours.as_ref()?;
            let pp_v = pp_v(&pp.v_re, &pp.v_im)?;
            Some(max_dv(&pp_v, ours))
        };
        let dv_flat = if cells[0].converged { dv(&v_flat_nr, &row.flat) } else { None };
        let dv_dc = if cells.get(2).map(|c| c.converged).unwrap_or(false) {
            dv(&v_dc_nr, &row.dc)
        } else {
            None
        };

        let show = |start: &str, method: &str| {
            cells
                .iter()
                .find(|c| c.start == start && c.method == method)
                .map(|c| c.mark())
                .unwrap_or_else(|| "—".into())
        };
        let pp_mark = |r: &PpResult| {
            if r.converged {
                format!("✓{}", r.iterations)
            } else {
                "✗".to_string()
            }
        };
        println!(
            "  α={alpha:.2} | our flat: NR {:>5} GN {:>5} | our dc: NR {:>5} GN {:>5} | pp: flat {:>5} dc {:>5} | ΔV flat {} dc {} | Δθstart {}",
            show("flat", "nr"),
            show("flat", "gn"),
            show("dc", "nr"),
            show("dc", "gn"),
            pp_mark(&row.flat),
            pp_mark(&row.dc),
            dv_flat.map(|x| format!("{x:.1e}")).unwrap_or("—".into()),
            dv_dc.map(|x| format!("{x:.1e}")).unwrap_or("—".into()),
            d_start.map(|x| format!("{x:.1e}")).unwrap_or("—".into()),
        );
        out.push(json!({
            "case": name, "alpha": alpha,
            "ours": cells.iter().map(|c| json!({
                "start": c.start, "method": c.method, "outcome": c.outcome(),
                "iterations": c.iterations, "linear_solves": c.linear_solves,
                "residual_inf": c.res_inf, "merit_half_r2": c.merit,
                "min_vmag_pq": c.min_vmag_pq,
            })).collect::<Vec<_>>(),
            "pp": {"flat": {"converged": row.flat.converged, "iterations": row.flat.iterations},
                   "dc": {"converged": row.dc.converged, "iterations": row.dc.iterations}},
            "max_dv_flat_nr": dv_flat, "max_dv_dc_nr": dv_dc,
            "dc_start_max_dangle": d_start,
        }));
    }
    println!("  --- 墙（最后收敛 α）：我方四格 vs pandapower ---");
    for (l, w) in wall_labels.iter().zip(&walls) {
        println!("  {l:12} : {}", if *w < 0.0 { "无".into() } else { format!("{w:.2}") });
    }
    let path_out = format!("{out_dir}/import-{name}.jsonl");
    let body: Vec<String> = out.iter().map(|r| r.to_string()).collect();
    std::fs::write(&path_out, body.join("\n") + "\n").unwrap();
    println!("已写出 {path_out}（{} 行）", out.len());
}
