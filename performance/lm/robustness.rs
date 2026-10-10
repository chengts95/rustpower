//! LM 鲁棒性扫描入口 `lm-robustness`：负荷因子 α 的点式扫描上比较
//! {flat, dc} 启动 × {NR, GN-LM} 四种组合的收敛域。
//!
//! 协议（每个算例、每个 α 四格完全同参）：
//! - 指定注入 S_spec × α（P、Q 同步缩放，功率因数不变）；
//! - flat 起点 = 算例自带初值；dc 起点 = 生产 DCPF 插件同一模型
//!   （import 模式带 pandapower 相移注入，zip 算例置零）；
//! - NR = 生产 `newton_pf`（KLU）；GN = 生产默认路径
//!   （triu-operator + QDLDL，参数见 --lm-options）；
//! - tol∞ = 1e-8，外层上限 300。
//!
//! 每格记录：收敛/失败、迭代数（GN 另记线性求解次数）、独立残差∞、
//! ½‖r‖²（越墙后的最小二乘证书）、PQ 最小电压幅值（< 0.8 标 low_voltage，
//! 用于识别鼻点附近的非物理解支）。α 以 0.05 步进，四格全败即停，上限 3.0。
//! 输出 JSONL + 每算例四堵墙汇总；更密的网格留作后处理，不在此自适应。

use super::comparison::{Case, load_case};
use super::harness::{MAXIT, TOL, lm_options, run_gn, run_nr, run_sd, run_ts};
use crate::bench;
use nalgebra::DVector;
use num_complex::Complex64;
use rustpower::basic::dcpf::{DcpfModel, dcpf_initial_v};
use rustpower::basic::solver::{KLUSolver, QDLDLSolver};
use rustpower::lm::gn_triu::GnTriuDriver;
use serde_json::json;

const ALPHA_STEP: f64 = 0.05;
const ALPHA_MAX: f64 = 3.0;

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
    let (opts, options_note) = lm_options();

    // 八格各自的墙（最后收敛的 α）；分开记"干净解"与"含低压解"。
    // 方法：NR = 生产 Newton；GN = 调度/信赖域 LM；SD = γ=‖F‖² 自缩放阻尼；
    // TS = SD 的三步复合（一次分解三枪，TSLM）。
    let mut wall = vec![(-1.0f64, -1.0f64); 8]; // (last_converged, last_clean)
    let labels = [
        "flat+NR", "flat+GN", "flat+SD", "flat+TS", "dc+NR", "dc+GN", "dc+SD", "dc+TS",
    ];

    let mut alpha = 1.0;
    println!(
        "\n=== {} (nb={}, n_act={}, npq={}) | tol={TOL} maxit={MAXIT} | GN: triu-operator+QDLDL {options_note} | SD: γ=‖F‖² ===",
        case.name,
        case.y.ncols(),
        n_act,
        case.npq
    );
    loop {
        let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();
        let s_dvec = DVector::from_column_slice(&s_scaled);
        let v_dc = dcpf_initial_v(&mut dc_model, &s_dvec, &v_init, &mut theta_ws, &mut dc_solver).ok();

        let mut cells = Vec::with_capacity(8);
        let (mut c, _) = run_nr(case, &s_dvec, &case.v);
        c.start = "flat";
        c.method = "nr";
        cells.push(c);
        let mut c = run_gn(case, &mut gn, &mut gn_solver, &s_scaled, &case.v, &opts);
        c.start = "flat";
        c.method = "gn";
        cells.push(c);
        let mut c = run_sd(case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "sd";
        cells.push(c);
        let mut c = run_ts(case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "ts";
        cells.push(c);
        if let Some(v0) = &v_dc {
            let (mut c, _) = run_nr(case, &s_dvec, v0.as_slice());
            c.start = "dc";
            c.method = "nr";
            cells.push(c);
            let mut c = run_gn(case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice(), &opts);
            c.start = "dc";
            c.method = "gn";
            cells.push(c);
            let mut c = run_sd(case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "sd";
            cells.push(c);
            let mut c = run_ts(case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "ts";
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
            "  α={alpha:.2} | flat: NR {:>6} GN {:>6} SD {:>6} TS {:>6} | dc: NR {:>6} GN {:>6} SD {:>6} TS {:>6}",
            show("flat", "nr"),
            show("flat", "gn"),
            show("flat", "sd"),
            show("flat", "ts"),
            show("dc", "nr"),
            show("dc", "gn"),
            show("dc", "sd"),
            show("dc", "ts"),
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
    if bench::option("--handoff").is_some() {
        if let Some(name) = bench::option("--import") {
            let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let path = format!("{dir}/target/research/lm_robustness/{name}.jsonl");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|_| panic!("找不到 {path}；先用 robustness_pp.py 生成"));
            let mut lines = text.lines();
            let (case, p_shift) = import_case(&name, lines.next().unwrap());
            let alphas: Vec<f64> = lines
                .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["alpha"].as_f64().unwrap())
                .collect();
            run_handoff(&case, &p_shift, &alphas);
        } else {
            let names: Vec<&str> = ["IEEE39", "IEEE118", "pegase9241"]
                .into_iter()
                .filter(|name| bench::selected(name))
                .collect();
            assert!(!names.is_empty(), "--case 没有匹配的算例");
            for name in names {
                let case = load_case(name);
                let p_shift = vec![0.0; case.v.len()];
                let alphas: Vec<f64> = (0..=40).map(|i| 1.0 + ALPHA_STEP * i as f64).collect();
                run_handoff(&case, &p_shift, &alphas);
            }
        }
        return;
    }
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
    /// DC 相移注入（pandapower makeBdc 的 Pbusinj，permuted 顺序）；
    /// 旧版导出文件没有此字段，按零注入退化。
    #[serde(default)]
    p_shift: Vec<f64>,
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

/// 从 pp 导出文件首行（base 记录）构造算例与 DC 相移注入。
/// 相移缺失时按零注入退化（与旧版导出文件兼容）。
fn import_case(name: &str, base_line: &str) -> (Case, Vec<f64>) {
    let base: PpBase = serde_json::from_str(base_line)
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
    let p_shift = if base.p_shift.len() == case.v.len() {
        base.p_shift.clone()
    } else {
        vec![0.0; case.v.len()]
    };
    (case, p_shift)
}

// ─── SD→NR 交接扫描（--handoff）──────────────────────────────────────
// flat 起点先由护送算法走 k 步，再交生产 NR（KLU）收尾。两种护送：
//   SD     = solve_self_damped（maxit=k，一次 QDLDL 分解/步）；
//   Cauchy = cauchy_escort（dSbus_dV Hermitian 解析梯度 + 精确线搜索，
//            3 次 Ybus matvec/步，零分解——"免费护送"候选）。
// 回答：最小交接步数 k*、总分解账、以及**真实耗时**（护送段与 NR 段
// 分别计时）。k=0 列即 flat+NR 对照；dc 列即 dc+NR 对照。

const K_GRID: &[usize] = &[0, 1, 2, 5, 10, 20, 50, 100, 200];
const CA_GRID: &[usize] = &[1, 2, 5, 10, 20];

struct EscortCell {
    k: usize,
    cauchy: bool,
    mark: String,
    converged: bool,
    solves: u64,
    escort_ms: f64,
    nr_ms: f64,
    nr_iters: usize,
    sd_self_converged: bool,
}

impl EscortCell {
    fn total_ms(&self) -> f64 {
        self.escort_ms + self.nr_ms
    }
}

fn escort_cell(
    case: &Case,
    gn: &mut GnTriuDriver,
    gn_solver: &mut QDLDLSolver,
    s_dvec: &DVector<Complex64>,
    s_scaled: &[Complex64],
    k: usize,
    cauchy: bool,
) -> EscortCell {
    gn.sbus.clear();
    gn.sbus.extend_from_slice(s_scaled);
    let mut v = case.v.clone();
    let t0 = std::time::Instant::now();
    let solves0 = gn.n_solves;
    let self_converged = if cauchy {
        gn.cauchy_escort(&case.y, &mut v, k);
        false // Cauchy 护送不判收敛，统一交 NR 裁决
    } else {
        gn.solve_self_damped(&case.y, gn_solver, &mut v, TOL, k).converged
    };
    let escort_ms = t0.elapsed().as_secs_f64() * 1e3;
    let escort_solves = gn.n_solves - solves0;
    if self_converged {
        return EscortCell {
            k, cauchy,
            mark: format!("SD✓{escort_solves}"),
            converged: true,
            solves: escort_solves,
            escort_ms, nr_ms: 0.0, nr_iters: 0,
            sd_self_converged: true,
        };
    }
    let (c, _) = run_nr(case, s_dvec, &v);
    EscortCell {
        k, cauchy,
        mark: if c.converged { format!("✓{}", c.iterations) } else { "✗".into() },
        converged: c.converged,
        solves: escort_solves + c.iterations as u64,
        escort_ms, nr_ms: c.wall_ms, nr_iters: c.iterations,
        sd_self_converged: false,
    }
}

fn run_handoff(case: &Case, p_shift: &[f64], alphas: &[f64]) {
    let n_act = case.npv + case.npq;
    let v_init = DVector::from_column_slice(&case.v);
    let identity: Vec<usize> = (0..case.v.len()).collect();
    let mut dc_model = DcpfModel::from_ybus(&case.y, &v_init, n_act, &identity, p_shift);
    let mut dc_solver = KLUSolver::default();
    let mut theta_ws = vec![0.0; n_act];
    let mut gn = GnTriuDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone());
    let mut gn_solver = QDLDLSolver::default();

    let out_dir = std::env::var("RUSTPOWER_ROBUSTNESS_OUTPUT")
        .expect("main 应设置 RUSTPOWER_ROBUSTNESS_OUTPUT");
    println!("\n=== {} SD/Cauchy→NR 交接扫描（flat 起点，tol={TOL}）===", case.name);
    println!("  k = SD护送步数（1次分解/步）；ca = Cauchy护送（3次matvec/步，零分解）");
    println!("  格 = NR收尾迭代数；best = 最小总耗时 标签(总分解数, 毫秒)");
    let mut out = Vec::new();
    let n_cols = 1 + K_GRID.len() + CA_GRID.len();
    let mut col_ms: Vec<Vec<f64>> = vec![Vec::new(); n_cols];
    for &alpha in alphas {
        let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();
        let s_dvec = DVector::from_column_slice(&s_scaled);
        let (dc_mark, dc_ms, dc_conv) =
            match dcpf_initial_v(&mut dc_model, &s_dvec, &v_init, &mut theta_ws, &mut dc_solver) {
                Ok(v0) => {
                    let (c, _) = run_nr(case, &s_dvec, v0.as_slice());
                    (
                        if c.converged { format!("✓{}", c.iterations) } else { "✗".into() },
                        c.wall_ms,
                        c.converged,
                    )
                }
                Err(_) => ("✗".into(), 0.0, false),
            };
        if dc_conv {
            col_ms[0].push(dc_ms);
        }
        out.push(json!({"case": case.name, "alpha": alpha, "kind": "dc", "converged": dc_conv, "total_ms": dc_ms}));

        let mut any_converged = dc_conv;
        let mut best: Option<(String, u64, f64)> = None; // (label, solves, ms)
        let mut cells = Vec::with_capacity(K_GRID.len() + CA_GRID.len());
        for &k in K_GRID {
            cells.push(escort_cell(case, &mut gn, &mut gn_solver, &s_dvec, &s_scaled, k, false));
        }
        for &k in CA_GRID {
            cells.push(escort_cell(case, &mut gn, &mut gn_solver, &s_dvec, &s_scaled, k, true));
        }
        for (ci, cell) in cells.iter().enumerate() {
            if cell.converged {
                any_converged = true;
                col_ms[1 + ci].push(cell.total_ms());
                if best.as_ref().map_or(true, |b| cell.total_ms() < b.2) {
                    let label = format!("{}{}", if cell.cauchy { "ca" } else { "k" }, cell.k);
                    best = Some((label, cell.solves, cell.total_ms()));
                }
            }
            out.push(json!({
                "case": case.name, "alpha": alpha,
                "kind": if cell.cauchy { "cauchy" } else { "sd" }, "k": cell.k,
                "converged": cell.converged, "solves": cell.solves,
                "escort_ms": cell.escort_ms, "nr_ms": cell.nr_ms,
                "nr_iters": cell.nr_iters, "sd_self_converged": cell.sd_self_converged,
            }));
        }
        let sd_marks = cells[..K_GRID.len()]
            .iter().map(|c| format!("{:>5}", c.mark)).collect::<Vec<_>>().join(" ");
        let ca_marks = cells[K_GRID.len()..]
            .iter().map(|c| format!("{:>5}", c.mark)).collect::<Vec<_>>().join(" ");
        let best_str = best.map_or("—".into(), |(l, s, ms)| format!("{l}({s}, {ms:.2}ms)"));
        println!("  α={alpha:.2} | dc {dc_mark:>4} | {sd_marks} | ca {ca_marks} | best {best_str}");
        if !any_converged && alpha > 1.05 {
            break; // 全灭即停（含 dc 对照列）
        }
    }
    let avg = |v: &Vec<f64>| if v.is_empty() { "—".into() } else { format!("{:.2}", v.iter().sum::<f64>() / v.len() as f64) };
    let sd_avgs = (1..=K_GRID.len()).map(|i| format!("{:>5}", avg(&col_ms[i]))).collect::<Vec<_>>().join(" ");
    let ca_avgs = (1 + K_GRID.len()..n_cols).map(|i| format!("{:>5}", avg(&col_ms[i]))).collect::<Vec<_>>().join(" ");
    println!("  平均总耗时ms | dc {:>4} | {sd_avgs} | ca {ca_avgs}", avg(&col_ms[0]));
    println!("  k 列顺序: {} | ca 列顺序: {}", K_GRID.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(" "),
        CA_GRID.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(" "));
    let path = format!("{out_dir}/handoff-{}.jsonl", case.name);
    let body: Vec<String> = out.iter().map(|r| r.to_string()).collect();
    std::fs::write(&path, body.join("\n") + "\n").unwrap();
    println!("已写出 {path}（{} 行）", out.len());
}

fn run_import(name: &str) {
    let out_dir = std::env::var("RUSTPOWER_ROBUSTNESS_OUTPUT")
        .expect("main 应设置 RUSTPOWER_ROBUSTNESS_OUTPUT");
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = format!("{dir}/target/research/lm_robustness/{name}.jsonl");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("找不到 {path}；先用 robustness_pp.py 生成"));
    let mut lines = text.lines();
    let (case, p_shift) = import_case(name, lines.next().unwrap());
    let n_act = case.npv + case.npq;
    let v_init = DVector::from_column_slice(&case.v);
    let identity: Vec<usize> = (0..case.v.len()).collect();
    let mut dc_model = DcpfModel::from_ybus(&case.y, &v_init, n_act, &identity, &p_shift);
    let mut dc_solver = KLUSolver::default();
    let mut theta_ws = vec![0.0; n_act];
    let mut gn = GnTriuDriver::build_operator(&case.y, case.npv, case.npq, case.s.clone());
    let mut gn_solver = QDLDLSolver::default();
    let (opts, options_note) = lm_options();

    let mut out = Vec::new();
    let mut walls = vec![-1.0f64; 10]; // our flatNR flatGN flatSD flatTS dcNR dcGN dcSD dcTS | pp flat dc
    let wall_labels = [
        "our flat+NR", "our flat+GN", "our flat+SD", "our flat+TS",
        "our dc+NR", "our dc+GN", "our dc+SD", "our dc+TS",
        "pp flat", "pp dc",
    ];

    println!("\n=== {name}（pandapower 同模型对照）| GN: {options_note} | SD: γ=‖F‖² ===");
    for line in lines {
        let row: PpRow = serde_json::from_str(line).unwrap();
        let alpha = row.alpha;
        let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();
        let s_dvec = DVector::from_column_slice(&s_scaled);
        let v_dc_ours = if bench::option("--pp-dc-start").is_some() {
            // 实验开关：dc 行直接用 pandapower 的 dc 起点（裁决"起点差异"还是"求解器差异"）。
            Some(DVector::from_column_slice(
                &row.dc_start_re.iter().zip(&row.dc_start_im)
                    .map(|(&r, &i)| Complex64::new(r, i))
                    .collect::<Vec<_>>(),
            ))
        } else {
            dcpf_initial_v(&mut dc_model, &s_dvec, &v_init, &mut theta_ws, &mut dc_solver).ok()
        };
        let d_start = v_dc_ours.as_ref().map(|v| {
            max_dangle(&row.dc_start_re, &row.dc_start_im, v.as_slice(), n_act)
        });

        let mut cells = Vec::with_capacity(8);
        let mut v_dc_nr: Option<Vec<Complex64>> = None;
        let (mut c, v_flat_nr) = run_nr(&case, &s_dvec, &case.v);
        let v_flat_nr = Some(v_flat_nr);
        c.start = "flat";
        c.method = "nr";
        cells.push(c);
        let mut c = run_gn(&case, &mut gn, &mut gn_solver, &s_scaled, &case.v, &opts);
        c.start = "flat";
        c.method = "gn";
        cells.push(c);
        let mut c = run_sd(&case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "sd";
        cells.push(c);
        let mut c = run_ts(&case, &mut gn, &mut gn_solver, &s_scaled, &case.v);
        c.start = "flat";
        c.method = "ts";
        cells.push(c);
        if let Some(v0) = &v_dc_ours {
            let (mut c, v) = run_nr(&case, &s_dvec, v0.as_slice());
            c.start = "dc";
            c.method = "nr";
            v_dc_nr = Some(v);
            cells.push(c);
            let mut c = run_gn(&case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice(), &opts);
            c.start = "dc";
            c.method = "gn";
            cells.push(c);
            let mut c = run_sd(&case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "sd";
            cells.push(c);
            let mut c = run_ts(&case, &mut gn, &mut gn_solver, &s_scaled, v0.as_slice());
            c.start = "dc";
            c.method = "ts";
            cells.push(c);
        }
        for (idx, cell) in cells.iter().enumerate() {
            if cell.converged {
                walls[idx] = alpha;
            }
        }
        if row.flat.converged {
            walls[8] = alpha;
        }
        if row.dc.converged {
            walls[9] = alpha;
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
        let dv_dc = if cells.get(4).map(|c| c.converged).unwrap_or(false) {
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
            "  α={alpha:.2} | our flat: NR {:>5} GN {:>5} SD {:>5} TS {:>5} | our dc: NR {:>5} GN {:>5} SD {:>5} TS {:>5} | pp: flat {:>5} dc {:>5} | ΔV flat {} dc {} | Δθstart {}",
            show("flat", "nr"),
            show("flat", "gn"),
            show("flat", "sd"),
            show("flat", "ts"),
            show("dc", "nr"),
            show("dc", "gn"),
            show("dc", "sd"),
            show("dc", "ts"),
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
