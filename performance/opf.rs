//! OPF性能入口：一份版本选择、一份计时、一份输出。
//! 原生网络比较与pandapower同模型审计共用solve_version；不把两种输入混为同一实验。
use crate::bench::{self, Report, timeit};
use rustpower::{new_opf, opf};
use serde_json::{Value, json};

pub const VERSIONS: &[&str] = &["V1", "V4", "V5.0", "V5.2", "V5.3", "V5.5", "V5.6"];
const COLUMNS: &[bench::Columns] = &[
    &[
        ("converged", "收敛"),
        ("message", "停止原因"),
        ("iterations", "迭代"),
        ("f", "目标值"),
        ("balance_inf", "功率平衡∞"),
        ("flow_violation", "支路平方功率越限"),
        ("bound_violation", "变量越限"),
    ],
    &[
        ("hess_ms", "Hessian/融合填充ms"),
        ("gh_ms", "G/H区域ms"),
        ("kkt_ms", "KKT区域ms"),
        ("assembly_ms", "组装区域合计ms"),
    ],
    &[
        ("first_solve_ms", "首次求解区域ms"),
        ("later_solves_ms", "后续求解区域ms"),
        ("total_ms", "总执行ms"),
    ],
    &[
        ("klu_symbolic_ms", "KLU符号ms"),
        ("klu_factor_ms", "KLU factor及fallback ms"),
        ("klu_refactor_ms", "KLU refactor ms"),
        ("klu_backsolve_ms", "KLU回代ms"),
    ],
    &[
        ("first_factor_count", "首次factor次数"),
        ("refactor_count", "refactor次数"),
        ("factor_fallback_count", "fallback次数"),
    ],
];

pub fn options(max_it: usize) -> opf::PipsOpt {
    opf::PipsOpt {
        max_it,
        cost_mult: 1e-4,
        ..Default::default()
    }
}
fn options_json(o: &opf::PipsOpt, version: &str) -> Value {
    json!({"feastol":o.feastol,"gradtol":o.gradtol,"comptol":o.comptol,"costtol":o.costtol,
        "max_it":o.max_it,"cost_mult":o.cost_mult,"merged_slacks":version != "V1"})
}

/// 初值/边界复制由调用方放在计时外；优化路径的模型复制、缓存和后端初始化在此计时内。
pub fn solve_version(
    version: &str,
    base: &opf::OPFData,
    seed: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    opt: opf::PipsOpt,
) -> opf::PipsResult {
    if version == "V1" {
        return opf::pips::pips(
            |x| opf::cost::opf_costfcn(base, x),
            |x| {
                let (g, h, dg, dh) = opf::constraints::opf_consfcn(base, x);
                (h, g, dh, dg)
            },
            |x, l, m, _z, c| opf::hessian::opf_hessfcn(base, x, l, m, c),
            seed,
            lower,
            upper,
            opt,
        );
    }
    let solve = match version {
        "V4" => new_opf::configurations::pips,
        "V5.0" => new_opf::configurations::pips_v5,
        "V5.2" => new_opf::configurations::pips_v5_2,
        "V5.3" => new_opf::configurations::pips_v5_3,
        "V5.5" => new_opf::configurations::pips_v5_5,
        "V5.6" => new_opf::configurations::pips_v5_6,
        _ => panic!("未知OPF版本：{version}"),
    };
    let data = new_opf::model::NewOPFData::new(base.clone());
    solve(&data, seed, lower, upper, opt)
}

/// 保留审计JSON字段；分项直接读取PipsTiming，首次/后续求解不是纯符号/数值分解。
pub fn measurement(result: &opf::PipsResult, total_ms: f64) -> Value {
    let t = &result.timing;
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    json!({"converged":result.converged,"iterations":result.iterations,"f":result.f,"message":result.message,
        "total_ms":total_ms,"hess_ms":ms(t.hess),"gh_ms":ms(t.gh),"kkt_ms":ms(t.kkt),
        "assembly_ms":ms(t.hess+t.gh+t.kkt),"first_solve_ms":ms(t.solve_sym),"later_solves_ms":ms(t.solve_num)})
}

/// 原有CSV成本配置只加载一次，所有版本共用同一模型和初值。
fn load_case(case: &str) -> opf::OPFData {
    let path = format!("{}/cases/{case}/data.zip", env!("CARGO_MANIFEST_DIR"));
    let net = rustpower::io::pandapower::load_csv_zip(&path).unwrap();
    let mut data = opf::builder::opf_data_from_network(&net);
    if let Some(cfg) = rustpower::io::pandapower::load_opf_cfg_zip(&path) {
        let (offset, count) = if case == "IEEE39" { (0, 10) } else { (1, 54) };
        if offset == 1 {
            if let Some(r) = cfg.get("ext_grid", 0) {
                data.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
            }
        }
        for g in 0..count {
            if let Some(r) = cfg.get("gen", g) {
                data.cost_coeffs[g as usize + offset] =
                    [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
            }
        }
    }
    data
}

/// 在计时外重新计算可行性；不以目标值相近代替约束检查。
fn feasibility(data: &opf::OPFData, x: &[f64], lo: &[f64], hi: &[f64]) -> (f64, f64, f64) {
    let (g, h, _, _) = opf::constraints::opf_consfcn(data, x);
    let max = |values: Vec<f64>| {
        values.into_iter().fold(
            0.0_f64,
            |a, b| {
                if b.is_nan() { f64::INFINITY } else { a.max(b) }
            },
        )
    };
    let balance = max(g.into_iter().map(f64::abs).collect());
    let flow = max(h);
    let bounds = max(x
        .iter()
        .zip(lo)
        .zip(hi)
        .map(|((&x, &lo), &hi)| {
            if !x.is_finite() {
                f64::INFINITY
            } else {
                (lo - x).max(x - hi)
            }
        })
        .collect());
    (balance, flow, bounds)
}

pub fn benchmark(versions: &[&str]) {
    let out = std::env::var("RUSTPOWER_OPF_OUTPUT").expect("OPF输出目录");
    let repeats = bench::repeats();
    let mut report = Report::new(out, COLUMNS);
    let cases: Vec<_> = ["IEEE39", "IEEE118", "pegase9241"]
        .into_iter()
        .filter(|c| bench::selected(c))
        .collect();
    assert!(!cases.is_empty(), "--case没有匹配的OPF算例");
    for case in cases {
        let base = load_case(case);
        let max_it = if case == "pegase9241" { 30 } else { 150 };
        let x0 = base.warm_x0();
        let (lo, hi) = base.bounds();
        println!(
            "\n{case}：原生网络输入，KLU，节点={}，发电机={}，支路={}；预热1次、测量{repeats}次。",
            base.nb, base.ng, base.nl
        );
        println!(
            "初值：warm_x0；参数：{}",
            options_json(&options(max_it), "V1")
        );
        println!(
            "V1不合并slack，V4–V5.6合并；Hessian/融合填充范围随版本变化。总时间包含模型缓存构建及首次分解。"
        );
        for (round, version) in bench::runs(versions, repeats) {
            let (seed, lower, upper) = (x0.clone(), lo.clone(), hi.clone());
            #[cfg(feature = "probe")]
            rustpower::basic::solver::klu_probe::reset();
            let (result, total_ms) = timeit!(solve_version(
                version,
                &base,
                seed,
                lower,
                upper,
                options(max_it)
            ));
            let mut row = measurement(&result, total_ms);
            // 独立检查前读取后端计时，避免诊断影响计数。
            #[cfg(feature = "probe")]
            {
                use rustpower::basic::solver::klu_probe as p;
                use std::sync::atomic::Ordering::Relaxed;
                for (key, counter) in [
                    ("klu_symbolic_ms", &p::SYM_NS),
                    ("klu_factor_ms", &p::FACTOR_NS),
                    ("klu_refactor_ms", &p::REFACTOR_NS),
                    ("klu_backsolve_ms", &p::SOLVE_NS),
                ] {
                    row[key] = (counter.load(Relaxed) as f64 / 1e6).into();
                }
                for (key, counter) in [
                    ("first_factor_count", &p::N_FIRST_FACTOR),
                    ("refactor_count", &p::N_REFACTOR),
                    ("factor_fallback_count", &p::N_FACTOR_FALLBACK),
                ] {
                    row[key] = counter.load(Relaxed).into();
                }
            }
            let (balance, flow, bounds) = feasibility(&base, &result.x, &lo, &hi);
            row.as_object_mut().unwrap().extend(json!({"case":case,"method":version,"version":version,"round":round,
                "options":options_json(&options(max_it),version),"input":"native CSV ZIP / warm_x0", "backend":"KLU",
                "balance_inf":balance,"flow_violation":flow,"bound_violation":bounds}).as_object().unwrap().clone());
            report.push(row, version);
        }
        report.summary(case);
    }
}
