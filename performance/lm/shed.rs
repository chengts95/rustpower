//! LS 证书信息量裁决入口 `lm-shed`：越墙后的最小二乘残差到底有没有
//! 弱区定位信息？
//!
//! 协议（单个算例、单个越墙 α）：
//! 1. GN-LM 从 flat 起点求 LS 点（参数可被 --lm-options 覆盖）；
//! 2. 逐 PQ 母线取残差 r_i = v_i·conj((Yv)_i) − s_i；
//! 3. 固定减负荷总预算（占总负荷的比例），三种分配方式对照：
//!    A = 按 |r_i| 比例分配（LS 证书建议的靶向切除）；
//!    B = 均匀分配（相当于退回墙内，无定位信息的基线）；
//!    C = 随机权重分配（固定种子，多试次）；
//!    只有带负荷的 PQ 母线参与切除；
//! 4. 判官是切除后的 NR(flat)——最严格的可行性判据。
//!
//! 若 A 在更小预算下恢复可行而 B/C 需要更大预算，则 LS 残差分布携带
//! 弱区定位信息；若三者无差别，则不可行是全局性的，证书无定位价值。
//! 用法：
//!   cargo bench --features benchmark --bench comparison -- lm-shed \
//!       --case IEEE39 --alpha 2.15

use super::comparison::{Case, load_case};
use super::harness::{lm_options, run_gn_v, run_nr};
use crate::bench;
use nalgebra::DVector;
use num_complex::Complex64;
use rustpower::basic::solver::QDLDLSolver;
use rustpower::lm::gn_triu::GnTriuDriver;
use serde_json::json;

/// 逐母线复残差（仅前 npv+npq 行有意义）。
fn residual_per_bus(case: &Case, s: &[Complex64], v: &[Complex64]) -> Vec<Complex64> {
    let current = &case.y * &DVector::from_column_slice(v);
    (0..case.npv + case.npq)
        .map(|i| v[i] * current[i].conj() - s[i])
        .collect()
}

/// 固定种子的 xorshift64——对照组抽样要可复现，不引外部依赖。
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// 把切片随机置换（用于 C 组：与 A 同一组权重、随机布置）。
    fn shuffle(&mut self, v: &mut [f64]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// 按权重从带负荷 PQ 母线切除总量为 budget 的负荷（等比例降 P、Q），
/// 并把实际切除的有功总量按当前出力比例同步下调 PV 发电
/// （只切负荷不切发电会让 slack 吸收数十 pu 盈余，物理上不可行）。
/// 返回切除后的 spec。权重 ≤ 0 的母线不参与。
fn shed(s: &[Complex64], npq: usize, npv: usize, weights: &[f64], budget: f64) -> Vec<Complex64> {
    let w_sum: f64 = (0..npq).filter(|&i| s[i].norm() > 1e-9).map(|i| weights[i].max(0.0)).sum();
    let mut out = s.to_vec();
    if w_sum <= 0.0 {
        return out;
    }
    let mut real_cut = 0.0;
    for i in 0..npq {
        let load = s[i].norm();
        if load <= 1e-9 {
            continue;
        }
        let cut = budget * weights[i].max(0.0) / w_sum;
        let keep = (1.0 - cut / load).max(0.0);
        real_cut += (1.0 - keep) * (-s[i].re).max(0.0);
        out[i] = s[i] * keep;
    }
    let gen_sum: f64 = (npq..npq + npv).map(|i| s[i].re.max(0.0)).sum();
    if gen_sum > 1e-9 {
        for i in npq..npq + npv {
            let g = s[i].re.max(0.0);
            out[i].re -= real_cut * g / gen_sum;
        }
    }
    out
}

pub fn run() {
    let name = bench::option("--case").unwrap_or_else(|| "IEEE39".into());
    let alpha: f64 = bench::option("--alpha")
        .map(|s| s.parse().expect("--alpha 需要浮点数"))
        .unwrap_or(2.15);
    let trials: usize = bench::option("--trials")
        .map(|s| s.parse().expect("--trials 需要正整数"))
        .unwrap_or(8);
    let case = load_case(&name);
    let (opts, options_note) = lm_options();
    let s_scaled: Vec<Complex64> = case.s.iter().map(|s| s * alpha).collect();

    // 1. LS 点（越墙时不收敛到 tol，但停在 LS 驻点）。
    let mut gn = GnTriuDriver::build_operator(&case.y, case.npv, case.npq, s_scaled.clone());
    let mut gn_solver = QDLDLSolver::default();
    let (ls, v_ls) = run_gn_v(&case, &mut gn, &mut gn_solver, &s_scaled, &case.v, &opts);
    let total_load: f64 = s_scaled[..case.npq].iter().map(|s| s.norm()).sum();
    println!(
        "=== {name} α={alpha:.2} LS 证书实验 | GN: {options_note} ===\n  LS 点: merit={:.6e} res∞={:.3e} minV_PQ={:.3} | 总负荷={total_load:.1}",
        ls.merit, ls.res_inf, ls.min_vmag_pq
    );

    // 2. 残差分布（定位信息候选）。
    let r = residual_per_bus(&case, &s_scaled, &v_ls);
    let w_res: Vec<f64> = r.iter().map(|x| x.norm()).collect();
    let r_sum: f64 = w_res[..case.npq].iter().sum();
    let mut rank: Vec<usize> = (0..case.npq).collect();
    rank.sort_by(|&a, &b| w_res[b].partial_cmp(&w_res[a]).unwrap());
    println!(
        "  Σ|r|={:.3e}（占总负荷 {:.2}%）；top 残差母线: {:?}",
        r_sum,
        100.0 * r_sum / total_load,
        rank.iter().take(8).map(|&i| format!("#{i}({:.1e})", w_res[i])).collect::<Vec<_>>()
    );

    // 3./4. 三种分配各自的最小可行预算（对分搜索）× NR(flat) 裁决。
    // Σ|r| 系统性低估真实所需切除量（非线性：切除改变运行点），
    // 因此不固定预算网格，而是直接测每种分配的恢复阈值。
    let out_dir = std::env::var("RUSTPOWER_ROBUSTNESS_OUTPUT")
        .expect("main 应设置 RUSTPOWER_ROBUSTNESS_OUTPUT");
    let n_loaded = s_scaled[..case.npq].iter().filter(|s| s.norm() > 1e-9).count();
    let feasible = |w: &[f64], budget: f64| {
        let s_cut = shed(&s_scaled, case.npq, case.npv, w, budget);
        run_nr(&case, &DVector::from_column_slice(&s_cut), &case.v).0.converged
    };
    // 最小可行预算：上界从 Σ|r| 起倍增直到可行，再对分 12 次。
    let min_budget = |w: &[f64]| -> f64 {
        let mut lo = 0.0;
        let mut hi = r_sum.max(total_load * 0.005);
        while !feasible(w, hi) && hi < total_load {
            hi *= 2.0;
        }
        if hi >= total_load {
            return f64::NAN; // 全切也不可行（不该发生）
        }
        for _ in 0..12 {
            let mid = 0.5 * (lo + hi);
            if feasible(w, mid) { hi = mid } else { lo = mid }
        }
        hi
    };
    // B 基线 = 按负荷比例（等效于整体退回 α，无定位信息）。
    // C 对照 = 与 A 同一组权重在带负荷母线集合内随机置换（检验"布置"是否重要）。
    let w_prop: Vec<f64> = (0..case.npq + case.npv)
        .map(|i| if i < case.npq { s_scaled[i].norm() } else { 0.0 })
        .collect();
    let loaded: Vec<usize> = (0..case.npq).filter(|&i| s_scaled[i].norm() > 1e-9).collect();
    let a = min_budget(&w_res);
    let b = min_budget(&w_prop);
    let mut rng = Rng(0x5eed_0000);
    let mut c_vals = Vec::new();
    for _ in 0..trials {
        let mut w_c = w_res.clone();
        let mut vals: Vec<f64> = loaded.iter().map(|&i| w_c[i]).collect();
        rng.shuffle(&mut vals);
        for (&i, &v) in loaded.iter().zip(&vals) {
            w_c[i] = v;
        }
        c_vals.push(min_budget(&w_c));
    }
    let c_mean = c_vals.iter().sum::<f64>() / trials as f64;
    let c_min = c_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let c_max = c_vals.iter().cloned().fold(0.0_f64, f64::max);
    let pct = |x: f64| 100.0 * x / total_load;
    println!("  最小可行减负荷预算（占总负荷%）：");
    println!("    A(按残差) = {:.2}%", pct(a));
    println!("    B(按负荷) = {:.2}%", pct(b));
    println!("    C(随机)   = {:.2}%（{trials} 次，{:.2}–{:.2}）", pct(c_mean), pct(c_min), pct(c_max));
    let rows = vec![json!({
        "case": name, "alpha": alpha,
        "total_load": total_load, "sum_residual": r_sum,
        "a_min_budget_pct": pct(a), "b_min_budget_pct": pct(b),
        "c_min_budget_pct_mean": pct(c_mean),
        "c_min_budget_pct_min": pct(c_min), "c_min_budget_pct_max": pct(c_max),
        "trials": trials, "loaded_pq_buses": n_loaded,
    })];
    let path = format!("{out_dir}/shed-{name}-a{alpha:.2}.jsonl");
    let body: Vec<String> = rows.iter().map(|r| r.to_string()).collect();
    std::fs::write(&path, body.join("\n") + "\n").unwrap();
    println!("已写出 {path}（{} 行）", rows.len());
}
