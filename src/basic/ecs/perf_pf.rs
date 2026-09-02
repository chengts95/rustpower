//! 最终用户 API（ECS app + 插件）口径的性能对照：default_app 模板，
//! 默认 NR / IwamotoPlugin / GnPlugin / LmPlugin 四家。
//! 每个 app 跑两次 `update()`：cold = 首次（含建网、符号分析、首分解），
//! warm = 第二次（热路径）。release 下跑（加 probe 得各相位 breakdown）：
//! `cargo test --release --features "klu probe" perf_pf -- --nocapture`
#![cfg(all(test, feature = "klu"))]

use bevy_app::App;
use std::time::Instant;

use crate::basic::ecs::elements::PPNetwork;
use crate::basic::ecs::gn_plugin::GnPlugin;
use crate::basic::ecs::lm_plugin::LmPlugin;
use crate::basic::ecs::network::{DataOps, PowerFlow};
use crate::basic::ecs::plugin::{ActiveSolver, IwamotoPlugin, default_app};
use crate::basic::ecs::powerflow::systems::PowerFlowResult;
use crate::io::pandapower::{Network, load_csv_zip};

struct Run {
    method: &'static str,
    converged: bool,
    iterations: usize,
    cold: std::time::Duration,
    warm: std::time::Duration,
}

fn timed_app(net: Network, method: &'static str, add: impl FnOnce(&mut App)) -> Run {
    let mut app = default_app();
    add(&mut app);
    app.world_mut().insert_resource(PPNetwork(net));
    let t = Instant::now();
    app.update();
    let cold = t.elapsed();
    let t = Instant::now();
    app.update();
    let warm = t.elapsed();
    let r = app
        .world()
        .get_resource::<PowerFlowResult>()
        .expect("no PowerFlowResult")
        .clone();
    Run { method, converged: r.converged, iterations: r.iterations, cold, warm }
}

fn four_way(name: &str, net: impl Fn() -> Network) {
    println!("=== {name}（最终用户 API，app.update() 计时）===");
    println!("方法      | 收敛 | 迭代 | cold 首次 | warm 热路径");
    let runs = [
        timed_app(net(), "NR(默认)", |_| {}),
        timed_app(net(), "Iwamoto", |app| {
            app.add_plugins(IwamotoPlugin);
            app.world_mut().insert_resource(ActiveSolver::Iwamoto);
        }),
        timed_app(net(), "GN-LM", |app| {
            app.add_plugins(GnPlugin);
            app.world_mut().insert_resource(ActiveSolver::GaussNewtonLm);
        }),
        timed_app(net(), "exact-LM", |app| {
            app.add_plugins(LmPlugin);
            app.world_mut().insert_resource(ActiveSolver::ExactLm);
        }),
    ];
    for r in &runs {
        println!(
            "{:<9} | {} | {:4} | {:9.?} | {:9.?}",
            r.method,
            if r.converged { "✓" } else { "✗" },
            r.iterations,
            r.cold,
            r.warm,
        );
    }
}

#[test]
fn perf_pf_app_api() {
    four_way("IEEE39", || {
        serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap()
    });
    four_way("PEGASE9241", || {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        load_csv_zip(&format!("{dir}/cases/pegase9241/data.zip")).unwrap()
    });
}

/// KLU 探针拆解：GN-LM 的总时间里，符号/分解/回代各占多少，剩下的才是
/// 我们的装配侧。预期：装配 ≈ 0，时间全在 KLU（2n 维增广系统的 n^k 代价）。
/// 对照 NR（n 维系统）。release:
/// `cargo test --release --features klu klu_breakdown -- --nocapture`
#[test]
fn perf_pf_klu_breakdown() {
    use crate::basic::solver::KLUSolver;
    #[cfg(feature = "probe")]
    use crate::basic::solver::klu_probe;
    use crate::lm::gn_flat::newton_pf_gn;

    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net: Network = load_csv_zip(&format!("{dir}/cases/pegase9241/data.zip")).unwrap();
    let mut pf = crate::basic::ecs::network::PowerGrid::default();
    pf.world_mut().insert_resource(PPNetwork(net));
    pf.init_pf_net();
    let world = pf.world();
    let mat = world
        .get_resource::<crate::basic::ecs::powerflow::systems::PowerFlowMat>()
        .unwrap();
    let (ybus, sbus, v_init, npv, npq) = (&mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq);
    let n = npv + 2 * npq;
    println!("PEGASE9241: n={n}（增广 2n={}）", 2 * n);

    // NR（n 维系统）
    let mut s = KLUSolver::default();
    #[cfg(feature = "probe")]
    klu_probe::reset();
    let t = Instant::now();
    let r = crate::basic::newtonpf::newton_pf(ybus, sbus, v_init, npv, npq, Some(1e-8), Some(100), &mut s);
    let total = t.elapsed();
    let it = r.map(|(_, it)| it).unwrap_or(usize::MAX);
    println!("NR      : total={total:9.?} it={it:2}");
    #[cfg(feature = "probe")]
    println!("  {}", klu_probe::report());

    // GN-LM（2n 维增广系统）
    let mut s = KLUSolver::default();
    #[cfg(feature = "probe")]
    klu_probe::reset();
    let t = Instant::now();
    let r = newton_pf_gn(ybus, sbus, v_init, npv, npq, Some(1e-8), Some(100), &mut s);
    let total = t.elapsed();
    let it = r.map(|(_, it)| it).unwrap_or(usize::MAX);
    println!("GN-LM   : total={total:9.?} it={it:2}");
    #[cfg(feature = "probe")]
    println!("  {}", klu_probe::report());
}

/// 增广系统为什么贵：nnz 对比 + KLU ordering 实验（AMD vs COLAMD）+
/// JᵀJ 法方程的符号 nnz 预估（决定"死磕 JᵀJ"值不值）。
#[test]
fn perf_pf_augmented_vs_normal() {
    use crate::basic::solver::KLUSolver;
    use crate::lm::gn_flat::newton_pf_gn;
    use crate::lm::pattern::KktPattern;

    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net: Network = load_csv_zip(&format!("{dir}/cases/pegase9241/data.zip")).unwrap();
    let mut pf = crate::basic::ecs::network::PowerGrid::default();
    pf.world_mut().insert_resource(PPNetwork(net));
    pf.init_pf_net();
    let world = pf.world();
    let mat = world
        .get_resource::<crate::basic::ecs::powerflow::systems::PowerFlowMat>()
        .unwrap();
    let (ybus, sbus, v_init, npv, npq) = (&mat.y_bus, &mat.s_bus, &mat.v_bus_init, mat.npv, mat.npq);
    let n = npv + 2 * npq;

    // ── nnz 事实 ──
    let pat = KktPattern::build(ybus, npv, npq);
    let (cs, ri) = (&pat.graph.col_starts, &pat.graph.row_indices);
    let nnz_j = pat.graph.nnz;
    let max_ri = ri.iter().max().copied().unwrap_or(0);
    println!("诊断: nb={} n_act={} n={} graph列数={} max_ri={max_ri}",
        ybus.ncols(), npv + npq, n, cs.len() - 1);
    let nnz_aug = 2 * nnz_j + 2 * n; // slim 布局
    // JᵀJ 符号 nnz：方程→状态集合（J 列转置散射），再逐列并集。
    let g_n = cs.len() - 1;
    let g_rows = ri.iter().max().copied().unwrap_or(0) + 1;
    let mut eq_states: Vec<Vec<usize>> = vec![Vec::new(); g_rows];
    for c in 0..g_n {
        for p in cs[c]..cs[c + 1] {
            eq_states[ri[p]].push(c);
        }
    }
    let mut mark = vec![usize::MAX; g_n];
    let mut nnz_jtj = 0usize;
    for j in 0..g_n {
        let mut cnt = 0usize;
        for p in cs[j]..cs[j + 1] {
            for &s in &eq_states[ri[p]] {
                if mark[s] != j {
                    mark[s] = j;
                    cnt += 1;
                }
            }
        }
        nnz_jtj += cnt;
    }
    println!("nnz: Ybus={} J={} 增广[μI Jᵀ;J -I]={} JᵀJ+μI={} (比={:.2})",
        ybus.nnz(), nnz_j, nnz_aug, nnz_jtj, nnz_jtj as f64 / nnz_aug as f64);

    // ── 决定性实验：JᵀJ+μI 真符号 + 假数值，量 KLU 分解耗时。
    // klu refactor 复用主元顺序，耗时与数值无关 → 假数值即可量 fill-in。
    let mut cp = vec![0usize; g_n + 1];
    let mut mark2 = vec![usize::MAX; g_n];
    let mut col_rows: Vec<usize> = Vec::new();
    for j in 0..g_n {
        let start = col_rows.len();
        for p in cs[j]..cs[j + 1] {
            for &s in &eq_states[ri[p]] {
                if mark2[s] != j {
                    mark2[s] = j;
                    col_rows.push(s);
                }
            }
        }
        col_rows[start..].sort_unstable();
        cp[j + 1] = col_rows.len();
    }
    let mut ax: Vec<f64> = (0..col_rows.len()).map(|_| 1.0).collect();
    for j in 0..g_n {
        for p in cp[j]..cp[j + 1] {
            if col_rows[p] == j {
                ax[p] = 10.0; // 对角加大，保证假数值良性
            }
        }
    }
    let mut b: Vec<f64> = vec![1.0; g_n];
    let mut s = KLUSolver::default();
    #[cfg(feature = "probe")]
    crate::basic::solver::klu_probe::reset();
    let t = Instant::now();
    for _ in 0..11 {
        let mut cp_ = cp.clone();
        let mut ri_ = col_rows.clone();
        let mut ax_ = ax.clone();
        let _ = crate::basic::solver::Solve::solve(&mut s, &mut cp_, &mut ri_, &mut ax_, &mut b, g_n);
    }
    println!("JᵀJ+μI 假数值 11 次 solve 总 {:?}", t.elapsed());
    #[cfg(feature = "probe")]
    println!("  {}", crate::basic::solver::klu_probe::report());

    // ── KLU ordering 实验：0=AMD（默认） vs 1=COLAMD ──
    for ord in [0i32, 1] {
        let mut s = KLUSolver::default();
        unsafe { (*s.0.common).ordering = ord };
        let t = Instant::now();
        let r = newton_pf_gn(ybus, sbus, v_init, npv, npq, Some(1e-8), Some(100), &mut s);
        println!("GN-LM ordering={ord}: total={:9.?} it={}",
            t.elapsed(), r.map(|(_, it)| it).unwrap_or(usize::MAX));
    }
}
