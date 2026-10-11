//! v1 行军语义验证：IEEE39 从 α=1 前进，必须拿到可行/无解区间。
//! Run: `cargo test --release --features ldl alphamax -- --nocapture`

use super::*;
use crate::basic::solver::{KLUSolver, LDLSolver};
use crate::lm::residual::fixtures::load_ieee39_mat;

#[cfg(feature = "ldl")]
#[test]
fn ieee39_march_brackets_alpha_max() {
    let mat = load_ieee39_mat();
    let p_shift = vec![0.0; mat.v_bus_init.len()];
    let opts = MarchOptions::default();
    let step = opts.step;

    // NR/DC 用 LU（Jacobian 非对称）；LMPF 用 QDLDL（增广 KKT 对称 quasidefinite）。
    let mut march = AlphaMarch::new(
        &mat.y_bus,
        mat.npv,
        mat.npq,
        mat.s_bus.clone(),
        mat.v_bus_init.clone(),
        &mat.to_perm,
        &p_shift,
        KLUSolver::default(),
        LDLSolver::default(),
    );
    let out = march.run(&mat.y_bus, &opts);

    for p in &out.points {
        println!("α={:.2}  {:?}", p.alpha, p.verdict);
    }
    println!(
        "bracket={:?}  nr_iters={} lm_iters={}",
        out.bracket, out.nr_iterations, out.lm_iterations
    );

    // 基态必须可行（dc 启动 + NR）。
    assert!(matches!(
        out.points[0].verdict,
        Verdict::Feasible { .. } | Verdict::Rescued { .. }
    ));
    // IEEE39 在 α≤10 内必须有鼻点：区间存在且宽度恰为一个步长。
    let (lo, hi) = out.bracket.expect("IEEE39 march did not bracket the fold");
    assert!((hi - lo - step).abs() < 1e-12, "bracket width != step");
    // 上端点必须持最小二乘证书，下端点可行。
    let last = out.points.last().unwrap();
    match last.verdict {
        Verdict::Infeasible { res_inf, .. } => {
            assert!(res_inf > opts.lm_tol, "certificate too weak: {res_inf:e}");
        }
        other => panic!("last point should be infeasible, got {other:?}"),
    }

    // 阶段二：二分收缩到 eps。
    let bi = BisectOptions::default();
    let eps = bi.eps;
    let res = march.bisect(&mat.y_bus, (lo, hi), out.v_at_lo.clone(), &opts, &bi);
    println!(
        "α_max ∈ ({:.6}, {:.6})  cert‖F‖∞={:.3e}  二分探点 {} 个  nr_iters={} lm_iters={}",
        res.lo, res.hi, res.hi_res_inf, res.probes.len(), res.nr_iterations, res.lm_iterations
    );
    assert!(res.hi - res.lo <= eps, "bracket not shrunk to eps");
    // 不变式：下端可行、上端持证书；宽度 ≈ eps（最后一次对半后 < eps）。
    assert!(res.hi_res_inf > opts.lm_tol);
    // 保底路径的探点数应≈log2(step/eps)，超 max_probes 即逻辑错误。
    let expect = (step / eps).log2().ceil() as usize;
    assert!(
        res.probes.len() <= expect + 1,
        "bisection used {} probes, expected ~{expect}",
        res.probes.len()
    );
}
