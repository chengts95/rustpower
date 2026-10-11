//! 连续潮流（Continuation Power Flow, CPF）——预测-校正沿 PV 曲线追踪到鼻点。
//!
//! # 方法来源
//!
//! 切向量预测 + perpendicular-intersection 校正，即 MATPOWER `runcpf` 的默认
//! 参数化方案：
//!
//! - V. Ajjarapu, C. Christy, "The continuation power flow: a tool for steady
//!   state voltage stability analysis", IEEE Trans. Power Systems, 7(1), 1992.
//! - H.-D. Chiang et al., "CPFLOW: a practical tool for tracing power system
//!   steady-state stationary behavior due to load and generation variations",
//!   IEEE Trans. Power Systems, 10(2), 1995.
//!
//! # 数学
//!
//! 负荷参数 λ 沿固定注入方向缩放：F(x, λ) = s_calc(x) − λ·d = 0，
//! 其中 d 是基准注入 S_base 的 reduced 形式（P: 全体 active 母线，Q: 仅 PQ），
//! ∂F/∂λ = −d 为常量。状态 x = [θ(全体 active); |V|(仅 PQ)]，与本仓库
//! newton_pf / fill_jacobian_v4 的 [PQ|PV|slack] 约简约定完全一致，因此
//! Jacobian 直接复用 v4 填充路径，不需要任何结构改动。
//!
//! - 预测：z = J⁻¹d 即 dλ=1 归一的切方向，t̂ = [z; 1]/√(‖z‖²+1)，步进 σ·t̂。
//!   定向用与上一切向量点积保号；**鼻点 = 定向后 t̂_λ 由正转负**。
//! - 校正（perpendicular intersection）：在过预测点、法向为 t̂ 的超平面上
//!   解 F=0。每次校正迭代 = 1 次分解（J·y = F）+ 1 次回代（J·z = d，经
//!   `Solve::resolve` 复用同一分解）。
//!
//! # 为什么是这个模块而不是行进-括弧 α_max
//!
//! 行进-括弧求解器已在 commit bef91b2 删除：鼻点附近 NR 吸引域收缩
//! （不收敛≠无解）、LMPF 残差存在 ~5e-4 停滞地板（不同 α 给出相同"证书"
//! 残差），裁决式括弧在鼻点附近结构性退化。连续法沿解流形行走，
//! 不做"有解/无解"二元裁决，是结构上的正路。
//!
//! # 记账纪律
//!
//! 与 lm 模块一致：成本按 **分解次数（factorizations）与回代次数（resolves）**
//! 记，不按外层步数。每次切向量求解也是一次分解，计入台账。

use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use std::time::{Duration, Instant};

use crate::basic::jacobian_cache::JacobianCache;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::newtonpf::{csc_matvec_and_scalc, fill_f_from_scalc, newton_pf};
use crate::basic::solver::Solve;

/// CPF 运行参数。
pub struct CpfOptions {
    /// 校正收敛容差（‖F‖∞，与 newton_pf 同标度）。
    pub tol: f64,
    /// 初始步长 σ。
    pub step0: f64,
    /// 步长上限。
    pub step_max: f64,
    /// 步长下限；跌破即终止（报告未过鼻点）。
    pub step_min: f64,
    /// 校正成功后的步长放大倍数。
    pub step_grow: f64,
    /// 校正失败后的步长收缩倍数。
    pub step_shrink: f64,
    /// 单个预测点的最大校正迭代数。
    pub max_corr_iter: usize,
    /// 最大接受点数（含初始点）。
    pub max_points: usize,
}

impl Default for CpfOptions {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            step0: 0.2,
            step_max: 1.0,
            step_min: 1e-3,
            step_grow: 1.3,
            step_shrink: 0.5,
            max_corr_iter: 10,
            max_points: 300,
        }
    }
}

/// 曲线上的一个接受点。
pub struct CpfPoint {
    pub lambda: f64,
    pub v: DVector<Complex64>,
    /// 该点校正迭代数（每次 = 1 分解 + 1 回代）。
    pub corr_iters: usize,
}

/// 鼻点报告。
pub struct NoseReport {
    /// 鼻点 λ（三点二次拟合顶点；退化时为最大 λ 接受点）。
    pub lambda_nose: f64,
    pub v_nose: DVector<Complex64>,
    /// 是否真正越过了鼻点（定向切向量 t̂_λ 转负）。
    pub passed_nose: bool,
    /// 终止原因（人读）。
    pub stop_reason: &'static str,
    pub points: Vec<CpfPoint>,
    /// 分解总次数（初始 NR + 每点切向量 + 每次校正迭代）。
    pub factorizations: usize,
    /// 回代总次数（resolve 复用分解；后端不支持时退化为分解）。
    pub resolves: usize,
    pub wall: Duration,
}

/// CPF 求解器。持有 Jacobian 符号结构、全部数值缓冲与线性求解器；
/// `ybus` 以参数借用（结构在追踪过程中不变，一套符号用到底）。
pub struct CpfSolver<S: Solve> {
    npv: usize,
    npq: usize,
    n_active: usize,
    n_state: usize,
    /// 基准注入（λ=1，已按 [PQ|PV|slack] 置换）。
    s_base: DVector<Complex64>,
    /// 启动电压（PV 幅值/平衡节点即取自此，与 newton_pf 启动一致）。
    v_spec: DVector<Complex64>,
    /// reduced 注入方向：d[i]=Re(S_base[i]) (i<n_active)，d[n_active+j]=Im(S_base[j]) (j<npq)。
    d: Vec<f64>,
    pat: JacobianCache,
    j_values: Vec<f64>,
    // ── 数值缓冲 ──
    v_cur: DVector<Complex64>,
    v_m: DVector<f64>,
    v_a: DVector<f64>,
    v_norm: DVector<Complex64>,
    ibus: DVector<Complex64>,
    s_calc: DVector<Complex64>,
    s_lam: DVector<Complex64>,
    /// 校正右端 F；solve 后即 y = J⁻¹F。
    fbuf: Vec<f64>,
    /// 第二右端 z = J⁻¹d。
    zbuf: Vec<f64>,
    solver: S,
}

impl<S: Solve> CpfSolver<S> {
    pub fn new(
        ybus: &CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
        s_base: DVector<Complex64>,
        v_spec: DVector<Complex64>,
        solver: S,
    ) -> Self {
        let n_active = npv + npq;
        let n_state = npv + 2 * npq;
        let n_bus = s_base.len();
        let pat = JacobianCache::build_from_permuted(
            ybus.col_offsets(),
            ybus.row_indices(),
            npv,
            npq,
        );
        let j_values = vec![0.0; pat.j_row_indices.len()];
        let mut d = vec![0.0; n_state];
        for i in 0..n_active {
            d[i] = s_base[i].re;
        }
        for j in 0..npq {
            d[n_active + j] = s_base[j].im;
        }
        Self {
            npv,
            npq,
            n_active,
            n_state,
            s_base,
            v_spec,
            d,
            pat,
            j_values,
            v_cur: DVector::zeros(n_bus),
            v_m: DVector::zeros(n_bus),
            v_a: DVector::zeros(n_bus),
            v_norm: DVector::zeros(n_bus),
            ibus: DVector::zeros(n_bus),
            s_calc: DVector::zeros(n_bus),
            s_lam: DVector::zeros(n_bus),
            fbuf: vec![0.0; n_state],
            zbuf: vec![0.0; n_state],
            solver,
        }
    }

    /// 由当前 v_m/v_a 重建复电压并刷新 v_norm。
    fn refresh_v(&mut self) {
        for i in 0..self.v_cur.len() {
            self.v_cur[i] = Complex64::from_polar(self.v_m[i], self.v_a[i]);
            self.v_norm[i] = Complex64::from_polar(1.0, self.v_a[i]);
        }
    }

    /// 当前状态向量 x = [θ(0..n_active); |V|(0..npq)]。
    fn state_x(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.n_state];
        x[..self.n_active].copy_from_slice(&self.v_a.as_slice()[..self.n_active]);
        x[self.n_active..].copy_from_slice(&self.v_m.as_slice()[..self.npq]);
        x
    }

    /// 在当前点填充 Jacobian 数值。
    fn fill_jacobian(&mut self, ybus: &CscMatrix<Complex64>) {
        fill_jacobian_v4::<false>(
            ybus,
            self.v_cur.as_slice(),
            self.v_norm.as_slice(),
            self.s_calc.as_slice(),
            &self.pat.j_col_ptrs,
            &self.pat.pq_ends,
            &self.pat.active_ends,
            &self.pat.diag_ptrs,
            self.npv,
            self.npq,
            &mut self.j_values,
        );
    }

    /// λ 缩放注入下的 mismatch F = s_calc − λ·S_base，返回 ‖F‖∞。
    fn mismatch(&mut self, ybus: &CscMatrix<Complex64>, lambda: f64) -> f64 {
        csc_matvec_and_scalc(
            ybus.col_offsets(),
            ybus.row_indices(),
            ybus.values(),
            self.v_cur.as_slice(),
            self.ibus.as_mut_slice(),
            self.s_calc.as_mut_slice(),
        );
        for i in 0..self.s_lam.len() {
            self.s_lam[i] = self.s_base[i] * lambda;
        }
        fill_f_from_scalc::<false>(
            self.s_calc.as_slice(),
            self.s_lam.as_slice(),
            self.npq,
            self.n_active,
            &mut self.fbuf,
        )
    }

    /// 当前点的定向切向量。返回 ([t̂x; t̂λ])。每次调用 = 1 次分解。
    fn tangent(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        factorizations: &mut usize,
    ) -> Result<Vec<f64>, String> {
        self.fill_jacobian(ybus);
        self.zbuf.copy_from_slice(&self.d);
        self.solver
            .solve(
                &mut self.pat.j_col_ptrs,
                &mut self.pat.j_row_indices,
                &mut self.j_values,
                &mut self.zbuf,
                self.n_state,
            )
            .map_err(|e| format!("tangent solve failed: {e}"))?;
        *factorizations += 1;
        let norm = (self.zbuf.iter().map(|e| e * e).sum::<f64>() + 1.0).sqrt();
        let mut t = Vec::with_capacity(self.n_state + 1);
        t.extend(self.zbuf.iter().map(|e| e / norm));
        t.push(1.0 / norm);
        Ok(t)
    }

    /// perpendicular-intersection 校正。从 (v_m/v_a 隐含状态, lambda) 出发，
    /// 约束在过 (x_pred, lambda_pred)、法向 t̂ 的超平面上解 F = 0。
    /// 返回校正迭代数；失败（不收/发散/λ<0）返回 Err。
    #[allow(clippy::too_many_arguments)]
    fn corrector(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        t: &[f64],
        x_pred: &[f64],
        lambda_pred: f64,
        lambda: &mut f64,
        opts: &CpfOptions,
        factorizations: &mut usize,
        resolves: &mut usize,
    ) -> Result<usize, ()> {
        let n_act = self.n_active;
        let n_st = self.n_state;
        let (tx, tl) = (&t[..n_st], t[n_st]);

        for it in 0..opts.max_corr_iter {
            let norm = self.mismatch(ybus, *lambda);
            if !norm.is_finite() || *lambda < 0.0 {
                return Err(());
            }
            if norm < opts.tol {
                return Ok(it);
            }

            // 平面残差 p = t̂xᵀ(x − x_pred) + t̂λ(λ − λ_pred)
            let x = self.state_x();
            let mut p = tl * (*lambda - lambda_pred);
            for i in 0..n_st {
                p += tx[i] * (x[i] - x_pred[i]);
            }

            self.fill_jacobian(ybus);
            if self
                .solver
                .solve(
                    &mut self.pat.j_col_ptrs,
                    &mut self.pat.j_row_indices,
                    &mut self.j_values,
                    &mut self.fbuf,
                    n_st,
                )
                .is_err()
            {
                return Err(());
            }
            *factorizations += 1;

            // z = J⁻¹d：优先 resolve 复用分解；后端不支持则重填重分解。
            self.zbuf.copy_from_slice(&self.d);
            if self.solver.resolve(&mut self.zbuf).is_ok() {
                *resolves += 1;
            } else {
                self.fill_jacobian(ybus);
                if self
                    .solver
                    .solve(
                        &mut self.pat.j_col_ptrs,
                        &mut self.pat.j_row_indices,
                        &mut self.j_values,
                        &mut self.zbuf,
                        n_st,
                    )
                    .is_err()
                {
                    return Err(());
                }
                *factorizations += 1;
            }

            // 增广牛顿步：δλ = (p − t̂xᵀy)/(t̂λ + t̂xᵀz)，x ← x − (y + δλ z)，λ ← λ − δλ
            let (y, z) = (&self.fbuf, &self.zbuf);
            let mut tx_y = 0.0;
            let mut tx_z = 0.0;
            for i in 0..n_st {
                tx_y += tx[i] * y[i];
                tx_z += tx[i] * z[i];
            }
            let denom = tl + tx_z;
            if denom.abs() < 1e-14 {
                return Err(());
            }
            let d_lambda = (p - tx_y) / denom;
            for i in 0..n_act {
                self.v_a[i] -= y[i] + d_lambda * z[i];
            }
            for j in 0..self.npq {
                self.v_m[j] -= y[n_act + j] + d_lambda * z[n_act + j];
            }
            *lambda -= d_lambda;
            self.refresh_v();
        }
        // 最后一次更新后再判一次
        let norm = self.mismatch(ybus, *lambda);
        if norm.is_finite() && norm < opts.tol && *lambda >= 0.0 {
            Ok(opts.max_corr_iter)
        } else {
            Err(())
        }
    }

    /// 追踪鼻点。λ 从 1 起步（先做一次标准 NR 拿基准解），沿固定方向加压，
    /// 定向切向量 t̂_λ 转负即判定过鼻，随后用最后三个接受点做二次拟合取顶点。
    pub fn run(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        opts: &CpfOptions,
    ) -> Result<NoseReport, String> {
        let t0 = Instant::now();
        let mut factorizations = 0usize;
        let mut resolves = 0usize;

        // ── 基准点 λ=1：标准 NR（与生产路径同一入口）──
        let (v0, it0) = newton_pf(
            ybus,
            &self.s_base,
            &self.v_spec,
            self.npv,
            self.npq,
            Some(opts.tol),
            None,
            &mut self.solver,
            None,
        )
        .map_err(|(msg, _, it)| {
            format!("base point λ=1 NR failed after {it} iters: {msg}")
        })?;
        factorizations += it0;

        self.v_cur = v0.clone();
        for i in 0..v0.len() {
            self.v_m[i] = v0[i].norm();
            self.v_a[i] = v0[i].arg();
        }
        self.refresh_v();
        // NR 出口的 s_calc 是旧缓存（用了独立缓冲），tangent 前必须刷新。
        let _ = self.mismatch(ybus, 1.0);

        let mut points = vec![CpfPoint {
            lambda: 1.0,
            v: v0,
            corr_iters: it0,
        }];
        let mut lambda = 1.0;
        let mut tangent = self.tangent(ybus, &mut factorizations)?;
        let mut sigma = opts.step0;
        let mut passed_nose = false;
        let mut stop_reason: &'static str = "max_points reached";

        loop {
            if points.len() >= opts.max_points {
                break;
            }
            // ── 预测 ──
            let x = self.state_x();
            let n_st = self.n_state;
            let mut x_pred = x.clone();
            for i in 0..n_st {
                x_pred[i] += sigma * tangent[i];
            }
            let lambda_pred = lambda + sigma * tangent[n_st];

            // 预测点写入状态缓冲
            for i in 0..self.n_active {
                self.v_a[i] = x_pred[i];
            }
            for j in 0..self.npq {
                self.v_m[j] = x_pred[self.n_active + j];
            }
            self.refresh_v();
            let mut lambda_corr = lambda_pred;

            match self.corrector(
                ybus,
                &tangent,
                &x_pred,
                lambda_pred,
                &mut lambda_corr,
                opts,
                &mut factorizations,
                &mut resolves,
            ) {
                Ok(iters) => {
                    points.push(CpfPoint {
                        lambda: lambda_corr,
                        v: self.v_cur.clone(),
                        corr_iters: iters,
                    });
                    lambda = lambda_corr;
                    sigma = (sigma * opts.step_grow).min(opts.step_max);

                    // 接受点处的新鲜切向量，定向保连续性
                    let mut t_new = self.tangent(ybus, &mut factorizations)?;
                    let dot: f64 = tangent.iter().zip(t_new.iter()).map(|(a, b)| a * b).sum();
                    if dot < 0.0 {
                        for e in t_new.iter_mut() {
                            *e = -*e;
                        }
                    }
                    if t_new[self.n_state] < 0.0 {
                        passed_nose = true;
                        stop_reason = "tangent t_lambda turned negative (nose passed)";
                        tangent = t_new;
                        break;
                    }
                    tangent = t_new;
                }
                Err(()) => {
                    // 校正失败：退回上一接受点，缩步重试
                    sigma *= opts.step_shrink;
                    if sigma < opts.step_min {
                        stop_reason = "step size below floor";
                        break;
                    }
                    // 状态恢复到上一接受点
                    let last = points.last().unwrap();
                    self.v_cur = last.v.clone();
                    for i in 0..last.v.len() {
                        self.v_m[i] = last.v[i].norm();
                        self.v_a[i] = last.v[i].arg();
                    }
                    self.refresh_v();
                    lambda = last.lambda;
                }
            }
        }

        let (lambda_nose, v_nose) = refine_nose(&points);
        Ok(NoseReport {
            lambda_nose,
            v_nose,
            passed_nose,
            stop_reason,
            points,
            factorizations,
            resolves,
            wall: t0.elapsed(),
        })
    }
}

/// 鼻点 refine：最后三个接受点按累计弦长 s 拟合 λ(s) 二次曲线取顶点；
/// 顶点越界或开口错误时退化为最大 λ 点。V 用同样的三点二次插值。
fn refine_nose(points: &[CpfPoint]) -> (f64, DVector<Complex64>) {
    let fallback = |points: &[CpfPoint]| {
        let p = points
            .iter()
            .max_by(|a, b| a.lambda.partial_cmp(&b.lambda).unwrap())
            .unwrap();
        (p.lambda, p.v.clone())
    };
    if points.len() < 3 {
        return fallback(points);
    }
    let n = points.len();
    let (p0, p1, p2) = (&points[n - 3], &points[n - 2], &points[n - 1]);
    let dist = |a: &CpfPoint, b: &CpfPoint| -> f64 {
        let mut s = (b.lambda - a.lambda).powi(2);
        for i in 0..a.v.len() {
            s += (b.v[i].norm() - a.v[i].norm()).powi(2)
                + (b.v[i].arg() - a.v[i].arg()).powi(2);
        }
        s.sqrt()
    };
    let s1 = dist(p0, p1);
    let s2 = s1 + dist(p1, p2);
    if s1 < 1e-12 || s2 - s1 < 1e-12 {
        return fallback(points);
    }
    // λ(s) = a s² + b s + c，过 (0,λ0) (s1,λ1) (s2,λ2)
    let (l0, l1, l2) = (p0.lambda, p1.lambda, p2.lambda);
    let a = ((l2 - l0) / s2 - (l1 - l0) / s1) / (s2 - s1);
    let b = (l1 - l0) / s1 - a * s1;
    if a >= 0.0 {
        return fallback(points); // 开口向上没有极大值
    }
    let s_star = -b / (2.0 * a);
    if !(0.0..=s2).contains(&s_star) {
        return fallback(points);
    }
    let lambda_nose = a * s_star * s_star + b * s_star + l0;
    // 同一参数化下逐分量插值 V
    let n_bus = p0.v.len();
    let mut v_nose = p0.v.clone();
    for i in 0..n_bus {
        let (v0, v1, v2) = (p0.v[i], p1.v[i], p2.v[i]);
        let av = ((v2 - v0) / s2 - (v1 - v0) / s1) / (s2 - s1);
        let bv = (v1 - v0) / s1 - av * s1;
        v_nose[i] = av * s_star * s_star + bv * s_star + v0;
    }
    (lambda_nose, v_nose)
}

#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use super::*;
    use crate::basic::solver::KLUSolver;
    use crate::lm::residual::fixtures::load_ieee39_mat;

    /// IEEE39 鼻点硬锁：真值 λ_nose = 2.1357（alphamax 线行军+阶梯+0.618
    /// 三方对账测得，见 commit 4607d81 及删除说明 bef91b2）。
    /// 硬锁区间 (2.13, 2.14)，防任何"路径全错但测试照过"的集体造假。
    #[test]
    fn ieee39_nose_hard_lock() {
        let mat = load_ieee39_mat();
        let solver = KLUSolver::default();
        let mut cpf = CpfSolver::new(
            &mat.y_bus,
            mat.npv,
            mat.npq,
            mat.s_bus.clone(),
            mat.v_bus_init.clone(),
            solver,
        );
        let rep = cpf.run(&mat.y_bus, &CpfOptions::default()).unwrap();
        println!(
            "IEEE39 CPF: λ_nose={:.6} passed_nose={} stop={}",
            rep.lambda_nose, rep.passed_nose, rep.stop_reason
        );
        println!(
            "  points={} factorizations={} resolves={} wall={:?}",
            rep.points.len(),
            rep.factorizations,
            rep.resolves,
            rep.wall
        );
        assert!(
            rep.passed_nose,
            "CPF did not pass the nose: {}",
            rep.stop_reason
        );
        assert!(
            rep.lambda_nose > 2.13 && rep.lambda_nose < 2.14,
            "nose λ={:.6} outside hard lock (2.13, 2.14)",
            rep.lambda_nose
        );
    }
}
