//! 使用完整CSC增广矩阵的Gauss–Newton LM潮流求解器。
//!
//! 状态步长为δ = [Δθ_active; Δ|V|_PQ]，残差为r = [ΔP_active; ΔQ_PQ]。
//! 每次试步求解：
//! ```text
//! [ μD  Jᵀ ] [δ] = [ 0]
//! [ J   −I ] [s]   [−r]
//! ```
//! D由阻尼尺度选项确定，默认是单位矩阵。每轮外迭代填充J、Jᵀ；
//! 同一轮的μ重试只更新对角和右端项，−I在构造时写入一次。
//!
//! `build`保留V4填J、转置和列拷贝流程；`build_operator`直接填最终矩阵。
//! 两者共用迭代流程和计时器。文件依次定义布局、填充工作区、求解器。
//! Ybus、指定功率和电压均按[PQ | PV | slack]的同一节点顺序传入。

use crate::basic::jacobian_operator::{JacobianBlock, JacobianOperator};
use crate::lm::step_control::{
    DampingWeights, LmOptions, TrialError, TrustRegion, polar_trial, predicted_reduction,
};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use std::time::Instant;

use super::kernels::fill_jt;
use super::pattern::KktPattern;
use super::residual::residual;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::basic::solver::Solve;

/// 完整增广矩阵[μD Jᵀ; J −I]的CSC结构。
pub struct GnFlatLayout {
    pub n_state: usize,
    /// `2·nnz + 2·n_state`.
    pub nnz_slim: usize,
    pub col_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
}

impl GnFlatLayout {
    pub fn build(pat: &KktPattern) -> Self {
        let cache = &pat.cache;
        let n = cache.n_active() + cache.n_pq();
        let nnz = pat.graph.nnz;
        let cs = &pat.graph.col_starts;

        let mut col_offsets = Vec::with_capacity(2 * n + 1);
        let mut row_indices = Vec::with_capacity(2 * nnz + 2 * n);

        // δ-columns: [μ diag (row c) | J col c (rows shifted by n)].
        for c in 0..n {
            col_offsets.push(c + cs[c]);
            row_indices.push(c);
            row_indices.extend(pat.graph.col_rows(c).iter().map(|row| row + n));
        }
        // s-columns: [Jᵀ segment | −I entry].
        for c in 0..n {
            col_offsets.push(n + nnz + cs[c] + c);
            row_indices.extend_from_slice(pat.graph.col_rows(c));
            row_indices.push(n + c);
        }
        col_offsets.push(2 * nnz + 2 * n);

        Self {
            n_state: n,
            nnz_slim: 2 * nnz + 2 * n,
            col_offsets,
            row_indices,
        }
    }

    /// Stamp the constant `−I` block (last entry of every s-column), once.
    pub fn stamp_neg_i(&self, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.nnz_slim);
        for c in 0..self.n_state {
            values[self.col_offsets[self.n_state + c + 1] - 1] = -1.0;
        }
    }
    /// J跳过列头的μ槽；Jᵀ位于后半组列，后面留有−I槽。
    fn jacobian_blocks(&self) -> (JacobianBlock<'_>, JacobianBlock<'_>) {
        (
            JacobianBlock {
                column_starts: &self.col_offsets,
                base: 0,
                prefix: 1,
            },
            JacobianBlock {
                column_starts: &self.col_offsets[self.n_state..],
                base: 0,
                prefix: 0,
            },
        )
    }

    /// 将原填充路径的两个紧凑块复制到最终CSC，保留μ和−I。
    fn copy_jacobians(&self, starts: &[usize], j: &[f64], jt: &[f64], values: &mut [f64]) {
        let n = self.n_state;
        let columns = &self.col_offsets;
        for c in 0..n {
            let count = columns[c + 1] - columns[c] - 1;
            values[columns[c] + 1..columns[c] + 1 + count]
                .copy_from_slice(&j[starts[c]..starts[c] + count]);
        }
        for c in 0..n {
            let count = columns[n + c + 1] - columns[n + c] - 1;
            values[columns[n + c]..columns[n + c] + count]
                .copy_from_slice(&jt[starts[c]..starts[c] + count]);
        }
    }
}

// 装配方式及其专属工作区；迭代控制不依赖装配方式。
enum Assembly {
    BlockCopy {
        j_block: Vec<f64>,
        jt_block: Vec<f64>,
        vnorm: Vec<Complex64>,
    },
    Direct {
        inv_vmag: Vec<f64>,
    },
}

impl Assembly {
    fn fill(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &[Complex64],
        power: &[Complex64],
        pattern: &KktPattern,
        layout: &GnFlatLayout,
        values: &mut [f64],
    ) {
        match self {
            Assembly::Direct { inv_vmag } => {
                for (inv, voltage) in inv_vmag.iter_mut().zip(v) {
                    *inv = 1.0 / voltage.norm();
                }
                let cache = &pattern.cache;
                let op = JacobianOperator {
                    ybus,
                    v,
                    inv_vmag,
                    scalc: power,
                    pq_ends: cache.pq_ends(),
                    active_ends: cache.active_ends(),
                    diag_ptrs: cache.diag_ptrs(),
                    mirror: cache.y_trans(),
                    npq: pattern.cache.n_pq(),
                    npv: pattern.cache.n_active() - pattern.cache.n_pq(),
                };
                let (j, jt) = layout.jacobian_blocks();
                op.fill::<false, false>(j, values); // CSC J
                op.fill::<false, true>(jt, values); // CSC Jᵀ
            }
            Assembly::BlockCopy {
                j_block,
                jt_block,
                vnorm,
            } => {
                let nb = ybus.ncols();
                for i in 0..nb {
                    let m = v[i].norm();
                    vnorm[i] = if m > 1e-12 {
                        v[i] / m
                    } else {
                        Complex64::new(1.0, 0.0)
                    };
                }
                let cache = &pattern.cache;
                let (npv, npq) = (
                    pattern.cache.n_active() - pattern.cache.n_pq(),
                    pattern.cache.n_pq(),
                );
                let cs = &pattern.graph.col_starts;
                fill_jacobian_v4::<false>(
                    ybus,
                    v,
                    vnorm,
                    power,
                    cs,
                    cache.pq_ends(),
                    cache.active_ends(),
                    cache.diag_ptrs(),
                    npv,
                    npq,
                    j_block,
                );
                fill_jt::<false>(ybus, pattern, j_block.as_ptr(), jt_block.as_mut_ptr());

                layout.copy_jacobians(cs, j_block, jt_block, values);
            }
        }
    }
}

/// 完整增广GN-LM求解器。工作数组只在构造时分配，试步不重建布局。
pub struct GnDriver {
    pub pat: KktPattern,
    pub gn: GnFlatLayout,
    /// 指定复功率注入，与Ybus使用相同节点顺序。
    pub sbus: Vec<Complex64>,
    assembly: Assembly,
    values: Vec<f64>,
    // 当前点与试探点分开保存；接受试步后才更新调用者的电压。
    currents: Vec<Complex64>,
    power: Vec<Complex64>,
    mismatch: Vec<f64>,
    trial_mismatch: Vec<f64>,
    gradient: Vec<f64>, // Jᵀr
    rhs: Vec<f64>,      // 求解前为[0; −r]，求解后为[δ; s]
    trial_voltage: Vec<Complex64>,
    n_active: usize,
    npq: usize,
    n_state: usize,
    /// J、Jᵀ填充及所需工作向量准备的累计纳秒数。
    pub prof_fill_ns: u64,
    /// μ对角及右端准备的累计时间。
    pub prof_mu_ns: u64,
    /// solver.solve调用的累计纳秒数。
    pub prof_solve_ns: u64,
    /// 包含被拒绝试步和失败重试的线性求解次数。
    pub n_solves: u64,
}

/// 求解结果；converged表示潮流残差无穷范数达到tol。
pub struct GnResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl GnDriver {
    /// 正确性测试读取最终矩阵。
    #[cfg(test)]
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    /// 原填充路径：V4填J，生成Jᵀ，再复制到完整增广矩阵。
    pub fn build(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let assembly = Assembly::BlockCopy {
            j_block: vec![0.0; pat.graph.nnz],
            jt_block: vec![0.0; pat.graph.nnz],
            vnorm: vec![Complex64::new(1.0, 0.0); ybus.ncols()],
        };
        Self::from_pattern(ybus.ncols(), pat, sbus, assembly)
    }

    /// 直接使用operator填充最终CSC矩阵中的J和Jᵀ。
    pub fn build_operator(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let assembly = Assembly::Direct {
            inv_vmag: vec![0.0; ybus.ncols()],
        };
        Self::from_pattern(ybus.ncols(), pat, sbus, assembly)
    }

    fn from_pattern(nb: usize, pat: KktPattern, sbus: Vec<Complex64>, assembly: Assembly) -> Self {
        let n_active = pat.cache.n_active();
        let npq = pat.cache.n_pq();
        let gn = GnFlatLayout::build(&pat);
        let n_state = gn.n_state;
        let mut values = vec![0.0; gn.nnz_slim];
        gn.stamp_neg_i(&mut values);
        Self {
            pat,
            gn,
            sbus,
            assembly,
            values,
            currents: vec![Complex64::new(0.0, 0.0); nb],
            power: vec![Complex64::new(0.0, 0.0); nb],
            mismatch: vec![0.0; n_state],
            trial_mismatch: vec![0.0; n_state],
            gradient: vec![0.0; n_state],
            rhs: vec![0.0; 2 * n_state],
            trial_voltage: vec![Complex64::new(0.0, 0.0); nb],
            n_active,
            npq,
            n_state,
            prof_fill_ns: 0,
            prof_mu_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    /// 清零累计计时和线性求解次数；不修改矩阵和工作数组。
    pub fn reset_prof(&mut self) {
        self.prof_fill_ns = 0;
        self.prof_mu_ns = 0;
        self.prof_solve_ns = 0;
        self.n_solves = 0;
    }

    /// 填充 J 和 Jᵀ：原路径先填紧凑块再拷贝，直接算子写入最终 CSC。
    /// 两者复用残差计算得到的 power，不修改 μ 和 −I。
    fn fill(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t_prof = Instant::now();
        self.assembly
            .fill(ybus, v, &self.power, &self.pat, &self.gn, &mut self.values);
        self.prof_fill_ns += t_prof.elapsed().as_nanos() as u64;
    }

    /// 更新μD和右端项，再复用外部solver求解；两个计时区间保持独立。
    fn solve_linear_system<S: Solve>(
        &mut self,
        solver: &mut S,
        v: &[Complex64],
        mu: f64,
        damping: &DampingWeights,
    ) -> bool {
        let n = self.n_state;
        let t_mu = Instant::now();
        // μ diagonal slot = head of every δ-column: set absolute μ.
        {
            let gp = &self.gn.col_offsets;
            for c in 0..n {
                self.values[gp[c]] = mu * damping.weight(v, c, self.n_active);
            }
        }

        self.rhs[..n].fill(0.0);
        for i in 0..n {
            self.rhs[n + i] = -self.mismatch[i];
        }
        self.prof_mu_ns += t_mu.elapsed().as_nanos() as u64;
        let t_solve = Instant::now();
        let solve_ok = solver
            .solve(
                &mut self.gn.col_offsets,
                &mut self.gn.row_indices,
                &mut self.values,
                &mut self.rhs,
                2 * n,
            )
            .is_ok();
        self.prof_solve_ns += t_solve.elapsed().as_nanos() as u64;
        self.n_solves += 1;
        solve_ok
    }

    /// 从最终CSC的J块计算梯度g = Jᵀr；跳过每列开头的μ槽。
    fn update_gradient(&mut self) {
        let (n, gp, ri) = (self.n_state, &self.gn.col_offsets, &self.gn.row_indices);
        for c in 0..n {
            let mut acc = 0.0;
            for p in gp[c] + 1..gp[c + 1] {
                acc += self.values[p] * self.mismatch[ri[p] - n];
            }
            self.gradient[c] = acc;
        }
    }

    /// 使用默认LM参数；v原地更新为最后一个接受的电压点。
    pub fn solve_gn<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnResult {
        self.solve_gn_with_options(ybus, solver, v, tol, maxit, &LmOptions::default())
    }

    /// tol检查潮流残差无穷范数，maxit限制外层迭代次数。
    /// 成功或失败均保留v中的最后接受点；所有试步复用传入的solver。
    /// options无效时panic；处理外部配置的调用者可先调用validate。
    #[allow(clippy::too_many_arguments)]
    pub fn solve_gn_with_options<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> GnResult {
        options.validate().expect("invalid LM options");
        let damping = options.damping_metric.prepare(ybus, self.n_active);
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let mut mu = options.initial_mu;
        let mut region = TrustRegion::new(options.trust_region.as_ref());
        for iteration in 0..maxit {
            // 当前点：残差与注入功率只计算一次，填充复用power。
            let (res_inf, cost) = super::residual::residual_with_power(
                ybus,
                &self.sbus,
                &mut self.currents,
                &mut self.power,
                self.n_active,
                self.npq,
                v,
                &mut self.mismatch,
            );
            if res_inf < tol {
                return GnResult {
                    iterations: iteration,
                    converged: true,
                    res_inf,
                };
            }
            self.fill(ybus, v);
            self.update_gradient();

            // 本轮J保持不变；拒绝试步后更新μ并重解线性方程。
            let mut accepted = false;
            for _ in 0..options.max_trials {
                let solve_ok = self.solve_linear_system(solver, v, mu, &damping);
                let delta = &self.rhs[..n];
                let finite = solve_ok && delta.iter().all(|x| x.is_finite());
                if !finite {
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        break;
                    }
                    continue;
                }

                let step_norm_squared = damping.step_norm_squared(v, delta, self.n_active);
                if !region.allows(step_norm_squared) {
                    if debug {
                        eprintln!(
                            "it={iteration} tryμ={mu:.3e} rejected=TrustRadius step={:.3e} radius={:.3e}",
                            step_norm_squared.sqrt(),
                            region.radius
                        );
                    }
                    if !options.increase_mu(&mut mu, options.mu_increase) {
                        break;
                    }
                    continue;
                }

                if let Err(reason) = polar_trial(
                    v,
                    delta,
                    self.n_active,
                    self.npq,
                    options.reject_nonpositive_voltage,
                    &mut self.trial_voltage,
                ) {
                    if debug {
                        eprintln!("it={iteration} tryμ={mu:.3e} rejected={reason:?}");
                    }
                    region.reject();
                    let factor = match reason {
                        TrialError::NonFinite => options.failed_step_increase,
                        TrialError::NonPositiveMagnitude { .. } => options.mu_increase,
                    };
                    if !options.increase_mu(&mut mu, factor) {
                        break;
                    }
                    continue;
                }

                let (_, trial_cost) = residual(
                    ybus,
                    &self.sbus,
                    &mut self.currents,
                    self.n_active,
                    self.npq,
                    &self.trial_voltage,
                    &mut self.trial_mismatch,
                );
                let prediction = predicted_reduction(&self.gradient, delta, mu, step_norm_squared);
                let rho = if prediction.is_finite() && prediction > 0.0 && trial_cost.is_finite() {
                    (cost - trial_cost) / prediction
                } else {
                    -1.0
                };
                if debug {
                    eprintln!(
                        "it={iteration} tryμ={mu:.3e} res={res_inf:.3e} f={cost:.4e} f_new={trial_cost:.4e} pred={prediction:.4e} ρ={rho:.4}"
                    );
                }
                if rho.is_finite() && rho > options.acceptance_threshold {
                    region.accept(rho, step_norm_squared, options.good_step_threshold);
                    v.copy_from_slice(&self.trial_voltage);
                    mu = options.accepted_mu(mu, rho);
                    accepted = true;
                    break;
                }
                region.reject();
                if !options.increase_mu(&mut mu, options.mu_increase) {
                    break;
                }
            }
            if !accepted {
                return GnResult {
                    iterations: iteration,
                    converged: false,
                    res_inf,
                };
            }
        }
        let (n_active, npq) = (self.n_active, self.npq);
        let (res_inf, _) = residual(
            ybus,
            &self.sbus,
            &mut self.currents,
            n_active,
            npq,
            v,
            &mut self.mismatch,
        );
        GnResult {
            iterations: maxit,
            converged: res_inf < tol,
            res_inf,
        }
    }
}

/// Classical GN-LM power flow with the same contract as
/// [`crate::basic::newton_pf`] — the ECS-plugin entry point. Slim-layout
/// counterpart of the exact-LM driver's `newton_pf_lm`.
#[allow(clippy::too_many_arguments)]
pub fn newton_pf_gn<Solver: Solve>(
    ybus: &CscMatrix<Complex64>,
    sbus: &DVector<Complex64>,
    v_init: &DVector<Complex64>,
    npv: usize,
    npq: usize,
    tolerance: Option<f64>,
    max_iter: Option<usize>,
    solver: &mut Solver,
) -> Result<(DVector<Complex64>, usize), (String, DVector<Complex64>, usize)> {
    let tol = tolerance.unwrap_or(1e-6);
    let maxit = max_iter.unwrap_or(100);
    let mut driver = GnDriver::build(ybus, npv, npq, sbus.iter().copied().collect());
    let mut v: Vec<Complex64> = v_init.iter().copied().collect();
    let res = driver.solve_gn(ybus, solver, &mut v, tol, maxit);
    let dv = DVector::from_vec(v);
    if res.converged {
        Ok((dv, res.iterations))
    } else {
        Err((
            format!("GN-LM did not converge (res_inf = {:.3e})", res.res_inf),
            dv,
            res.iterations,
        ))
    }
}

#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use super::*;
    use crate::basic::solver::KLUSolver;
    use crate::lm::residual::fixtures::load_ieee39_mat;

    /// IEEE39 收敛性（与 fat-GN 同数学，同迭代数）。
    #[test]
    fn slim_gn_ieee39() {
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();
        let mut driver = GnDriver::build(ybus, npv, npq, sbus);
        let mut solver = KLUSolver::default();
        let mut v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let res = driver.solve_gn(ybus, &mut solver, &mut v, 1e-8, 100);
        println!(
            "IEEE39 slim-GN: converged={} it={} res={:.2e} | nnz slim={} fat={}",
            res.converged,
            res.iterations,
            res.res_inf,
            driver.gn.nnz_slim,
            crate::lm::flat::FlatLayout::build(&driver.pat).nnz_flat
        );
        assert!(res.converged);
    }

    /// 非病态标准算例：NR / GN-LM / exact-LM 三家在严格容差（1e-12）下
    /// 解的逐点一致性——验证 LM 两侧与生产 NR 算的是同一个东西。
    #[test]
    fn three_way_matches_nr_tight() {
        use crate::basic::newtonpf::newton_pf;
        use crate::lm::exact::driver::newton_pf_lm;
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus = DVector::from_vec(mat.s_bus.iter().copied().collect::<Vec<_>>());
        let v_init = DVector::from_vec(mat.v_bus_init.iter().copied().collect::<Vec<_>>());

        let mut s_nr = KLUSolver::default();
        let (v_nr, it_nr) = newton_pf(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_nr,
            None,
        )
        .expect("NR should converge");
        let mut s_gn = KLUSolver::default();
        let (v_gn, it_gn) = newton_pf_gn(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_gn,
        )
        .expect("GN-LM should converge");
        let mut s_lm = KLUSolver::default();
        let (v_lm, it_lm) = newton_pf_lm(
            ybus,
            &sbus,
            &v_init,
            npv,
            npq,
            Some(1e-12),
            Some(100),
            &mut s_lm,
        )
        .expect("exact-LM should converge");

        let diff = |a: &DVector<Complex64>, b: &DVector<Complex64>| {
            a.iter()
                .zip(b.iter())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
        };
        println!("严格容差 1e-12: NR it={it_nr} | GN-LM it={it_gn} | exact-LM it={it_lm}");
        println!(
            "max|ΔV|: GN vs NR = {:.3e} | exact vs NR = {:.3e} | exact vs GN = {:.3e}",
            diff(&v_gn, &v_nr),
            diff(&v_lm, &v_nr),
            diff(&v_lm, &v_gn)
        );
        assert!(
            diff(&v_gn, &v_nr) < 1e-9,
            "GN-LM and NR disagree at tight tolerance"
        );
        assert!(
            diff(&v_lm, &v_nr) < 1e-9,
            "exact-LM and NR disagree at tight tolerance"
        );
    }
}
