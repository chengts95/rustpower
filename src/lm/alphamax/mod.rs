//! Alpha-max（最大负荷点）求解系统。
//!
//! 方法骨架对照 Gómez-Quiles, Gómez-Expósito, Vargas, "Computation of
//! Maximum Loading Points via the Factored Load Flow"：前向行军 + 区间收缩。
//! 我们的不同之处在于无解裁决器：他们用分解潮流（FLF）越过鼻点后的复数解
//! 做判决；我们用 LMPF 的最小二乘证书（min ‖F‖ > tol）做判决——定量、
//! 无启发式阈值，且 NR 吸引域失败时 LMPF 可能直接把解救回来。
//!
//! v1（本模块）：阶段一，前向固定步进行军。
//!   - dc 启动（B′ 与 α 无关，模型只建一次，求解器对象复用）；
//!   - 每个探测点走种子阶梯：暖种子 NR → dc 重播种 NR → LMPF 终审；
//!   - 暖种子只取自最后一个可行解，从未验证的外推点不接力；
//!   - 输出可行/无解区间 [α_lo, α_hi]，供阶段二（二分法）收缩。
//!
//! 阶段二（二分）与割线加速实验尚未实现，见模块底部 TODO。

use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use crate::basic::dcpf::{DcpfModel, dcpf_initial_v};
use crate::basic::newtonpf::{NewtonCache, newton_pf};
use crate::basic::solver::{DefaultSolver, Solve};
use crate::lm::gn_triu::GnTriuDriver;
use crate::lm::step_control::{LmOptions, TrustRegionOptions};

/// 行军参数。
#[derive(Clone)]
pub struct MarchOptions {
    /// 起始加载因子（基态 α=1）。
    pub alpha0: f64,
    /// 固定步长（文献取 0.5，已被证明是稳妥的默认）。
    pub step: f64,
    /// 行军上限保险丝：到此仍未判出无解则放弃并如实报告。
    pub alpha_cap: f64,
    pub nr_tol: f64,
    /// NR 迭代上限。达到上限不算无解，只触发种子阶梯下一级。
    pub nr_max_iter: usize,
    pub lm_tol: f64,
    /// LMPF 终审的迭代预算（TR-LMPF 实测 19~25 次分解收敛，给足余量）。
    pub lm_max_iter: usize,
    /// LMPF 步控：默认自适应信赖域（探针 + μ 热启动）。
    pub lm_options: LmOptions,
}

impl Default for MarchOptions {
    fn default() -> Self {
        Self {
            alpha0: 1.0,
            step: 0.5,
            alpha_cap: 10.0,
            nr_tol: 1e-8,
            nr_max_iter: 20,
            lm_tol: 1e-8,
            lm_max_iter: 200,
            lm_options: LmOptions {
                trust_region: Some(TrustRegionOptions {
                    adaptive_initial: true,
                    ..TrustRegionOptions::default()
                }),
                ..LmOptions::default()
            },
        }
    }
}

/// 本探测点实际奏效的种子来源（记账与诊断用）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seed {
    /// α=α0 的首次 dc 启动。
    Dc,
    /// 最后一个可行解。
    Warm,
    /// NR 失败后 B′ 回代重播种（dc 只修角度，V 幅值取 spec）。
    DcReseed,
    /// 两级 NR 均失败，LMPF 收敛到零残差把解救回。
    LmRescued,
}

/// 单个探测点的判决。
#[derive(Clone, Copy, Debug)]
pub enum Verdict {
    /// NR 收敛。
    Feasible { seed: Seed, nr_iters: usize },
    /// NR 两级失败，LMPF 收敛到零残差：点可行，解由 LMPF 白送。
    Rescued { lm_iters: usize },
    /// LMPF 最小二乘证书：min ‖F‖∞ 仍超差，判无解。
    Infeasible { res_inf: f64, lm_iters: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct MarchPoint {
    pub alpha: f64,
    pub verdict: Verdict,
}

/// 行军结果。`bracket = Some((α_lo, α_hi))` 表示 α_lo 可行、α_hi 持
/// 最小二乘无解证书；阶段二在此区间上收缩。
pub struct MarchOutcome {
    pub points: Vec<MarchPoint>,
    pub bracket: Option<(f64, f64)>,
    /// 区间下端可行解：阶段二的暖种子。
    pub v_at_lo: DVector<Complex64>,
    /// 台账：NR 迭代数（每次迭代一次分解）。
    pub nr_iterations: usize,
    /// 台账：LMPF 外层迭代数（含救援与终审）。
    pub lm_iterations: usize,
}

/// dc 播种机：B′ 只与拓扑有关，模型与求解器对象全程复用；
/// 每个 α 的播种成本是一次回代。
struct DcSeeder {
    model: DcpfModel,
    solver: DefaultSolver,
    buf: Vec<f64>,
}

impl DcSeeder {
    /// `v_spec`：平启动/spec 电压（slack 角固定、PV 幅值取 spec），
    /// 既是 dc 结果的 V 幅值来源，也是角度边界条件的来源。
    fn build(
        ybus: &CscMatrix<Complex64>,
        v_spec: &DVector<Complex64>,
        n_active: usize,
        to_perm: &[usize],
        p_shift: &[f64],
    ) -> Self {
        let model = DcpfModel::from_ybus(ybus, v_spec, n_active, to_perm, p_shift);
        let buf = vec![0.0; n_active];
        Self {
            model,
            solver: DefaultSolver::default(),
            buf,
        }
    }

    fn seed(&mut self, s_alpha: &DVector<Complex64>, v_spec: &DVector<Complex64>) -> Option<DVector<Complex64>> {
        dcpf_initial_v(
            &mut self.model,
            s_alpha,
            v_spec,
            &mut self.buf,
            &mut self.solver,
        )
        .ok()
    }
}

/// 阶段一行军器。拓扑全程不变：NR 的 Jacobian 符号缓存、LMPF 的增广
/// KKT 符号与 DCPF 模型全部只建一次，α 变化只触发数值 fill 与分解。
pub struct AlphaMarch<SNR: Solve, SLM: Solve> {
    npv: usize,
    npq: usize,
    s_base: DVector<Complex64>,
    v_spec: DVector<Complex64>,
    nr_cache: NewtonCache,
    nr_solver: SNR,
    lm: GnTriuDriver,
    lm_solver: SLM,
    dc: Option<DcSeeder>,
}

impl<SNR: Solve, SLM: Solve> AlphaMarch<SNR, SLM> {
    /// `p_shift`：移相变压器折算的 nodal 注入（无则传零向量）。
    /// `to_perm`：母线排列置换（与 `DcpfModel::from_ybus` 约定一致）。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ybus: &CscMatrix<Complex64>,
        npv: usize,
        npq: usize,
        s_base: DVector<Complex64>,
        v_spec: DVector<Complex64>,
        to_perm: &[usize],
        p_shift: &[f64],
        nr_solver: SNR,
        lm_solver: SLM,
    ) -> Self {
        let lm = GnTriuDriver::build_operator(ybus, npv, npq, s_base.as_slice().to_vec());
        let dc = DcSeeder::build(ybus, &v_spec, npv + npq, to_perm, p_shift);
        Self {
            npv,
            npq,
            s_base,
            v_spec,
            nr_cache: NewtonCache::default(),
            nr_solver,
            lm,
            lm_solver,
            dc: Some(dc),
        }
    }

    /// 不用 dc 播种（纯暖种子 + LMPF 终审），供对照实验。
    pub fn without_dc(mut self) -> Self {
        self.dc = None;
        self
    }

    fn nr_probe(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        s_alpha: &DVector<Complex64>,
        v0: &DVector<Complex64>,
        opts: &MarchOptions,
    ) -> Option<(DVector<Complex64>, usize)> {
        newton_pf(
            ybus,
            s_alpha,
            v0,
            self.npv,
            self.npq,
            Some(opts.nr_tol),
            Some(opts.nr_max_iter),
            &mut self.nr_solver,
            Some(&mut self.nr_cache),
        )
        .ok()
        .map(|(v, it)| (v, it))
    }

    fn lm_verdict(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        s_alpha: &DVector<Complex64>,
        v0: &DVector<Complex64>,
        opts: &MarchOptions,
    ) -> (bool, usize, f64, DVector<Complex64>) {
        self.lm.sbus.copy_from_slice(s_alpha.as_slice());
        let mut v = v0.as_slice().to_vec();
        let r = self.lm.solve_gn_with_options(
            ybus,
            &mut self.lm_solver,
            &mut v,
            opts.lm_tol,
            opts.lm_max_iter,
            &opts.lm_options,
        );
        (r.converged, r.iterations, r.res_inf, DVector::from_vec(v))
    }

    /// 执行行军。`ybus` 与构造时同一对象（借用规避自持引用）。
    pub fn run(&mut self, ybus: &CscMatrix<Complex64>, opts: &MarchOptions) -> MarchOutcome {
        let mut points = Vec::new();
        let mut nr_iterations = 0;
        let mut lm_iterations = 0;

        // α=α0：dc 启动（无 dc 时退化为 spec 平启动）。
        let s0 = self.s_base.scale(opts.alpha0);
        let mut v_warm = match self.dc.as_mut().map(|d| d.seed(&s0, &self.v_spec)) {
            Some(Some(v)) => v,
            _ => self.v_spec.clone(),
        };
        let mut alpha_lo: Option<f64> = None;
        let mut alpha = opts.alpha0;
        let mut first_probe = true;

        let bracket = loop {
            let s_alpha = self.s_base.scale(alpha);
            // 第一级：暖种子 NR。
            let warm_seed = if first_probe { Seed::Dc } else { Seed::Warm };
            let mut verdict = match self.nr_probe(ybus, &s_alpha, &v_warm, opts) {
                Some((v, it)) => {
                    nr_iterations += it;
                    v_warm = v;
                    Verdict::Feasible { seed: warm_seed, nr_iters: it }
                }
                None => {
                    // 第二级：dc 重播种 NR（B′ 回代，~免费）。
                    let reseeded = match self.dc.as_mut().map(|d| d.seed(&s_alpha, &self.v_spec)) {
                        Some(Some(v_dc)) => match self.nr_probe(ybus, &s_alpha, &v_dc, opts) {
                            Some((v, it)) => {
                                nr_iterations += it;
                                v_warm = v;
                                Some(Verdict::Feasible { seed: Seed::DcReseed, nr_iters: it })
                            }
                            None => None,
                        },
                        _ => None,
                    };
                    match reseeded {
                        Some(v) => v,
                        // 终审：LMPF 从最后可行解出发（NR 的发散残渣绝不用）。
                        None => {
                            let (ok, it, res_inf, v_lm) =
                                self.lm_verdict(ybus, &s_alpha, &v_warm, opts);
                            lm_iterations += it;
                            if ok {
                                v_warm = v_lm;
                                Verdict::Rescued { lm_iters: it }
                            } else {
                                Verdict::Infeasible { res_inf, lm_iters: it }
                            }
                        }
                    }
                }
            };

            let this_alpha = alpha;
            points.push(MarchPoint { alpha: this_alpha, verdict });
            first_probe = false;

            match points.last().unwrap().verdict {
                Verdict::Infeasible { .. } => break alpha_lo.map(|lo| (lo, this_alpha)),
                _ => {
                    alpha_lo = Some(this_alpha);
                    alpha += opts.step;
                    if alpha > opts.alpha_cap {
                        break None;
                    }
                }
            }
        };

        MarchOutcome {
            points,
            bracket,
            v_at_lo: v_warm,
            nr_iterations,
            lm_iterations,
        }
    }
}

#[cfg(test)]
mod tests;
