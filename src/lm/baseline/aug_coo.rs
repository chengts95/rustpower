//! COO增广基线：保留原Jacobian计算及COO转换，复用求解器和CSC输入缓冲区。
//! 步长控制与当前GN-LM一致，参数通过LmOptions传入。

use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

use super::super::pattern::KktPattern;
use super::super::residual::residual;
use super::CooSystem;
use crate::basic::new_dsdvbus4::fill_jacobian_v4;
use crate::lm::step_control::{
    DampingWeights, LmOptions, TrialError, TrustRegion,
    augmented_predicted_reduction as predicted_reduction, polar_trial,
};

/// V4填J，然后以COO组装完整增广矩阵。
pub struct AugCooDriver {
    pub pat: KktPattern,
    pub sbus: Vec<Complex64>,
    /// true只写增广上三角；false保留完整COO基线。默认false。
    pub upper_only: bool,
    /// J values in the shared block-CSC layout (kernel output, per sweep).
    j_vals: Vec<f64>,
    /// Column starts **with the nnz sentinel** (graph.col_starts lacks it).
    j_cols: Vec<usize>,
    // Scratch (allocated once).
    ibus: Vec<Complex64>,
    scalc: Vec<Complex64>,
    vnorm: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    coo: CooMatrix<f64>,
    system: CooSystem,
    b: Vec<f64>,
    vt: Vec<Complex64>,
    n_act: usize,
    npq: usize,
    n_state: usize,
    // Profiling (ns), same convention as `normal_eq::NeDriver`.
    /// V4填J，包含scalc和vnorm准备。
    pub prof_fill_ns: u64,
    /// COO写入、转CSC及固定缓冲区更新。
    pub prof_coo_ns: u64,
    /// 线性求解累计时间；同一结构复用符号分析。
    pub prof_solve_ns: u64,
    /// 右端项准备；μ写入已计入COO组装。
    pub prof_mu_ns: u64,
    /// Number of linear solves (outer iterations + μ retries).
    pub n_solves: u64,
}

/// Outcome of one run (same shape as `GnResult` / `NeResult`).
pub struct AugCooResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl AugCooDriver {
    pub fn build(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let n_state = pat.graph.n_cols;
        let nnz = pat.graph.nnz;
        let mut j_cols = pat.graph.col_starts.clone();
        j_cols.push(nnz); // sentinel
        Self {
            pat,
            sbus,
            upper_only: false,
            j_vals: vec![0.0; nnz],
            j_cols,
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            scalc: vec![Complex64::new(0.0, 0.0); nb],
            vnorm: vec![Complex64::new(1.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            coo: CooMatrix::new(2 * n_state, 2 * n_state),
            system: CooSystem::default(),
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            n_act: n_pv + n_pq,
            npq: n_pq,
            n_state,
            prof_fill_ns: 0,
            prof_coo_ns: 0,
            prof_solve_ns: 0,
            prof_mu_ns: 0,
            n_solves: 0,
        }
    }

    /// J fill via the shared offset kernel (identical to every other path).
    fn fill_j(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t = std::time::Instant::now();
        let nb = ybus.ncols();
        for i in 0..nb {
            self.scalc[i] = v[i] * self.ibus[i].conj();
            let m = v[i].norm();
            self.vnorm[i] = if m > 1e-12 {
                v[i] / m
            } else {
                Complex64::new(1.0, 0.0)
            };
        }
        let cache = &self.pat.cache;
        fill_jacobian_v4::<false>(
            ybus,
            v,
            &self.vnorm,
            &self.scalc,
            &self.pat.graph.col_starts,
            cache.pq_ends(),
            cache.active_ends(),
            cache.diag_ptrs(),
            self.n_act - self.npq,
            self.npq,
            &mut self.j_vals,
        );
        self.prof_fill_ns += t.elapsed().as_nanos() as u64;
    }

    /// 将[μI Jᵀ; J −I]写入COO并转为CSC；Jᵀ复用J的数值。
    fn coo_assemble(&mut self, v: &[Complex64], mu: f64, damping: &DampingWeights) {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let (cs, rows) = (&self.j_cols, &self.pat.graph.row_indices);
        let coo = &mut self.coo;
        coo.clear_triplets();
        // 首次按完整/上三角模式预留容量；后续迭代和重试复用。
        let copies = if self.upper_only { 1 } else { 2 };
        coo.reserve(copies * self.j_vals.len() + 2 * n);
        // μI block.
        for c in 0..n {
            coo.push(c, c, mu * damping.weight(v, c, self.n_act));
        }
        // J block at rows n+r, and Jᵀ block mirrored at (c, n+r).
        for c in 0..n {
            for p in cs[c]..cs[c + 1] {
                let (r, jv) = (rows[p], self.j_vals[p]);
                if !self.upper_only {
                    coo.push(n + r, c, jv);
                }
                coo.push(c, n + r, jv);
            }
        }
        // −I block.
        for c in 0..n {
            coo.push(n + c, n + c, -1.0);
        }
        let csc = CscMatrix::from(&*coo);
        self.system.update(&csc);
        self.prof_coo_ns += t.elapsed().as_nanos() as u64;
    }

    pub fn solve_aug_coo(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> AugCooResult {
        self.solve_aug_coo_with_options(ybus, v, tol, maxit, &LmOptions::default())
    }

    pub fn solve_aug_coo_with_options(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> AugCooResult {
        options.validate().expect("invalid LM options");
        let damping = options.damping_metric.prepare(ybus, self.n_act);
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let mut mu = options.initial_mu;
        let mut region = TrustRegion::new(options.trust_region.as_ref());
        let mut res_inf;
        for it in 0..maxit {
            let f;
            (res_inf, f) = residual(
                ybus,
                &self.sbus,
                &mut self.ibus,
                self.n_act,
                self.npq,
                v,
                &mut self.r,
            );
            if res_inf < tol {
                return AugCooResult {
                    iterations: it,
                    converged: true,
                    res_inf,
                };
            }
            self.fill_j(ybus, v);

            let mut accepted = false;
            for _ in 0..options.max_trials {
                self.coo_assemble(v, mu, &damping);
                let t_mu = std::time::Instant::now();
                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.prof_mu_ns += t_mu.elapsed().as_nanos() as u64;
                let t_solve = std::time::Instant::now();
                let solve_ok = self.system.solve(&mut self.b);
                self.prof_solve_ns += t_solve.elapsed().as_nanos() as u64;
                self.n_solves += 1;
                let delta = &self.b[..n];
                let finite = solve_ok && self.b.iter().all(|x| x.is_finite());
                if !finite {
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        return AugCooResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                let step_norm_squared = damping.step_norm_squared(v, delta, self.n_act);
                if !region.allows(step_norm_squared) {
                    if debug {
                        eprintln!(
                            "it={it} tryμ={mu:.3e} rejected=TrustRadius step={:.3e} radius={:.3e}",
                            step_norm_squared.sqrt(),
                            region.radius
                        );
                    }
                    if !options.increase_mu(&mut mu, options.mu_increase) {
                        return AugCooResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                if let Err(reason) = polar_trial(
                    v,
                    delta,
                    self.n_act,
                    self.npq,
                    options.reject_nonpositive_voltage,
                    &mut self.vt,
                ) {
                    if debug {
                        eprintln!("it={it} tryμ={mu:.3e} rejected={reason:?}");
                    }
                    region.reject();
                    let factor = match reason {
                        TrialError::NonFinite => options.failed_step_increase,
                        TrialError::NonPositiveMagnitude { .. } => options.mu_increase,
                    };
                    if !options.increase_mu(&mut mu, factor) {
                        return AugCooResult {
                            iterations: it,
                            converged: false,
                            res_inf,
                        };
                    }
                    continue;
                }

                let (_, f_new) = {
                    let (n_act, npq) = (self.n_act, self.npq);
                    residual(
                        ybus,
                        &self.sbus,
                        &mut self.ibus,
                        n_act,
                        npq,
                        &self.vt,
                        &mut self.rt,
                    )
                };
                let pred = predicted_reduction(&self.r, &self.b[n..], mu, step_norm_squared);
                let rho = if pred.is_finite() && pred > 0.0 && f_new.is_finite() {
                    (f - f_new) / pred
                } else {
                    -1.0
                };
                if debug {
                    eprintln!(
                        "it={it} tryμ={mu:.3e} res={res_inf:.3e} f={f:.4e} f_new={f_new:.4e} pred={pred:.4e} ρ={rho:.4}"
                    );
                }
                if rho.is_finite() && rho > options.acceptance_threshold {
                    region.accept(rho, step_norm_squared, options.good_step_threshold);
                    v.copy_from_slice(&self.vt);
                    mu = options.accepted_mu(mu, rho);
                    accepted = true;
                    break;
                }
                region.reject();
                if !options.increase_mu(&mut mu, options.mu_increase) {
                    return AugCooResult {
                        iterations: it,
                        converged: false,
                        res_inf,
                    };
                }
            }
            if !accepted {
                return AugCooResult {
                    iterations: it,
                    converged: false,
                    res_inf,
                };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        AugCooResult {
            iterations: maxit,
            converged: res_inf < tol,
            res_inf,
        }
    }
}
