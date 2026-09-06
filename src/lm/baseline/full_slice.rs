//! COO增广基线：保留原Jacobian计算及COO转换，复用求解器和CSC输入缓冲区。
//! 步长控制与当前GN-LM一致，参数通过LmOptions传入。

use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;

use super::super::residual::residual;
use super::CooSystem;
use crate::lm::step_control::{
    DampingWeights, LmOptions, TrialError, TrustRegion,
    augmented_predicted_reduction as predicted_reduction, polar_trial,
};

/// 全Jacobian计算、裁剪、COO组装完整增广矩阵。
pub struct AugFsDriver {
    pub sbus: Vec<Complex64>,
    /// true只写增广上三角；false保留完整COO基线。默认false。
    pub upper_only: bool,
    n_bus: usize,
    n_act: usize,
    npq: usize,
    n_state: usize,
    /// Reduced-system remap: full row/col index → reduced index (or
    /// `usize::MAX`). Rows: P of active buses (i), Q of PQ buses (nb + i);
    /// cols: θ of active (j), |V| of PQ (nb + j).
    row_map: Vec<usize>,
    col_map: Vec<usize>,
    /// Ybus transpose (CSC of Yᵀ = row-wise view of Y). Built once at
    /// symbolic time: the full-J walk needs `Y_ij` per *row* i, and our CSC
    /// only serves columns. Required for correctness on phase-shifter cases
    /// where Ybus is numerically asymmetric (PEGASE9241); walking column i
    /// as "row i" silently substitutes `Y_ji` there.
    ybus_t: CscMatrix<Complex64>,
    // Scratch.
    ibus: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    full_j: CooMatrix<f64>,
    coo: CooMatrix<f64>,
    reduced_j_nnz: usize,
    system: CooSystem,
    b: Vec<f64>,
    vt: Vec<Complex64>,
    // Profiling (ns), same convention as the other baselines.
    /// Full-J computation (all quadrants, all buses — the 2× waste).
    pub prof_full_j_ns: u64,
    /// Slice + augmented COO stack + sort/convert.
    pub prof_slice_coo_ns: u64,
    /// 线性求解累计时间；同一结构复用符号分析。
    pub prof_solve_ns: u64,
    /// 右端项准备；μ写入已计入COO组装。
    pub prof_mu_ns: u64,
    pub n_solves: u64,
}

/// Outcome of one run (same shape as the other baselines').
pub struct AugFsResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl AugFsDriver {
    pub fn build(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        let nb = ybus.ncols();
        let n_act = n_pv + n_pq;
        let n_state = n_act + n_pq;
        // 输入顺序为[PQ | PV | slack]，预先建立全Jacobian到保留方程的索引。
        let mut row_map = vec![usize::MAX; 2 * nb];
        let mut col_map = vec![usize::MAX; 2 * nb];
        for i in 0..n_act {
            row_map[i] = i; // P row of active bus
            col_map[i] = i; // θ col of active bus
        }
        for i in 0..n_pq {
            row_map[nb + i] = n_act + i; // Q row of PQ bus
            col_map[nb + i] = n_act + i; // |V| col of PQ bus
        }
        // 每个Y条目在保留的P/Q行、θ/|V|列中贡献的导数个数。
        let mut reduced_j_nnz = 0;
        for j in 0..n_act {
            let columns = 1 + usize::from(j < n_pq);
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                let i = ybus.row_indices()[p];
                let rows = usize::from(i < n_act) + usize::from(i < n_pq);
                reduced_j_nnz += rows * columns;
            }
        }
        Self {
            sbus,
            upper_only: false,
            n_bus: nb,
            n_act,
            npq: n_pq,
            n_state,
            row_map,
            col_map,
            ybus_t: ybus.transpose(),
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            full_j: CooMatrix::new(2 * nb, 2 * nb),
            coo: CooMatrix::new(2 * n_state, 2 * n_state),
            reduced_j_nnz,
            system: CooSystem::default(),
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            prof_full_j_ns: 0,
            prof_slice_coo_ns: 0,
            prof_solve_ns: 0,
            prof_mu_ns: 0,
            n_solves: 0,
        }
    }

    /// Test-only accessors for the cross-validation bench.
    #[cfg(test)]
    pub(crate) fn full_j_coo_pub(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &[Complex64],
    ) -> CooMatrix<f64> {
        self.full_j_coo(ybus, v);
        self.full_j.clone()
    }
    #[cfg(test)]
    pub(crate) fn map_row(&self, r: usize) -> usize {
        self.row_map[r]
    }
    #[cfg(test)]
    pub(crate) fn map_col(&self, c: usize) -> usize {
        self.col_map[c]
    }

    /// The full 2nb×2nb polar Jacobian as COO triplets — every quadrant of
    /// every bus, slack and PV included. Reuses the triplet buffers;
    /// slicing remains a separate, measurable pass.
    fn full_j_coo(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t = std::time::Instant::now();
        let nb = self.n_bus;
        // 对角导数需要的节点电流。
        for x in self.ibus.iter_mut() {
            *x = Complex64::new(0.0, 0.0);
        }
        for j in 0..nb {
            for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
                self.ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
            }
        }
        let coo = &mut self.full_j;
        coo.clear_triplets();
        coo.reserve(4 * ybus.nnz());
        // Walk ROW i via the transpose: entries (i, j, Y_ij). (Walking
        // column i of Ybus would silently substitute Y_ji — wrong for
        // phase-shifter branches where Ybus is numerically asymmetric.)
        let (y_cp, y_ri, y_v) = (
            self.ybus_t.col_offsets(),
            self.ybus_t.row_indices(),
            self.ybus_t.values(),
        );
        for i in 0..nb {
            let (mi, thi) = v[i].to_polar();
            let si = v[i] * self.ibus[i].conj(); // P_i + jQ_i
            for p in y_cp[i]..y_cp[i + 1] {
                let j = y_ri[p];
                let y = y_v[p]; // G + jB
                let (mj, thj) = v[j].to_polar();
                if i != j {
                    let (sij, cij) = (thi - thj).sin_cos();
                    let mm = mi * mj;
                    // off-diagonal quadrant entries
                    let j11 = mm * (y.re * sij - y.im * cij);
                    let j21 = -mm * (y.re * cij + y.im * sij);
                    let j12 = mi * (y.re * cij + y.im * sij);
                    let j22 = mi * (y.re * sij - y.im * cij);
                    coo.push(i, j, j11);
                    coo.push(nb + i, j, j21);
                    coo.push(i, nb + j, j12);
                    coo.push(nb + i, nb + j, j22);
                } else {
                    let m2 = mi * mi;
                    coo.push(i, i, -si.im - y.im * m2); // -Q_i - B_ii |V_i|²
                    coo.push(nb + i, i, si.re - y.re * m2); //  P_i - G_ii |V_i|²
                    coo.push(i, nb + i, si.re / mi + y.re * mi); // P_i/|V_i| + G_ii |V_i|
                    coo.push(nb + i, nb + i, si.im / mi - y.im * mi); // Q_i/|V_i| - B_ii |V_i|
                }
            }
        }
        self.prof_full_j_ns += t.elapsed().as_nanos() as u64;
    }

    /// 裁剪全Jacobian，以COO组装增广矩阵并转为CSC；Jᵀ复用J的数值。
    fn slice_stack_convert(
        &mut self,
        v: &[Complex64],
        mu: f64,
        damping: &DampingWeights,
    ) {
        let t = std::time::Instant::now();
        let n = self.n_state;
        let full = &self.full_j;
        let (fr, fc, fv) = (full.row_indices(), full.col_indices(), full.values());
        let coo = &mut self.coo;
        coo.clear_triplets();
        let copies = if self.upper_only { 1 } else { 2 };
        coo.reserve(copies * self.reduced_j_nnz + 2 * n);
        for c in 0..n {
            coo.push(c, c, mu * damping.weight(v, c, self.n_act));
            coo.push(n + c, n + c, -1.0);
        }
        for k in 0..full.nnz() {
            let (rr, cc) = (self.row_map[fr[k]], self.col_map[fc[k]]);
            if rr != usize::MAX && cc != usize::MAX {
                let v = fv[k];
                if !self.upper_only {
                    coo.push(n + rr, cc, v); // J block
                }
                coo.push(cc, n + rr, v); // Jᵀ block
            }
        }
        let csc = CscMatrix::from(&*coo);
        self.system.update(&csc);
        self.prof_slice_coo_ns += t.elapsed().as_nanos() as u64;
    }

    pub fn solve_aug_fs(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> AugFsResult {
        self.solve_aug_fs_with_options(ybus, v, tol, maxit, &LmOptions::default())
    }

    pub fn solve_aug_fs_with_options(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> AugFsResult {
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
                return AugFsResult {
                    iterations: it,
                    converged: true,
                    res_inf,
                };
            }
            self.full_j_coo(ybus, v);

            let mut accepted = false;
            for _ in 0..options.max_trials {
                self.slice_stack_convert(v, mu, &damping);
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
                        return AugFsResult {
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
                        return AugFsResult {
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
                        return AugFsResult {
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
                    return AugFsResult {
                        iterations: it,
                        converged: false,
                        res_inf,
                    };
                }
            }
            if !accepted {
                return AugFsResult {
                    iterations: it,
                    converged: false,
                    res_inf,
                };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        AugFsResult {
            iterations: maxit,
            converged: res_inf < tol,
            res_inf,
        }
    }
}
