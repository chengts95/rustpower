//! **Triu-slim GN-LM**: the upper-triangle-only counterpart of
//! [`super::gn_flat`], for LDLᵀ backends that never read the lower triangle.
//!
//! Layout (`[μI Jᵀ; J −I]` is symmetric; only its upper triangle is stored):
//!
//! ```text
//! δ-column c     (0..n)  : [μ]                        (one diagonal entry)
//! s-column n + c (0..n)  : [Jᵀ col c | −I]            (graph rows | n + c)
//! nnz_triu = nnz + 2n     (vs slim-full 2·nnz + 2n — storage halved)
//! ```
//!
//! The baseline fill is **row-oriented**: instead of computing J column-wise from the
//! Ybus columns and transpose-copying (the scatter-write pass that dominates
//! the slim-full fill), we walk `Ybusᵀ` — built once at symbolic time — and
//! compute J *row* r directly into s-column r. Writes are fully sequential;
//! only the bus-level vectors (`v`, `Vnorm`, `scalc`) are read scattered,
//! and those are L2-resident. Because the Ybus *pattern* is symmetric, the
//! `pq_ends`/`active_ends`/diagonal-rank caches apply to the row walk
//! unchanged — no new symbolic machinery.
//!
//! The formulas per entry are byte-identical to [`fill_jacobian_v4`]'s (same
//! products, same order, diagonal correction added after the base value), so
//! the output equals `fill_jt`'s transpose of the v4 block **bitwise** (the
//! `triu_matches_flat_*` tests assert exactly this).
//!
//! `build_operator` reuses the same LM loop with the borrowed block operator.
//! It reads the live Y values through the existing symbolic mirror indices,
//! and combines P/Q and angle/magnitude contributions in a single edge sweep.
//! It stores no numerical Yᵀ. Equality to the baseline is at rounding precision.
//!
//! KLU-class full-matrix solvers cannot consume this layout; they keep using
//! [`super::gn_flat::GnFlatLayout`]. Neither can SuiteSparse LDL — it reads
//! the upper triangle of the **permuted** PAP′ and therefore needs the full
//! symmetric pattern. Pure-Rust QDLDL is the one backend that takes the
//! plain upper triangle; the pairing lives in
//! [`super::newton_pf_gn_default`]'s feature ladder.

use crate::lm::step_control::{
    LmOptions, TrialError, TrustRegion, augmented_predicted_reduction as predicted_reduction,
    polar_trial,
};
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::pattern::KktPattern;
use super::residual::residual;
use crate::basic::solver::Solve;

/// Upper-triangle-only slim CSC of `[μI Jᵀ; J −I]`, symbolic part.
pub struct GnTriuLayout {
    pub n_state: usize,
    /// `nnz + 2·n_state`.
    pub nnz_triu: usize,
    pub col_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
}

impl GnTriuLayout {
    pub fn build(pat: &KktPattern) -> Self {
        let n = pat.graph.n_cols;
        let nnz = pat.graph.nnz;
        let cs = &pat.graph.col_starts;

        let mut col_offsets = Vec::with_capacity(2 * n + 1);
        let mut row_indices = Vec::with_capacity(nnz + 2 * n);

        // δ-columns: just the μ diagonal (row c).
        for c in 0..n {
            col_offsets.push(c);
            row_indices.push(c);
        }
        // s-columns: [Jᵀ segment (graph rows) | −I entry at row n+c].
        for c in 0..n {
            col_offsets.push(n + cs[c] + c);
            row_indices.extend_from_slice(pat.graph.col_rows(c));
            row_indices.push(n + c);
        }
        col_offsets.push(nnz + 2 * n);

        Self {
            n_state: n,
            nnz_triu: nnz + 2 * n,
            col_offsets,
            row_indices,
        }
    }

    /// Stamp the constant `−I` tail of every s-column, once.
    pub fn stamp_neg_i(&self, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.nnz_triu);
        for c in 0..self.n_state {
            values[self.col_offsets[self.n_state + c + 1] - 1] = -1.0;
        }
    }
}

/// Row-oriented Jᵀ fill: compute J row r directly into s-column r of the
/// triu values array. Formulas byte-identical to [`fill_jacobian_v4`] with
/// the loop roles swapped (row bus loop-invariant, column bus per-entry).
///
/// The diagonal correction rides the `j == k` entry exactly like v4 adds it
/// after the base value; the diagonal's rank within row k equals
/// `diag_ptrs[k] − y_cp[k]` because the Ybus pattern is symmetric.
pub fn fill_jt_rows(
    ybus_t: &CscMatrix<Complex64>, // CSC of Yᵀ (row view of Ybus)
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    pat: &KktPattern,
    values: &mut [f64],
    col_offsets: &[usize],
) {
    let cache = &pat.cache;
    let n_act = cache.n_active();
    let (pq_ends, active_ends, diag_off) = (cache.pq_ends(), cache.active_ends(), cache.diag_off());
    let (yt_cp, yt_ri, yt_v) = (ybus_t.col_offsets(), ybus_t.row_indices(), ybus_t.values());
    let n = pat.graph.n_cols;

    // s-column c (c in 0..n): bus k = c % n_act, row kind P if c < n_act.
    for c in 0..n {
        let (k, is_p) = if c < n_act {
            (c, true)
        } else {
            (c - n_act, false)
        };
        let (pq_end, active_end) = (pq_ends[k], active_ends[k]);
        let out_start = col_offsets[n + c];
        let out = unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr().add(out_start),
                col_offsets[n + c + 1] - out_start - 1, // exclude the −I tail
            )
        };

        let (ec, fc) = (v[k].re, v[k].im);
        let (enc, fnc) = (vnorm[k].re, vnorm[k].im);
        let (pk, qk) = (scalc[k].re, scalc[k].im);
        let vmag = ec * enc + fc * fnc;
        let inv_vmag = 1.0 / vmag;
        let t_diag = diag_off[k];

        // θ-segment (all active neighbours j < n_act, ascending j):
        //   P-row: J11[k,j] =  f_c·Va_re − e_c·Va_im   (diag += −q_k)
        //   Q-row: J21[k,j] = −(e_c·Va_re + f_c·Va_im) (diag += +p_k)
        for t in 0..active_end {
            let p = yt_cp[k] + t;
            let j = yt_ri[p];
            let y = yt_v[p]; // = Ybus[k, j]
            let va_re = y.re * v[j].re - y.im * v[j].im;
            let va_im = y.re * v[j].im + y.im * v[j].re;
            let mut val = if is_p {
                fc * va_re - ec * va_im
            } else {
                -(ec * va_re + fc * va_im)
            };
            if t == t_diag {
                val += if is_p { -qk } else { pk };
            }
            out[t] = val;
        }
        // |V|-segment (PQ neighbours j < npq, ascending j):
        //   P-row: J12[k,j] = e_c·Vm_re + f_c·Vm_im (diag += +p_k/|V_k|)
        //   Q-row: J22[k,j] = f_c·Vm_re − e_c·Vm_im (diag += +q_k/|V_k|)
        for t in 0..pq_end {
            let p = yt_cp[k] + t;
            let j = yt_ri[p];
            let y = yt_v[p];
            let vm_re = y.re * vnorm[j].re - y.im * vnorm[j].im;
            let vm_im = y.re * vnorm[j].im + y.im * vnorm[j].re;
            let mut val = if is_p {
                ec * vm_re + fc * vm_im
            } else {
                fc * vm_re - ec * vm_im
            };
            if t == t_diag {
                val += if is_p { pk * inv_vmag } else { qk * inv_vmag };
            }
            out[active_end + t] = val;
        }
    }
}

// 装配方式及其专属工作区；迭代控制不依赖装配方式。
enum Assembly {
    TransposedRows {
        ybus_t: CscMatrix<Complex64>,
        vnorm: Vec<Complex64>,
    },
    Direct {
        inv_vmag: Vec<f64>,
    },
}

/// GN-LM driver on the triu-slim layout (LDLᵀ backends only).
pub struct GnTriuDriver {
    pub pat: KktPattern,
    pub triu: GnTriuLayout,
    pub sbus: Vec<Complex64>,
    assembly: Assembly,
    values: Vec<f64>,
    // Scratch (allocated once).
    ibus: Vec<Complex64>,
    scalc: Vec<Complex64>,
    r: Vec<f64>,
    rt: Vec<f64>,
    b: Vec<f64>, // 求解前[0; −r]，求解后[δ; s]，s = r + Jδ。
    vt: Vec<Complex64>,
    n_act: usize,
    npq: usize,
    n_state: usize,
    // Profiling (ns), same convention as the other drivers.
    pub prof_fill_ns: u64,
    /// μ对角及右端准备的累计时间。
    pub prof_mu_ns: u64,
    pub prof_solve_ns: u64,
    pub n_solves: u64,
}

/// Outcome of one run (same shape as `GnResult`).
pub struct GnTriuResult {
    pub iterations: usize,
    pub converged: bool,
    pub res_inf: f64,
}

impl GnTriuDriver {
    #[cfg(test)]
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn build(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        Self::build_with_assembly(ybus, n_pv, n_pq, sbus, |_| Assembly::TransposedRows {
            ybus_t: ybus.transpose(),
            vnorm: vec![Complex64::new(1.0, 0.0); ybus.ncols()],
        })
    }

    /// Use the borrowed Jacobian block operator; retain the original driver as baseline.
    pub fn build_operator(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
    ) -> Self {
        Self::build_with_assembly(ybus, n_pv, n_pq, sbus, |_| Assembly::Direct {
            inv_vmag: vec![0.0; ybus.ncols()],
        })
    }

    fn build_with_assembly(
        ybus: &CscMatrix<Complex64>,
        n_pv: usize,
        n_pq: usize,
        sbus: Vec<Complex64>,
        make_assembly: impl FnOnce(&KktPattern) -> Assembly,
    ) -> Self {
        let nb = ybus.ncols();
        let pat = KktPattern::build(ybus, n_pv, n_pq);
        let assembly = make_assembly(&pat);
        let triu = GnTriuLayout::build(&pat);
        let n_state = triu.n_state;
        let mut values = vec![0.0; triu.nnz_triu];
        triu.stamp_neg_i(&mut values);
        Self {
            pat,
            triu,
            sbus,
            assembly,
            values,
            ibus: vec![Complex64::new(0.0, 0.0); nb],
            scalc: vec![Complex64::new(0.0, 0.0); nb],
            r: vec![0.0; n_state],
            rt: vec![0.0; n_state],
            b: vec![0.0; 2 * n_state],
            vt: vec![Complex64::new(0.0, 0.0); nb],
            n_act: n_pv + n_pq,
            npq: n_pq,
            n_state,
            prof_fill_ns: 0,
            prof_mu_ns: 0,
            prof_solve_ns: 0,
            n_solves: 0,
        }
    }

    pub fn reset_prof(&mut self) {
        self.prof_fill_ns = 0;
        self.prof_mu_ns = 0;
        self.prof_solve_ns = 0;
        self.n_solves = 0;
    }

    /// 填充上三角中的 Jᵀ：原路径读取缓存 Yᵀ，直接算子读取 Ybus。
    /// 两者复用残差计算得到的 scalc，不修改 μ 和 −I。
    fn fill(&mut self, ybus: &CscMatrix<Complex64>, v: &[Complex64]) {
        let t = std::time::Instant::now();
        match &mut self.assembly {
            Assembly::Direct { inv_vmag } => {
                for (inv, voltage) in inv_vmag.iter_mut().zip(v) {
                    *inv = 1.0 / voltage.norm();
                }
                use crate::basic::jacobian_operator::{JacobianBlock, JacobianOperator};
                let cache = &self.pat.cache;
                let op = JacobianOperator {
                    ybus,
                    v,
                    inv_vmag,
                    scalc: &self.scalc,
                    pq_ends: cache.pq_ends(),
                    active_ends: cache.active_ends(),
                    diag_ptrs: cache.diag_ptrs(),
                    mirror: cache.y_trans(),
                    npq: self.npq,
                    npv: self.n_act - self.npq,
                };
                op.fill::<false, true>(
                    JacobianBlock {
                        column_starts: &self.triu.col_offsets[self.n_state..],
                        base: 0,
                        prefix: 0,
                    },
                    &mut self.values,
                );
            }
            Assembly::TransposedRows { ybus_t, vnorm } => {
                let nb = ybus.ncols();
                for i in 0..nb {
                    let m = v[i].norm();
                    vnorm[i] = if m > 1e-12 {
                        v[i] / m
                    } else {
                        Complex64::new(1.0, 0.0)
                    };
                }
                fill_jt_rows(
                    ybus_t,
                    v,
                    vnorm,
                    &self.scalc,
                    &self.pat,
                    &mut self.values,
                    &self.triu.col_offsets,
                );
            }
        }
        self.prof_fill_ns += t.elapsed().as_nanos() as u64;
    }

    /// Classical GN-LM with the same default step control as the full layout.
    /// See `solve_gn_with_options` for configurable settings.
    pub fn solve_gn<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
    ) -> GnTriuResult {
        self.solve_gn_with_options(ybus, solver, v, tol, maxit, &LmOptions::default())
    }

    /// Configurable damping and trial acceptance. Invalid options panic;
    /// callers accepting external settings can use `LmOptions::validate` first.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_gn_with_options<S: Solve>(
        &mut self,
        ybus: &CscMatrix<Complex64>,
        solver: &mut S,
        v: &mut [Complex64],
        tol: f64,
        maxit: usize,
        options: &LmOptions,
    ) -> GnTriuResult {
        options.validate().expect("invalid LM options");
        let damping = options.damping_metric.prepare(ybus, self.n_act);
        let n = self.n_state;
        let debug = std::env::var("RUSTPOWER_LM_DEBUG").is_ok();
        let mut mu = options.initial_mu;
        let mut region = TrustRegion::new(options.trust_region.as_ref());
        let mut res_inf;
        for it in 0..maxit {
            let f;
            {
                let (n_act, npq) = (self.n_act, self.npq);
                (res_inf, f) = crate::lm::residual::residual_with_power(
                    ybus,
                    &self.sbus,
                    &mut self.ibus,
                    &mut self.scalc,
                    n_act,
                    npq,
                    v,
                    &mut self.r,
                );
            }
            if res_inf < tol {
                return GnTriuResult {
                    iterations: it,
                    converged: true,
                    res_inf,
                };
            }
            self.fill(ybus, v);

            let mut accepted = false;
            for _ in 0..options.max_trials {
                let t_mu = std::time::Instant::now();
                // μ slot = the lone entry of δ-column c.
                for c in 0..n {
                    self.values[c] = mu * damping.weight(v, c, self.n_act);
                }

                self.b[..n].fill(0.0);
                for i in 0..n {
                    self.b[n + i] = -self.r[i];
                }
                self.prof_mu_ns += t_mu.elapsed().as_nanos() as u64;
                let t_solve = std::time::Instant::now();
                let solve_ok = solver
                    .solve(
                        &mut self.triu.col_offsets,
                        &mut self.triu.row_indices,
                        &mut self.values,
                        &mut self.b,
                        2 * n,
                    )
                    .is_ok();
                self.prof_solve_ns += t_solve.elapsed().as_nanos() as u64;
                self.n_solves += 1;
                let delta = &self.b[..n];
                let finite = solve_ok && self.b.iter().all(|x| x.is_finite());
                if !finite {
                    if !options.increase_mu(&mut mu, options.failed_step_increase) {
                        return GnTriuResult {
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
                        return GnTriuResult {
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
                        return GnTriuResult {
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
                    return GnTriuResult {
                        iterations: it,
                        converged: false,
                        res_inf,
                    };
                }
            }
            if !accepted {
                return GnTriuResult {
                    iterations: it,
                    converged: false,
                    res_inf,
                };
            }
        }
        let (n_act, npq) = (self.n_act, self.npq);
        let (res_inf, _) = residual(ybus, &self.sbus, &mut self.ibus, n_act, npq, v, &mut self.r);
        GnTriuResult {
            iterations: maxit,
            converged: res_inf < tol,
            res_inf,
        }
    }
}

/// Classical GN-LM power flow with the same contract as
/// [`crate::basic::newtonpf::newton_pf`] — the ECS-plugin entry point on the
/// triu-slim layout. QDLDL only (it consumes the plain upper triangle);
/// SuiteSparse LDL needs the full symmetric pattern (permuted-triangle
/// access), and KLU-class LU backends need the full matrix — both must use
/// [`super::gn_flat::newton_pf_gn`].
#[allow(clippy::too_many_arguments)]
pub fn newton_pf_gn_triu<Solver: Solve>(
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
    let mut driver = GnTriuDriver::build(ybus, npv, npq, sbus.iter().copied().collect());
    let mut v: Vec<Complex64> = v_init.iter().copied().collect();
    let res = driver.solve_gn(ybus, solver, &mut v, tol, maxit);
    let dv = DVector::from_vec(v);
    if res.converged {
        Ok((dv, res.iterations))
    } else {
        Err((
            format!(
                "GN-LM(triu) did not converge (res_inf = {:.3e})",
                res.res_inf
            ),
            dv,
            res.iterations,
        ))
    }
}

#[cfg(all(test, any(feature = "klu", feature = "klu_dyn")))]
mod tests {
    use super::*;
    use crate::basic::new_dsdvbus4::fill_jacobian_v4;
    use crate::basic::solver::KLUSolver;
    use crate::lm::gn_flat::GnDriver;
    use crate::lm::kernels::fill_jt;
    use crate::lm::residual::fixtures::load_ieee39_mat;

    #[test]
    fn prediction_from_augmented_solution_matches_quadratic_model() {
        use nalgebra::{Matrix2, Matrix4, Vector2, Vector4};

        let j = Matrix2::new(3.0, -2.0, 1.0, 4.0);
        let r = Vector2::new(2.0, -3.0);
        for weights in [Vector2::new(1.0, 1.0), Vector2::new(0.25, 4.0)] {
            for mu in [1e-4, 1e-2, 10.0] {
                let a = Matrix4::new(
                    mu * weights[0],
                    0.0,
                    j[(0, 0)],
                    j[(1, 0)],
                    0.0,
                    mu * weights[1],
                    j[(0, 1)],
                    j[(1, 1)],
                    j[(0, 0)],
                    j[(0, 1)],
                    -1.0,
                    0.0,
                    j[(1, 0)],
                    j[(1, 1)],
                    0.0,
                    -1.0,
                );
                let solution = a.lu().solve(&Vector4::new(0.0, 0.0, -r[0], -r[1])).unwrap();
                let delta = Vector2::new(solution[0], solution[1]);
                let norm = delta.component_mul(&weights).dot(&delta);
                let got = predicted_reduction(r.as_slice(), &solution.as_slice()[2..], mu, norm);
                // 直接计算未加阻尼的二次模型下降量，不使用待测公式。
                let expected = 0.5 * (r.norm_squared() - (r + j * delta).norm_squared());
                assert!(
                    (got - expected).abs() < 1e-12,
                    "mu={mu}: {got} != {expected}"
                );
            }
        }
    }

    /// 逐位对照:triu 行向直填的 Jᵀ 段必须等于 v4 列向 fill + fill_jt 的
    /// 转置输出——同一公式、无求和顺序差异,必须严格逐位相等。
    /// 同时验证"模式对称 ⇒ 行向对角 rank = diag_off"这一关键假设。
    #[test]
    fn triu_matches_flat_bitwise() {
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();
        let v: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();

        // triu 路径(先 residual 填 ibus,再 fill,与 solve 循环同序)。
        let mut d = GnTriuDriver::build(ybus, npv, npq, sbus);
        {
            let (n_act, npq_) = (d.n_act, d.npq);
            crate::lm::residual::residual_with_power(
                ybus,
                &d.sbus,
                &mut d.ibus,
                &mut d.scalc,
                n_act,
                npq_,
                &v,
                &mut d.r,
            );
        }
        d.fill(ybus, &v);

        let Assembly::TransposedRows { vnorm, .. } = &d.assembly else {
            unreachable!()
        };
        // 参考:v4 列向 kernel + fill_jt 转置拷贝(同一 v、同一 scalc/vnorm)。
        let pat = &d.pat;
        let cache = &pat.cache;
        let nnz = pat.graph.nnz;
        let cs = &pat.graph.col_starts;
        let mut j_block = vec![0.0; nnz];
        let mut jt_block = vec![0.0; nnz];
        fill_jacobian_v4::<false>(
            ybus,
            &v,
            vnorm,
            &d.scalc,
            cs,
            cache.pq_ends(),
            cache.active_ends(),
            cache.diag_ptrs(),
            npv,
            npq,
            &mut j_block,
        );
        fill_jt::<false>(ybus, pat, j_block.as_ptr(), jt_block.as_mut_ptr());

        let n = d.n_state;
        let gp = &d.triu.col_offsets;
        assert_eq!(d.triu.nnz_triu, nnz + 2 * n);
        for c in 0..n {
            let s = gp[n + c];
            let e = gp[n + c + 1] - 1; // 排除 −I 尾巴
            let l = e - s;
            let got = &d.values[s..e];
            let want = &jt_block[cs[c]..cs[c] + l];
            for (t, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "s-col {c} entry {t}: triu={a:e} ref={b:e}"
                );
            }
            // −I 尾巴保持 write-once。
            assert_eq!(d.values[gp[n + c + 1] - 1].to_bits(), (-1.0f64).to_bits());
        }
    }

    /// IEEE39 收敛性:triu(QDLDL)与 flat(KLU)同迭代数、同解。
    #[cfg(feature = "qdldl")]
    #[test]
    fn triu_gn_ieee39_matches_flat() {
        use crate::basic::solver::QDLDLSolver;
        let mat = load_ieee39_mat();
        let ybus = &mat.y_bus;
        let (npv, npq) = (mat.npv, mat.npq);
        let sbus: Vec<Complex64> = mat.s_bus.iter().copied().collect();

        let mut d_flat = GnDriver::build(ybus, npv, npq, sbus.clone());
        let mut s_klu = KLUSolver::default();
        let mut v_flat: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let r_flat = d_flat.solve_gn(ybus, &mut s_klu, &mut v_flat, 1e-8, 100);

        let mut d_triu = GnTriuDriver::build(ybus, npv, npq, sbus);
        let mut s_qd = QDLDLSolver::default();
        let mut v_triu: Vec<Complex64> = mat.v_bus_init.iter().copied().collect();
        let r_triu = d_triu.solve_gn(ybus, &mut s_qd, &mut v_triu, 1e-8, 100);

        let dv = v_flat
            .iter()
            .zip(v_triu.iter())
            .fold(0.0f64, |m, (a, b)| m.max((a - b).norm()));
        println!(
            "IEEE39: flat(KLU) it={} | triu(QDLDL) it={} | max|ΔV|={dv:.3e} | nnz slim={} triu={}",
            r_flat.iterations, r_triu.iterations, d_flat.gn.nnz_slim, d_triu.triu.nnz_triu
        );
        assert!(r_flat.converged && r_triu.converged);
        assert_eq!(r_flat.iterations, r_triu.iterations, "迭代数不一致");
        assert!(dv < 1e-9, "解不一致");
    }
}
