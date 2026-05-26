use nalgebra_sparse::CscMatrix;

/// Solution returned by PIPS.
pub struct PipsResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub converged: bool,
    pub iterations: usize,
    pub lam_eq: Vec<f64>,   // equality multipliers
    pub mu_ineq: Vec<f64>,  // inequality multipliers
    pub mu_lower: Vec<f64>, // lower-bound multipliers
    pub mu_upper: Vec<f64>, // upper-bound multipliers
    pub message: String,
}

/// PIPS options
pub struct PipsOpt {
    pub feastol: f64,
    pub gradtol: f64,
    pub comptol: f64,
    pub costtol: f64,
    pub max_it: usize,
    pub cost_mult: f64,
}

impl Default for PipsOpt {
    fn default() -> Self {
        Self {
            feastol: 1e-6,
            gradtol: 1e-6,
            comptol: 1e-6,
            costtol: 1e-6,
            max_it: 150,
            cost_mult: 1.0,
        }
    }
}

/// Primal-Dual Interior-Point Solver (PIPS) for NLP:
///
///   min  f(x)            subject to:
///    x      g(x) = 0             (nonlinear equalities)
///          h(x) ≤ 0              (nonlinear inequalities)
///          xmin ≤ x ≤ xmax       (variable bounds)
///
/// `f_fcn`    : (x) → (f, df)
/// `gh_fcn`   : (x) → (h, g, dh, dg)   dh/dg are (nx × n_ineq/n_eq) transposed Jacobians
/// `hess_fcn` : (x, lam_eq, mu_ineq, cost_mult) → Lxx (nx × nx)
///
/// Port of MATPOWER's MIPS / pypower's pips.py.
pub fn pips<F, GH, H>(
    f_fcn: F,
    gh_fcn: GH,
    hess_fcn: H,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
) -> PipsResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    GH: Fn(&[f64]) -> (Vec<f64>, Vec<f64>, CscMatrix<f64>, CscMatrix<f64>),
    H: Fn(&[f64], &[f64], &[f64], f64) -> CscMatrix<f64>,
{
    const XI: f64 = 0.99995;
    const SIGMA: f64 = 0.1;
    const Z0: f64 = 1.0;
    const ALPHA_MIN: f64 = 1e-8;
    const MU_THRESHOLD: f64 = 1e-5;

    let nx = x0.len();
    let cm = opt.cost_mult;

    // Variable bounds become additional inequality constraints:
    //   x - xmax ≤ 0  (upper)
    //  -x + xmin ≤ 0  (lower)
    // Merged with nonlinear h(x) ≤ 0.
    // We keep them separate and handle via slack variables z, multipliers mu.

    // Extend AA with identity for variable bounds:
    //   AA = [I_nx; A_lin] where A_lin = nothing here (no linear constr)
    //   ll = [xmin; l_lin],  uu = [xmax; u_lin]
    // Split into:
    //   ieq: |uu-ll| ≤ eps  → equality
    //   ilt: ll=-inf, uu<inf → upper bound  → uu - x ≥ 0 → x - uu ≤ 0
    //   igt: ll>-inf, uu=inf → lower bound  → x - ll ≥ 0 → ll - x ≤ 0
    //   ibx: both finite     → both bounds

    let eps = f64::EPSILON;
    let mut ieq: Vec<usize> = Vec::new();
    let mut ilt: Vec<usize> = Vec::new();
    let mut igt: Vec<usize> = Vec::new();
    let mut ibx: Vec<usize> = Vec::new();

    for i in 0..nx {
        let lo = xmin[i];
        let hi = xmax[i];
        if (hi - lo).abs() <= eps {
            ieq.push(i);
        } else if lo <= -1e10 && hi < 1e10 {
            ilt.push(i);
        } else if lo > -1e10 && hi >= 1e10 {
            igt.push(i);
        } else if lo > -1e10 && hi < 1e10 {
            ibx.push(i);
        }
    }

    // bi = upper bounds on linear inequalities
    // Ai * x ≤ bi:
    //   for ilt: x_i ≤ xmax_i         → coeff +1, bound xmax
    //   for igt: -x_i ≤ -xmin_i       → coeff -1, bound -xmin
    //   for ibx: x_i ≤ xmax_i         → coeff +1
    //            -x_i ≤ -xmin_i
    let _n_lin_ineq = ilt.len() + igt.len() + 2 * ibx.len();
    let _n_lin_eq = ieq.len();

    // Build Ae (n_lin_eq × nx), be (n_lin_eq)
    // Build Ai (n_lin_ineq × nx), bi (n_lin_ineq)
    let (ai, bi_vec, ae, be_vec) =
        build_linear_constraints(nx, &ieq, &ilt, &igt, &ibx, &xmin, &xmax);

    // ── initial evaluation ───────────────────────────────────────────────────
    let mut x = x0.clone();
    let (f0_raw, df0) = f_fcn(&x);
    let mut f = f0_raw * cm;
    let mut df: Vec<f64> = df0.iter().map(|&v| v * cm).collect();

    let (hn, gn, dhn, dgn) = gh_fcn(&x);

    // Merge: h = [hn; Ai*x - bi], g = [gn; Ae*x - be]
    let (mut h, mut g, mut dh, mut dg) = merge_constraints(
        &x, &hn, &gn, &dhn, &dgn,
        &ai, &bi_vec, &ae, &be_vec,
    );

    let neq = g.len();
    let niq = h.len();
    let neqnln = gn.len();
    let niqnln = hn.len();

    // ── initialize multipliers and slacks ────────────────────────────────────
    let mut lam = vec![0.0f64; neq];
    let mut z: Vec<f64> = vec![Z0; niq];
    let mut mu: Vec<f64> = vec![Z0; niq];

    for k in 0..niq {
        if h[k] < -Z0 {
            z[k] = -h[k];
        }
    }
    // MATPOWER MIPS convention: mu[k] = Z0/z[k] so that z[k]*mu[k] = Z0 = 1.
    // The previous `if 1/z[k] > Z0` branch was never triggered (z[k] ≥ Z0 always),
    // leaving mu=1 for all slack constraints and inflating initial complementarity.
    for k in 0..niq {
        mu[k] = Z0 / z[k];
    }

    let mut gamma;

    // ── initial Lx ───────────────────────────────────────────────────────────
    let mut lx = df.clone();
    if let Some(dg_ref) = dg.as_ref() {
        // lx += dg * lam
        matvec_add_to(&mut lx, dg_ref, &lam);
    }
    if let Some(dh_ref) = dh.as_ref() {
        matvec_add_to(&mut lx, dh_ref, &mu);
    }

    let (mut feascond, mut gradcond, mut compcond, mut costcond) =
        convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f);

    let mut converged = feascond < opt.feastol
        && gradcond < opt.gradtol
        && compcond < opt.comptol
        && costcond < opt.costtol;

    let mut i = 0usize;
    let mut f0 = f;

    // ── Newton iteration ─────────────────────────────────────────────────────
    while !converged && i < opt.max_it {
        i += 1;

        let lam_for_hess = &lam[..neqnln];
        let mu_for_hess = &mu[..niqnln];
        let lxx = hess_fcn(&x, lam_for_hess, mu_for_hess, cm);

        // ── centering (MIPS-style fixed sigma) ───────────────────────────────
        let gap = if niq > 0 { z.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>() } else { 0.0 };
        gamma = if niq > 0 { SIGMA * gap / niq as f64 } else { 0.0 };

        let (dx, dlam_n, dz_n, dmu_n) =
            solve_kkt(&lxx, &lx, &dg, &dh, &g, &h, &z, &mu, gamma, nx, neq, niq);

        // step size control
        let alphap = step_size(&z, &dz_n, XI);
        let alphad = step_size(&mu, &dmu_n, XI);

        // per-iteration diagnostics
        {
            let (pd_k, _) = dmu_n.iter().enumerate()
                .filter(|&(_, d)| *d < 0.0)
                .map(|(k, d)| (k, mu[k] / (-d)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, 1.0));
            let (pp_k, _) = dz_n.iter().enumerate()
                .filter(|&(_, d)| *d < 0.0)
                .map(|(k, d)| (k, z[k] / (-d)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, 1.0));
            let pd_type = if pd_k < niqnln { "nl" } else { "bnd" };
            let pp_type = if pp_k < niqnln { "nl" } else { "bnd" };
            eprintln!(
                "  iter {:3}: ap={:.4e} ad={:.4e} | pp_k={}/{} z={:.3e} dz={:.3e} | pd_k={}/{} mu={:.3e} dmu={:.3e} g={:.3e}",
                i, alphap, alphad,
                pp_k, pp_type, z[pp_k], dz_n[pp_k],
                pd_k, pd_type, mu[pd_k], dmu_n[pd_k], gamma
            );
        }

        // update
        for i in 0..nx {
            x[i] += alphap * dx[i];
        }
        for i in 0..niq {
            z[i] += alphap * dz_n[i];
        }
        for i in 0..neq {
            lam[i] += alphad * dlam_n[i];
        }
        for i in 0..niq {
            mu[i] += alphad * dmu_n[i];
        }

        // re-evaluate
        let (f_new, df_new) = f_fcn(&x);
        f = f_new * cm;
        df = df_new.iter().map(|&v| v * cm).collect();

        let (hn_new, gn_new, dhn_new, dgn_new) = gh_fcn(&x);
        let (h_new, g_new, dh_new, dg_new) = merge_constraints(
            &x, &hn_new, &gn_new, &dhn_new, &dgn_new,
            &ai, &bi_vec, &ae, &be_vec,
        );
        h = h_new;
        g = g_new;
        dh = dh_new;
        dg = dg_new;

        lx = df.clone();
        if let Some(dg_ref) = dg.as_ref() {
            matvec_add_to(&mut lx, dg_ref, &lam);
        }
        if let Some(dh_ref) = dh.as_ref() {
            matvec_add_to(&mut lx, dh_ref, &mu);
        }

        let (fc, gc, cc, cc2) =
            convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f0);
        feascond = fc;
        gradcond = gc;
        compcond = cc;
        costcond = cc2;

        if feascond < opt.feastol
            && gradcond < opt.gradtol
            && compcond < opt.comptol
            && costcond < opt.costtol
        {
            converged = true;
        } else if x.iter().any(|v| v.is_nan()) {
            eprintln!("PIPS: NaN detected at iter {}", i);
            break;
        } else if alphap < ALPHA_MIN || alphad < ALPHA_MIN {
            eprintln!(
                "PIPS: step too small at iter {} (alphap={:.2e} alphad={:.2e}) \
                 feas={:.2e} grad={:.2e} comp={:.2e}",
                i, alphap, alphad, feascond, gradcond, compcond
            );
            break;
        }
        if i % 10 == 0 {
            eprintln!(
                "  iter {:3}: f={:.6} feas={:.2e} grad={:.2e} comp={:.2e} ap={:.4} ad={:.4}",
                i, f / cm, feascond, gradcond, compcond, alphap, alphad
            );
        }
        f0 = f;
    }

    // zero out non-binding inequality multipliers
    for k in 0..niq {
        if h[k] < -opt.feastol && mu[k] < MU_THRESHOLD {
            mu[k] = 0.0;
        }
    }

    let message = if converged {
        "Converged".to_string()
    } else {
        "Did not converge".to_string()
    };

    PipsResult {
        x: x.clone(),
        f: f / cm,
        converged,
        iterations: i,
        lam_eq: lam[..neqnln].to_vec(),
        mu_ineq: mu[..niqnln].to_vec(),
        mu_lower: vec![0.0; nx],
        mu_upper: vec![0.0; nx],
        message,
    }
}

// ── KKT system solver ─────────────────────────────────────────────────────────

fn solve_kkt(
    lxx: &CscMatrix<f64>,
    lx: &[f64],
    dg: &Option<CscMatrix<f64>>,
    dh: &Option<CscMatrix<f64>>,
    g: &[f64],
    h: &[f64],
    z: &[f64],
    mu: &[f64],
    gamma: f64,
    nx: usize,
    neq: usize,
    niq: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let extra = vec![0.0f64; niq];
    solve_kkt_with_extra(lxx, lx, dg, dh, g, h, z, mu, gamma, &extra, nx, neq, niq)
}

/// Solve the KKT system with an optional second-order correction term in w.
/// w[k] = (mu[k]*h[k] + gamma) / z[k] + extra_w[k]
fn solve_kkt_with_extra(
    lxx: &CscMatrix<f64>,
    lx: &[f64],
    dg: &Option<CscMatrix<f64>>,
    dh: &Option<CscMatrix<f64>>,
    g: &[f64],
    h: &[f64],
    z: &[f64],
    mu: &[f64],
    gamma: f64,
    extra_w: &[f64],
    nx: usize,
    neq: usize,
    niq: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Compute N = Lx + dh * w, w[k] = (mu[k]*h[k] + gamma) / z[k] + extra_w[k]
    let mut n_vec = lx.to_vec();
    if let Some(dh_ref) = dh {
        let w: Vec<f64> = (0..niq)
            .map(|k| (mu[k] * h[k] + gamma) / z[k] + extra_w[k])
            .collect();
        matvec_add_to(&mut n_vec, dh_ref, &w);
    }

    // Build augmented matrix M
    // M = Lxx + dh * diag(mu/z) * dh^T
    let m_mat = if let Some(dh_ref) = dh {
        let mu_over_z: Vec<f64> = (0..niq).map(|k| mu[k] / z[k]).collect();
        // M = Lxx + dh * diag(mu/z) * dh^T  where dh is (nx × niq)
        let dh_cs = col_scale_f64(dh_ref, &mu_over_z);
        // dh_cs * dh^T: (nx × niq) * (niq × nx) = nx × nx
        let dh_t = transpose_f64(dh_ref);
        let prod = spgemm_f64(&dh_cs, &dh_t);
        f64_add_sparse(lxx, &prod)
    } else {
        lxx.clone()
    };

    // Build and solve the saddle-point system
    if neq == 0 {
        // No equality constraints — just solve M*dx = -N
        let dx = solve_linear(&m_mat, &n_vec.iter().map(|&v| -v).collect::<Vec<_>>());
        let dz: Vec<f64> = if let Some(dh_ref) = dh {
            // dz = -h - z - dh^T * dx
            let mut dz: Vec<f64> = (0..niq).map(|k| -h[k] - z[k]).collect();
            for k in 0..niq {
                for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k + 1] {
                    dz[k] -= dh_ref.values()[idx] * dx[dh_ref.row_indices()[idx]];
                }
            }
            dz
        } else {
            vec![]
        };

        let dmu: Vec<f64> = (0..niq)
            .map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k] + extra_w[k])
            .collect();
        (dx, vec![], dz, dmu)
    } else {
        // Saddle-point system [M dg; dg^T 0] * [dx; dlam] = [-N; -g]
        let dim = nx + neq;
        let rhs: Vec<f64> = n_vec.iter().map(|&v| -v).chain(g.iter().map(|&v| -v)).collect();
        let ab = build_saddle_point(&m_mat, dg, nx, neq);
        let sol = solve_linear(&ab, &rhs);
        let dx = sol[..nx].to_vec();
        let dlam = sol[nx..dim].to_vec();

        let dz: Vec<f64> = if let Some(dh_ref) = dh {
            // dz = -h - z - dh^T * dx
            let mut dz: Vec<f64> = (0..niq).map(|k| -h[k] - z[k]).collect();
            // dh is (nx × niq): dh^T * dx gives niq values
            // dh^T[k, :] = dh[:, k]
            for k in 0..niq {
                for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k + 1] {
                    let var = dh_ref.row_indices()[idx];
                    dz[k] -= dh_ref.values()[idx] * dx[var];
                }
            }
            dz
        } else {
            (0..niq).map(|k| -h[k] - z[k]).collect()
        };

        let dmu: Vec<f64> = (0..niq)
            .map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k] + extra_w[k])
            .collect();

        (dx, dlam, dz, dmu)
    }
}

// ── sparse linear algebra helpers ────────────────────────────────────────────

fn col_scale_f64(a: &CscMatrix<f64>, scale: &[f64]) -> CscMatrix<f64> {
    let cp = a.col_offsets();
    let ri = a.row_indices();
    let vals: Vec<f64> = a
        .values()
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let col = cp.partition_point(|&o| o <= idx) - 1;
            v * scale[col]
        })
        .collect();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), cp.to_vec(), ri.to_vec(), vals).unwrap()
}

fn transpose_f64(a: &CscMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let nnz = a.nnz();
    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();

    let mut row_counts = vec![0usize; m];
    for &r in a_ri {
        row_counts[r] += 1;
    }
    let mut t_cp = vec![0usize; m + 1];
    for i in 0..m {
        t_cp[i + 1] = t_cp[i] + row_counts[i];
    }
    let mut t_ri = vec![0usize; nnz];
    let mut t_v = vec![0.0f64; nnz];
    let mut pos = t_cp[..m].to_vec();

    let mut col_of = vec![0usize; nnz];
    for j in 0..n {
        for idx in a_cp[j]..a_cp[j + 1] {
            col_of[idx] = j;
        }
    }
    for idx in 0..nnz {
        let r = a_ri[idx];
        let p = pos[r];
        t_ri[p] = col_of[idx];
        t_v[p] = a_v[idx];
        pos[r] += 1;
    }
    for i in 0..m {
        let s = t_cp[i];
        let e = t_cp[i + 1];
        let mut pairs: Vec<(usize, f64)> = (s..e).map(|p| (t_ri[p], t_v[p])).collect();
        pairs.sort_unstable_by_key(|&(c, _)| c);
        for (p, (c, v)) in (s..e).zip(pairs) {
            t_ri[p] = c;
            t_v[p] = v;
        }
    }
    CscMatrix::try_from_csc_data(n, m, t_cp, t_ri, t_v).unwrap()
}

fn spgemm_f64(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows();
    let n = b.ncols();
    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();
    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();

    let mut acc = vec![0.0f64; m];
    let mut visited = vec![false; m];
    let mut col_nz: Vec<usize> = Vec::new();
    let mut c_cp = vec![0usize; n + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    for j in 0..n {
        col_nz.clear();
        for idx_b in b_cp[j]..b_cp[j + 1] {
            let kk = b_ri[idx_b];
            let b_kj = b_v[idx_b];
            for idx_a in a_cp[kk]..a_cp[kk + 1] {
                let i = a_ri[idx_a];
                if !visited[i] {
                    visited[i] = true;
                    col_nz.push(i);
                }
                acc[i] += a_v[idx_a] * b_kj;
            }
        }
        col_nz.sort_unstable();
        for &i in &col_nz {
            c_ri.push(i);
            c_v.push(acc[i]);
            acc[i] = 0.0;
            visited[i] = false;
        }
        c_cp[j + 1] = c_ri.len();
    }
    CscMatrix::try_from_csc_data(m, n, c_cp, c_ri, c_v).unwrap()
}

fn f64_add_sparse(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let nb = a.ncols();
    let a_cp = a.col_offsets();
    let a_ri = a.row_indices();
    let a_v = a.values();
    let b_cp = b.col_offsets();
    let b_ri = b.row_indices();
    let b_v = b.values();
    let mut c_cp = vec![0usize; nb + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    for j in 0..nb {
        let mut ia = a_cp[j];
        let mut ib = b_cp[j];
        let ea = a_cp[j + 1];
        let eb = b_cp[j + 1];
        while ia < ea || ib < eb {
            let ra = if ia < ea { a_ri[ia] } else { usize::MAX };
            let rb = if ib < eb { b_ri[ib] } else { usize::MAX };
            if ra < rb {
                c_ri.push(ra); c_v.push(a_v[ia]); ia += 1;
            } else if rb < ra {
                c_ri.push(rb); c_v.push(b_v[ib]); ib += 1;
            } else {
                c_ri.push(ra); c_v.push(a_v[ia] + b_v[ib]); ia += 1; ib += 1;
            }
        }
        c_cp[j + 1] = c_ri.len();
    }
    CscMatrix::try_from_csc_data(a.nrows(), nb, c_cp, c_ri, c_v).unwrap()
}

fn build_saddle_point(
    m: &CscMatrix<f64>,
    dg: &Option<CscMatrix<f64>>,
    nx: usize,
    neq: usize,
) -> CscMatrix<f64> {
    // [M  dg ] (nx+neq) × (nx+neq)
    // [dg^T 0]
    let dim = nx + neq;
    let mut c_cp = vec![0usize; dim + 1];
    let mut c_ri: Vec<usize> = Vec::new();
    let mut c_v: Vec<f64> = Vec::new();

    // First nx columns: M col j + dg^T row j (= dg col j transposed)
    let m_cp = m.col_offsets();
    let m_ri = m.row_indices();
    let m_v = m.values();

    for j in 0..nx {
        // M[:, j]
        for idx in m_cp[j]..m_cp[j + 1] {
            c_ri.push(m_ri[idx]);
            c_v.push(m_v[idx]);
        }
        // dg^T col j = dg row j: entries from dg (nx × neq) where row_index == j
        if let Some(dg_ref) = dg {
            let dg_cp = dg_ref.col_offsets();
            let dg_ri = dg_ref.row_indices();
            let dg_v = dg_ref.values();
            for eq in 0..neq {
                for idx in dg_cp[eq]..dg_cp[eq + 1] {
                    if dg_ri[idx] == j {
                        c_ri.push(nx + eq); // row offset by nx
                        c_v.push(dg_v[idx]);
                        break;
                    }
                }
            }
        }
        c_cp[j + 1] = c_ri.len();
    }

    // Next neq columns: dg col eq (rows 0..nx), then zeros (rows nx..nx+neq)
    if let Some(dg_ref) = dg {
        let dg_cp = dg_ref.col_offsets();
        let dg_ri = dg_ref.row_indices();
        let dg_v = dg_ref.values();
        for eq in 0..neq {
            for idx in dg_cp[eq]..dg_cp[eq + 1] {
                c_ri.push(dg_ri[idx]);
                c_v.push(dg_v[idx]);
            }
            c_cp[nx + eq + 1] = c_ri.len();
        }
    }

    CscMatrix::try_from_csc_data(dim, dim, c_cp, c_ri, c_v).unwrap()
}

/// Non-symmetric iterative equilibration. Returns (r, c) where the scaled matrix is
/// R*A*C (R = diag(r), C = diag(c)). Scales rows and columns independently so that
/// all row and column maxima converge to 1. This correctly handles saddle-point systems
/// where the M block (large entries) and dg block (small entries) need independent scaling.
fn equilibrate(a: &CscMatrix<f64>) -> (Vec<f64>, Vec<f64>) {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let cp = a.col_offsets();
    let ri = a.row_indices();
    let v = a.values();

    let mut r = vec![1.0f64; nrows];
    let mut c = vec![1.0f64; ncols];

    for _ in 0..10 {
        // Row scaling: scale r so each row max → 1
        let mut row_max = vec![0.0f64; nrows];
        for j in 0..ncols {
            for idx in cp[j]..cp[j + 1] {
                let abs_v = (v[idx] * r[ri[idx]] * c[j]).abs();
                if abs_v > row_max[ri[idx]] { row_max[ri[idx]] = abs_v; }
            }
        }
        for i in 0..nrows {
            if row_max[i] > 1e-300 { r[i] /= row_max[i].sqrt(); }
        }
        // Column scaling: scale c so each col max → 1
        let mut col_max = vec![0.0f64; ncols];
        for j in 0..ncols {
            for idx in cp[j]..cp[j + 1] {
                let abs_v = (v[idx] * r[ri[idx]] * c[j]).abs();
                if abs_v > col_max[j] { col_max[j] = abs_v; }
            }
        }
        for j in 0..ncols {
            if col_max[j] > 1e-300 { c[j] /= col_max[j].sqrt(); }
        }
    }
    (r, c)
}

/// Sparse direct solver with non-symmetric iterative equilibration to reduce condition
/// number, then rsparse LU (falls back to dense if it fails).
/// Solve: A*x = b  →  (R*A*C)*(C^{-1}*x) = R*b  →  recover x[j] = c[j]*y[j].
fn solve_linear(a: &CscMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let (r, c) = equilibrate(a);

    let cp = a.col_offsets();
    let ri = a.row_indices();
    let v = a.values();

    // Scale values: A_scaled[i,j] = r[i] * A[i,j] * c[j]
    let mut ax: Vec<f64> = v.to_vec();
    for j in 0..n {
        let cj = c[j];
        for idx in cp[j]..cp[j + 1] {
            ax[idx] = r[ri[idx]] * v[idx] * cj;
        }
    }
    let rhs_scaled: Vec<f64> = (0..n).map(|i| r[i] * b[i]).collect();

    let y = {
        let mut solved = false;
        let mut y = vec![0.0f64; n];

        #[cfg(feature = "klu")]
        if !solved {
            use crate::basic::solver::{KLUSolver, Solve};
            let mut solver = KLUSolver::default();
            let mut ap = cp.to_vec();
            let mut ai = ri.to_vec();
            let mut ax_s = ax.clone();
            let mut rhs = rhs_scaled.clone();
            if solver.solve(&mut ap, &mut ai, &mut ax_s, &mut rhs, n).is_ok() {
                y = rhs;
                solved = true;
            }
        }

        #[cfg(feature = "rsparse")]
        if !solved {
            use crate::basic::solver::{RSparseSolver, Solve};
            let mut solver = RSparseSolver::default();
            let mut ap = cp.to_vec();
            let mut ai = ri.to_vec();
            let mut ax_s = ax.clone();
            let mut rhs = rhs_scaled.clone();
            if solver.solve(&mut ap, &mut ai, &mut ax_s, &mut rhs, n).is_ok() {
                y = rhs;
                solved = true;
            }
        }

        if !solved {
            let a_scaled = CscMatrix::try_from_csc_data(
                a.nrows(), a.ncols(), cp.to_vec(), ri.to_vec(), ax.clone()
            ).unwrap();
            y = dense_solve_fallback(&a_scaled, &rhs_scaled);
        }
        y
    };

    let x_sol: Vec<f64> = y.iter().enumerate().map(|(j, &yj)| c[j] * yj).collect();

    // Debug: check backward error of the solve (original unscaled system)
    if cfg!(debug_assertions) {
        let mut max_res = 0.0f64;
        let mut res = b.to_vec();
        for j in 0..n {
            for idx in cp[j]..cp[j + 1] {
                res[ri[idx]] -= v[idx] * x_sol[j];
            }
        }
        for &rv in &res { max_res = max_res.max(rv.abs()); }
        let b_norm = b.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
        if max_res > 1e-6 * b_norm {
            eprintln!("  [solve] residual={:.2e} b_norm={:.2e} ratio={:.2e}",
                max_res, b_norm, max_res / b_norm);
        }
    }

    x_sol
}

fn dense_solve_fallback(a: &CscMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Convert to dense
    let mut mat = vec![0.0f64; n * n];
    let cp = a.col_offsets();
    let ri = a.row_indices();
    let v = a.values();
    for j in 0..n {
        for idx in cp[j]..cp[j + 1] {
            mat[ri[idx] * n + j] = v[idx];
        }
    }
    let mut rhs = b.to_vec();
    // LU decomposition with partial pivoting
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // find pivot
        let mut max_val = mat[perm[k] * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = mat[perm[i] * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        perm.swap(k, max_row);
        let pk = perm[k];
        let pivot = mat[pk * n + k];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for i in (k + 1)..n {
            let pi = perm[i];
            let factor = mat[pi * n + k] / pivot;
            mat[pi * n + k] = factor;
            for j in (k + 1)..n {
                mat[pi * n + j] -= factor * mat[pk * n + j];
            }
            rhs[pi] -= factor * rhs[pk];
        }
    }
    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let pi = perm[i];
        let mut s = rhs[pi];
        for j in (i + 1)..n {
            s -= mat[pi * n + j] * x[j];
        }
        x[i] = s / mat[pi * n + i];
    }
    x
}

// ── linear constraint helpers ─────────────────────────────────────────────────

fn build_linear_constraints(
    nx: usize,
    ieq: &[usize],
    ilt: &[usize],
    igt: &[usize],
    ibx: &[usize],
    xmin: &[f64],
    xmax: &[f64],
) -> (Option<CscMatrix<f64>>, Vec<f64>, Option<CscMatrix<f64>>, Vec<f64>) {
    // Ai (n_ineq × nx): rows for ilt (+1), igt (-1), ibx (+1 then -1)
    let n_ineq = ilt.len() + igt.len() + 2 * ibx.len();
    let n_eq = ieq.len();

    let ai = if n_ineq > 0 {
        let mut row = 0usize;
        let mut entries: Vec<(usize, usize, f64)> = Vec::new();
        for &i in ilt { entries.push((row, i,  1.0)); row += 1; }
        for &i in igt { entries.push((row, i, -1.0)); row += 1; }
        for &i in ibx { entries.push((row, i,  1.0)); row += 1; }
        for &i in ibx { entries.push((row, i, -1.0)); row += 1; }
        Some(coo_to_csc_f64(n_ineq, nx, &entries))
    } else {
        None
    };

    let bi: Vec<f64> = ilt
        .iter()
        .map(|&i| xmax[i])
        .chain(igt.iter().map(|&i| -xmin[i]))
        .chain(ibx.iter().map(|&i| xmax[i]))
        .chain(ibx.iter().map(|&i| -xmin[i]))
        .collect();

    let ae = if n_eq > 0 {
        let entries: Vec<(usize, usize, f64)> =
            ieq.iter().enumerate().map(|(r, &i)| (r, i, 1.0)).collect();
        Some(coo_to_csc_f64(n_eq, nx, &entries))
    } else {
        None
    };

    let be: Vec<f64> = ieq.iter().map(|&i| xmax[i]).collect();

    (ai, bi, ae, be)
}

fn merge_constraints(
    x: &[f64],
    hn: &[f64],
    gn: &[f64],
    dhn: &CscMatrix<f64>,  // nx × niqnln
    dgn: &CscMatrix<f64>,  // nx × neqnln
    ai: &Option<CscMatrix<f64>>,  // n_lin_ineq × nx
    bi: &[f64],
    ae: &Option<CscMatrix<f64>>,  // n_lin_eq × nx
    be: &[f64],
) -> (Vec<f64>, Vec<f64>, Option<CscMatrix<f64>>, Option<CscMatrix<f64>>) {
    let (h, dh) = match ai {
        Some(ai_ref) => {
            let ax = spmv_f64(ai_ref, x);
            let h_lin: Vec<f64> = ax.iter().zip(bi.iter()).map(|(a, b)| a - b).collect();
            let h = [hn, h_lin.as_slice()].concat();
            let ait = transpose_f64(ai_ref);
            (h, Some(hstack_csc(dhn, &ait)))
        }
        None => (hn.to_vec(), Some(dhn.clone())),
    };

    let (g, dg) = match ae {
        Some(ae_ref) => {
            let ax = spmv_f64(ae_ref, x);
            let g_lin: Vec<f64> = ax.iter().zip(be.iter()).map(|(a, b)| a - b).collect();
            let g = [gn, g_lin.as_slice()].concat();
            let aet = transpose_f64(ae_ref);
            (g, Some(hstack_csc(dgn, &aet)))
        }
        None => (gn.to_vec(), Some(dgn.clone())),
    };

    (h, g, dh, dg)
}

fn hstack_csc(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    debug_assert_eq!(a.nrows(), b.nrows());
    let m = a.nrows();
    let na = a.ncols();
    let nb = b.ncols();
    let n = na + nb;
    let nnz_a = a.nnz();

    let a_cp = a.col_offsets();
    let b_cp = b.col_offsets();

    let mut c_cp = vec![0usize; n + 1];
    c_cp[..na + 1].copy_from_slice(&a_cp[..na + 1]);
    for j in 0..nb {
        c_cp[na + j + 1] = nnz_a + b_cp[j + 1];
    }

    let c_ri: Vec<usize> = [a.row_indices(), b.row_indices()].concat();
    let c_v: Vec<f64> = [a.values(), b.values()].concat();

    CscMatrix::try_from_csc_data(m, n, c_cp, c_ri, c_v).unwrap()
}

fn coo_to_csc_f64(nrows: usize, ncols: usize, entries: &[(usize, usize, f64)]) -> CscMatrix<f64> {
    let mut sorted = entries.to_vec();
    sorted.sort_unstable_by_key(|&(r, c, _)| (c, r));
    let mut c_cp = vec![0usize; ncols + 1];
    let mut c_ri: Vec<usize> = Vec::with_capacity(sorted.len());
    let mut c_v: Vec<f64> = Vec::with_capacity(sorted.len());
    for &(r, c, v) in &sorted {
        c_cp[c + 1] += 1;
        c_ri.push(r);
        c_v.push(v);
    }
    for j in 0..ncols {
        c_cp[j + 1] += c_cp[j];
    }
    CscMatrix::try_from_csc_data(nrows, ncols, c_cp, c_ri, c_v).unwrap()
}

fn spmv_f64(a: &CscMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let mut y = vec![0.0f64; m];
    let cp = a.col_offsets();
    let ri = a.row_indices();
    let v = a.values();
    for j in 0..a.ncols() {
        for idx in cp[j]..cp[j + 1] {
            y[ri[idx]] += v[idx] * x[j];
        }
    }
    y
}

// ── convergence and step helpers ─────────────────────────────────────────────

fn convergence_measures(
    g: &[f64],
    h: &[f64],
    lx: &[f64],
    lam: &[f64],
    mu: &[f64],
    z: &[f64],
    x: &[f64],
    f: f64,
    f0: f64,
) -> (f64, f64, f64, f64) {
    let gnorm = inf_norm(g);
    let maxh = h.iter().cloned().fold(0.0f64, f64::max);
    let xnorm = inf_norm(x);
    let znorm = inf_norm(z);
    let lam_norm = inf_norm(lam);
    let mu_norm = inf_norm(mu);

    let feascond = f64::max(gnorm, maxh) / (1.0 + f64::max(xnorm, znorm));
    let gradcond = inf_norm(lx) / (1.0 + f64::max(lam_norm, mu_norm));
    let compcond = if z.is_empty() {
        0.0
    } else {
        z.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>() / (1.0 + xnorm)
    };
    let costcond = (f - f0).abs() / (1.0 + f0.abs());
    (feascond, gradcond, compcond, costcond)
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()))
}

fn step_size(z: &[f64], dz: &[f64], xi: f64) -> f64 {
    let mut alpha = 1.0f64;
    for k in 0..z.len() {
        if dz[k] < 0.0 {
            alpha = alpha.min(xi * z[k] / (-dz[k]));
        }
    }
    alpha
}

/// Multiply dg (nx × neq) by vec (neq) and add to out (nx).
fn matvec_add_to(out: &mut [f64], dg: &CscMatrix<f64>, vec: &[f64]) {
    let cp = dg.col_offsets();
    let ri = dg.row_indices();
    let v = dg.values();
    for j in 0..dg.ncols() {
        let vj = vec[j];
        if vj == 0.0 {
            continue;
        }
        for idx in cp[j]..cp[j + 1] {
            out[ri[idx]] += v[idx] * vj;
        }
    }
}
