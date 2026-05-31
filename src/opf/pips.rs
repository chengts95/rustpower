use nalgebra_sparse::{CscMatrix, CooMatrix};

pub struct PipsResult {
    pub x: Vec<f64>, pub f: f64, pub converged: bool, pub iterations: usize,
    pub lam_eq: Vec<f64>, pub mu_ineq: Vec<f64>, pub mu_lower: Vec<f64>, pub mu_upper: Vec<f64>,
    pub message: String,
    /// Per-stage wall-clock totals over all iterations (for ablation breakdown plots).
    pub timing: PipsTiming,
}

/// Structured per-stage timing breakdown (sums over the whole solve).
#[derive(Clone, Copy, Default)]
pub struct PipsTiming {
    pub hess: std::time::Duration,
    pub gh: std::time::Duration,
    pub kkt: std::time::Duration,
    pub solve_sym: std::time::Duration,
    pub solve_num: std::time::Duration,
}

pub struct PipsOpt {
    pub feastol: f64, pub gradtol: f64, pub comptol: f64, pub costtol: f64,
    pub max_it: usize, pub cost_mult: f64, pub merged_slacks: bool,
}

impl Default for PipsOpt {
    fn default() -> Self {
        Self { feastol: 1e-6, gradtol: 1e-6, comptol: 1e-6, costtol: 1e-6, max_it: 150, cost_mult: 1.0, merged_slacks: false }
    }
}

pub fn pips<F, GH, H>(f_fcn: F, gh_fcn: GH, mut hess_fcn: H, x0: Vec<f64>, xmin: Vec<f64>, xmax: Vec<f64>, opt: PipsOpt) -> PipsResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    GH: Fn(&[f64]) -> (Vec<f64>, Vec<f64>, CscMatrix<f64>, CscMatrix<f64>),
    H: FnMut(&[f64], &[f64], &[f64], &[f64], f64) -> CscMatrix<f64>,
{
    let mut solver = crate::basic::solver::DefaultSolver::default();
    pips_with_solver(f_fcn, gh_fcn, &mut hess_fcn, x0, xmin, xmax, opt, &mut solver, None)
}

pub fn pips_with_solver<F, GH, H, S>(
    f_fcn: F, gh_fcn: GH, mut hess_fcn: H, x0: Vec<f64>, xmin: Vec<f64>, xmax: Vec<f64>, opt: PipsOpt, solver: &mut S,
    v5: Option<&crate::new_opf::v5_kkt::KKTSymbolicV5>,
) -> PipsResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    GH: Fn(&[f64]) -> (Vec<f64>, Vec<f64>, CscMatrix<f64>, CscMatrix<f64>),
    H: FnMut(&[f64], &[f64], &[f64], &[f64], f64) -> CscMatrix<f64>,
    S: crate::basic::solver::Solve,
{
    const XI: f64 = 0.99995; const SIGMA: f64 = 0.1; const Z0: f64 = 1.0;
    const ALPHA_MIN: f64 = 1e-8;

    let nx = x0.len(); let cm = opt.cost_mult;
    let eps = f64::EPSILON;
    let mut ieq = Vec::new(); let mut ilt = Vec::new(); let mut igt = Vec::new(); let mut ibx = Vec::new();
    for i in 0..nx {
        let lo = xmin[i]; let hi = xmax[i];
        if (hi - lo).abs() <= eps { ieq.push(i); }
        else if lo <= -1e10 && hi < 1e10 { ilt.push(i); }
        else if lo > -1e10 && hi >= 1e10 { igt.push(i); }
        else if lo > -1e10 && hi < 1e10 { ibx.push(i); }
    }
    let (ai, bi_vec, ae, be_vec) = build_linear_constraints(nx, &ieq, &ilt, &igt, &ibx, &xmin, &xmax);

    let mut x = x0.clone();
    let (f0_raw, df0) = f_fcn(&x);
    let mut f = f0_raw * cm;
    let mut df: Vec<f64> = df0.iter().map(|&v| v * cm).collect();

    let (hn, gn, dhn, dgn) = gh_fcn(&x);
    let (mut h, mut g, mut dh, mut dg) = merge_constraints(&x, &hn, &gn, &dhn, &dgn, &ai, &bi_vec, &ae, &be_vec);

    let neq = g.len(); let niq = h.len();
    let neqnln = gn.len(); let niqnln = hn.len();

    let mut lam = vec![0.0f64; neq];
    let mut z = vec![Z0; niq]; let mut mu = vec![Z0; niq];
    for k in 0..niq { if h[k] < -Z0 { z[k] = -h[k]; } mu[k] = Z0 / z[k]; }

    let mut lx = df.clone();
    if let Some(ref dg_ref) = dg { matvec_add_to(&mut lx, dg_ref, &lam); }
    if let Some(ref dh_ref) = dh { matvec_add_to(&mut lx, dh_ref, &mu); }

    let (mut feascond, mut gradcond, mut compcond, mut costcond) = convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f);
    let mut converged = feascond < opt.feastol && gradcond < opt.gradtol && compcond < opt.comptol && costcond < opt.costtol;
    let mut i = 0usize; let mut f0 = f;

    let mut total_hess = std::time::Duration::ZERO;
    let mut total_kkt = std::time::Duration::ZERO;
    let mut total_solve_sym = std::time::Duration::ZERO;
    let mut total_solve_num = std::time::Duration::ZERO;
    let mut total_gh = std::time::Duration::ZERO;

    while !converged && i < opt.max_it {
        i += 1;
        let t_start = std::time::Instant::now();
        let lxx = hess_fcn(&x, &lam[..neqnln], &mu[..niqnln], &z[..niqnln], cm);
        total_hess += t_start.elapsed();

        let gap = if niq > 0 { z.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>() } else { 0.0 };
        let gamma = if niq > 0 { SIGMA * gap / niq as f64 } else { 0.0 };

        let mut dt_kkt = std::time::Duration::ZERO;
        let mut dt_solve = std::time::Duration::ZERO;
        let (dx, dlam_n, dz_n, dmu_n) = solve_kkt_timed(&lxx, &lx, &dg, &dh, &g, &h, &z, &mu, gamma, nx, neq, niq, niqnln, solver, opt.merged_slacks, v5, &mut dt_kkt, &mut dt_solve);
        total_kkt += dt_kkt;
        if i == 1 { total_solve_sym += dt_solve; } else { total_solve_num += dt_solve; }

        let alphap = step_size(&z, &dz_n, 0.99995);
        let alphad = step_size(&mu, &dmu_n, 0.99995);

        for j in 0..nx { x[j] += alphap * dx[j]; }
        for j in 0..niq { z[j] += alphap * dz_n[j]; }
        for j in 0..neq { lam[j] += alphad * dlam_n[j]; }
        for j in 0..niq { mu[j] += alphad * dmu_n[j]; }

        let t_gh_start = std::time::Instant::now();
        let (f_new, df_new) = f_fcn(&x);
        f = f_new * cm;
        df = df_new.iter().map(|&v| v * cm).collect();

        let (hn_new, gn_new, dhn_new, dgn_new) = gh_fcn(&x);
        let (h_new, g_new, dh_new, dg_new) = merge_constraints(&x, &hn_new, &gn_new, &dhn_new, &dgn_new, &ai, &bi_vec, &ae, &be_vec);
        h = h_new; g = g_new; dh = dh_new; dg = dg_new;

        lx = df.clone();
        if let Some(ref dg_ref) = dg { matvec_add_to(&mut lx, dg_ref, &lam); }
        if let Some(ref dh_ref) = dh { matvec_add_to(&mut lx, dh_ref, &mu); }
        total_gh += t_gh_start.elapsed();

        let (fc, gc, cc, cc2) = convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f0);
        feascond = fc; gradcond = gc; compcond = cc; costcond = cc2;

        if feascond < opt.feastol && gradcond < opt.gradtol && compcond < opt.comptol && costcond < opt.costtol {
            converged = true;
        } else if x.iter().any(|v| v.is_nan()) || alphap < 1e-8 || alphad < 1e-8 {
            break;
        }
        f0 = f;
    }

    if i > 0 {
        eprintln!("\nPIPS ({} iters): Hess: {:?} G/H: {:?} KKT: {:?} Solv(Sym): {:?} Solv(Num): {:?}", i, total_hess, total_gh, total_kkt, total_solve_sym, total_solve_num);
    }

    PipsResult {
        x, f: f / cm, converged, iterations: i,
        lam_eq: lam[..neqnln].to_vec(), mu_ineq: mu[..niqnln].to_vec(),
        mu_lower: vec![0.0; nx], mu_upper: vec![0.0; nx],
        message: if converged { "Converged".to_string() } else { "Failed".to_string() },
        timing: PipsTiming { hess: total_hess, gh: total_gh, kkt: total_kkt, solve_sym: total_solve_sym, solve_num: total_solve_num },
    }
}

fn solve_kkt_timed<S: crate::basic::solver::Solve>(
    lxx: &CscMatrix<f64>, lx: &[f64], dg: &Option<CscMatrix<f64>>, dh: &Option<CscMatrix<f64>>,
    g: &[f64], h: &[f64], z: &[f64], mu: &[f64], gamma: f64,
    nx: usize, neq: usize, niq: usize, niqnln: usize, solver: &mut S, merged_slacks: bool,
    v5: Option<&crate::new_opf::v5_kkt::KKTSymbolicV5>,
    total_kkt: &mut std::time::Duration, total_solve: &mut std::time::Duration,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t_kkt = std::time::Instant::now();
    let mut n_vec = lx.to_vec();
    if let Some(dh_ref) = dh {
        let w: Vec<f64> = (0..niq).map(|k| (mu[k] * h[k] + gamma) / z[k]).collect();
        matvec_add_to(&mut n_vec, dh_ref, &w);
    }

    let ab = if neq > 0 {
        if lxx.nrows() == nx + neq {
            Some(lxx.clone())
        } else {
            let m_mat = if let Some(dh_ref) = dh {
                if merged_slacks {
                    let lxx_mod = lxx.clone();
                    let mut diag_vals = vec![0.0; nx];
                    for k in niqnln..niq {
                        let weight = mu[k] / z[k];
                        for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k+1] {
                            let r = dh_ref.row_indices()[idx];
                            let v = dh_ref.values()[idx];
                            diag_vals[r] += weight * v * v;
                        }
                    }
                    let mut lxx_v = lxx_mod.values().to_vec();
                    for j in 0..nx {
                        if diag_vals[j] == 0.0 { continue; }
                        let s = lxx_mod.col_offsets()[j]; let e = lxx_mod.col_offsets()[j+1];
                        if let Ok(pos) = lxx_mod.row_indices()[s..e].binary_search(&j) { lxx_v[s + pos] += diag_vals[j]; }
                    }
                    CscMatrix::try_from_csc_data(lxx.nrows(), lxx.ncols(), lxx_mod.col_offsets().to_vec(), lxx_mod.row_indices().to_vec(), lxx_v).unwrap()
                } else {
                    let mu_over_z: Vec<f64> = (0..niq).map(|k| mu[k] / z[k]).collect();
                    let dh_cs = col_scale_f64(dh_ref, &mu_over_z);
                    let prod = spgemm_f64(&dh_cs, &transpose_f64(dh_ref));
                    f64_add_sparse(lxx, &prod)
                }
            } else { lxx.clone() };
            let kkt = if let Some(v5c) = v5 {
                // V5 streaming KKT fill (structure == build_saddle_point, proven byte-exact)
                let dg_ref = dg.as_ref().expect("v5 KKT path requires equality dg");
                let dgt = dg_ref.transpose();
                let mut vals = vec![0.0f64; v5c.row_idx.len()];
                v5c.fill_from_merged(
                    m_mat.col_offsets(), m_mat.values(),
                    dg_ref.col_offsets(), dg_ref.values(),
                    dgt.col_offsets(), dgt.values(),
                    &mut vals,
                );
                CscMatrix::try_from_csc_data(v5c.dim, v5c.dim, v5c.col_ptrs.clone(), v5c.row_idx.clone(), vals).unwrap()
            } else {
                build_saddle_point(&m_mat, dg, nx, neq)
            };
            Some(kkt)
        }
    } else { None };
    *total_kkt += t_kkt.elapsed();

    let t_solve = std::time::Instant::now();
    let res = if neq == 0 {
        let m_mat = if let Some(dh_ref) = dh {
            let mu_over_z: Vec<f64> = (0..niq).map(|k| mu[k] / z[k]).collect();
            let dh_cs = col_scale_f64(dh_ref, &mu_over_z);
            let prod = spgemm_f64(&dh_cs, &transpose_f64(dh_ref));
            f64_add_sparse(lxx, &prod)
        } else { lxx.clone() };
        let mut rhs = n_vec.iter().map(|&v| -v).collect::<Vec<_>>();
        let (mut ap, mut ai, mut ax) = (m_mat.col_offsets().to_vec(), m_mat.row_indices().to_vec(), m_mat.values().to_vec());
        solver.solve(&mut ap, &mut ai, &mut ax, &mut rhs, nx).unwrap();
        let dx = rhs;
        let dz = if let Some(dh_ref) = dh {
            let mut tmp = (0..niq).map(|k| -h[k] - z[k]).collect::<Vec<_>>();
            for k in 0..niq { for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k+1] { tmp[k] -= dh_ref.values()[idx] * dx[dh_ref.row_indices()[idx]]; } }
            tmp
        } else { vec![] };
        let dmu = (0..niq).map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k]).collect::<Vec<_>>();
        (dx, vec![], dz, dmu)
    } else {
        let ab_ref = ab.as_ref().unwrap();
        let mut rhs = n_vec.iter().map(|&v| -v).chain(g.iter().map(|&v| -v)).collect::<Vec<_>>();
        let (mut ap, mut ai, mut ax) = (ab_ref.col_offsets().to_vec(), ab_ref.row_indices().to_vec(), ab_ref.values().to_vec());
        solver.solve(&mut ap, &mut ai, &mut ax, &mut rhs, nx + neq).unwrap();
        let dx = rhs[..nx].to_vec(); let dlam = rhs[nx..].to_vec();
        let dz = if let Some(dh_ref) = dh {
            let mut tmp = (0..niq).map(|k| -h[k] - z[k]).collect::<Vec<_>>();
            for k in 0..niq { for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k+1] { tmp[k] -= dh_ref.values()[idx] * dx[dh_ref.row_indices()[idx]]; } }
            tmp
        } else { (0..niq).map(|k| -h[k] - z[k]).collect() };
        let dmu = (0..niq).map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k]).collect();
        (dx, dlam, dz, dmu)
    };
    *total_solve += t_solve.elapsed();
    res
}

fn col_scale_f64(a: &CscMatrix<f64>, scale: &[f64]) -> CscMatrix<f64> {
    let cp = a.col_offsets(); let ri = a.row_indices();
    let vals: Vec<f64> = a.values().iter().enumerate().map(|(idx, &v)| {
        let col = cp.partition_point(|&o| o <= idx) - 1; v * scale[col]
    }).collect();
    CscMatrix::try_from_csc_data(a.nrows(), a.ncols(), cp.to_vec(), ri.to_vec(), vals).unwrap()
}

fn transpose_f64(a: &CscMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows(); let n = a.ncols(); let nnz = a.nnz();
    let a_cp = a.col_offsets(); let a_ri = a.row_indices(); let a_v = a.values();
    let mut row_counts = vec![0usize; m]; for &r in a_ri { row_counts[r] += 1; }
    let mut t_cp = vec![0usize; m + 1]; for i in 0..m { t_cp[i + 1] = t_cp[i] + row_counts[i]; }
    let mut t_ri = vec![0usize; nnz]; let mut t_v = vec![0.0f64; nnz];
    let mut pos = t_cp[..m].to_vec();
    let mut col_of = vec![0usize; nnz]; for j in 0..n { for idx in a_cp[j]..a_cp[j + 1] { col_of[idx] = j; } }
    for idx in 0..nnz { let r = a_ri[idx]; let p = pos[r]; t_ri[p] = col_of[idx]; t_v[p] = a_v[idx]; pos[r] += 1; }
    CscMatrix::try_from_csc_data(n, m, t_cp, t_ri, t_v).unwrap()
}

fn spgemm_f64(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let m = a.nrows(); let n = b.ncols();
    let (a_cp, a_ri, b_cp, b_ri, b_v) = (a.col_offsets(), a.row_indices(), b.col_offsets(), b.row_indices(), b.values());
    let mut acc = vec![0.0f64; m]; let mut visited = vec![false; m];
    let mut c_cp = vec![0usize; n + 1]; let mut c_ri = Vec::new(); let mut c_v = Vec::new();
    for j in 0..n {
        let mut col_nz = Vec::new();
        for idx_b in b_cp[j]..b_cp[j + 1] {
            let kk = b_ri[idx_b]; let b_kj = b_v[idx_b];
            for idx_a in a_cp[kk]..a_cp[kk + 1] {
                let i = a_ri[idx_a];
                if !visited[i] { visited[i] = true; col_nz.push(i); }
                acc[i] += a.values()[idx_a] * b_kj;
            }
        }
        col_nz.sort_unstable();
        for &i in &col_nz { c_ri.push(i); c_v.push(acc[i]); acc[i] = 0.0; visited[i] = false; }
        c_cp[j + 1] = c_ri.len();
    }
    CscMatrix::try_from_csc_data(m, n, c_cp, c_ri, c_v).unwrap()
}

fn f64_add_sparse(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let (nb, a_cp, a_ri, a_v, b_cp, b_ri, b_v) = (a.ncols(), a.col_offsets(), a.row_indices(), a.values(), b.col_offsets(), b.row_indices(), b.values());
    let mut c_cp = vec![0usize; nb + 1]; let mut c_ri = Vec::new(); let mut c_v = Vec::new();
    for j in 0..nb {
        let (mut ia, mut ib) = (a_cp[j], b_cp[j]);
        while ia < a_cp[j+1] || ib < b_cp[j+1] {
            let ra = if ia < a_cp[j+1] { a_ri[ia] } else { usize::MAX };
            let rb = if ib < b_cp[j+1] { b_ri[ib] } else { usize::MAX };
            if ra < rb { c_ri.push(ra); c_v.push(a_v[ia]); ia += 1; }
            else if rb < ra { c_ri.push(rb); c_v.push(b_v[ib]); ib += 1; }
            else { c_ri.push(ra); c_v.push(a_v[ia] + b_v[ib]); ia += 1; ib += 1; }
        }
        c_cp[j + 1] = c_ri.len();
    }
    CscMatrix::try_from_csc_data(a.nrows(), nb, c_cp, c_ri, c_v).unwrap()
}

pub fn build_saddle_point(m: &CscMatrix<f64>, dg: &Option<CscMatrix<f64>>, nx: usize, neq: usize) -> CscMatrix<f64> {
    let dim = nx + neq;
    let mut k_coo = CooMatrix::<f64>::new(dim, dim);
    for j in 0..m.ncols() {
        for idx in m.col_offsets()[j]..m.col_offsets()[j+1] { k_coo.push(m.row_indices()[idx], j, m.values()[idx]); }
    }
    if let Some(dg_ref) = dg {
        let dg_cp = dg_ref.col_offsets(); let dg_ri = dg_ref.row_indices(); let dg_v = dg_ref.values();
        for j in 0..dg_ref.ncols() {
            for idx in dg_cp[j]..dg_cp[j+1] {
                let var_i = dg_ri[idx];
                let val = dg_v[idx];
                k_coo.push(var_i, nx + j, val);
                k_coo.push(nx + j, var_i, val);
            }
        }
    }
    CscMatrix::from(&k_coo)
}

fn build_linear_constraints(nx: usize, ieq: &[usize], ilt: &[usize], igt: &[usize], ibx: &[usize], xmin: &[f64], xmax: &[f64]) -> (Option<CscMatrix<f64>>, Vec<f64>, Option<CscMatrix<f64>>, Vec<f64>) {
    let ni = ilt.len() + igt.len() + 2 * ibx.len();
    let ai = if ni > 0 {
        let mut row = 0usize; let mut ent = Vec::new();
        for &i in ilt { ent.push((row, i, 1.0)); row += 1; }
        for &i in igt { ent.push((row, i, -1.0)); row += 1; }
        for &i in ibx { ent.push((row, i, 1.0)); ent.push((row+1, i, -1.0)); row += 2; }
        Some(coo_to_csc_f64(ni, nx, &ent))
    } else { None };
    let bi: Vec<f64> = ilt.iter().map(|&i| xmax[i]).chain(igt.iter().map(|&i| -xmin[i])).chain(ibx.iter().flat_map(|&i| vec![xmax[i], -xmin[i]])).collect();
    let ae = if !ieq.is_empty() {
        let ent: Vec<_> = ieq.iter().enumerate().map(|(r, &i)| (r, i, 1.0)).collect();
        Some(coo_to_csc_f64(ieq.len(), nx, &ent))
    } else { None };
    let be: Vec<f64> = ieq.iter().map(|&i| xmax[i]).collect();
    (ai, bi, ae, be)
}

fn merge_constraints(x: &[f64], hn: &[f64], gn: &[f64], dhn: &CscMatrix<f64>, dgn: &CscMatrix<f64>, ai: &Option<CscMatrix<f64>>, bi: &[f64], ae: &Option<CscMatrix<f64>>, be: &[f64]) -> (Vec<f64>, Vec<f64>, Option<CscMatrix<f64>>, Option<CscMatrix<f64>>) {
    let (h, dh) = match ai {
        Some(r) => {
            let hlin: Vec<f64> = spmv_f64(r, x).iter().zip(bi).map(|(a, b)| a - b).collect();
            ([hn, &hlin].concat(), Some(hstack_csc(dhn, &transpose_f64(r))))
        }
        None => (hn.to_vec(), Some(dhn.clone())),
    };
    let (g, dg) = match ae {
        Some(r) => {
            let glin: Vec<f64> = spmv_f64(r, x).iter().zip(be).map(|(a, b)| a - b).collect();
            ([gn, &glin].concat(), Some(hstack_csc(dgn, &transpose_f64(r))))
        }
        None => (gn.to_vec(), Some(dgn.clone())),
    };
    (h, g, dh, dg)
}

fn hstack_csc(a: &CscMatrix<f64>, b: &CscMatrix<f64>) -> CscMatrix<f64> {
    let (m, na, nb) = (a.nrows(), a.ncols(), b.ncols());
    let mut cp = vec![0usize; na + nb + 1]; cp[..na+1].copy_from_slice(a.col_offsets());
    for j in 0..nb { cp[na+j+1] = a.nnz() + b.col_offsets()[j+1]; }
    let ri = [a.row_indices(), b.row_indices()].concat();
    let v = [a.values(), b.values()].concat();
    CscMatrix::try_from_csc_data(m, na + nb, cp, ri, v).unwrap()
}

fn coo_to_csc_f64(nr: usize, nc: usize, ent: &[(usize, usize, f64)]) -> CscMatrix<f64> {
    let mut s = ent.to_vec(); s.sort_unstable_by_key(|&(r, c, _)| (c, r));
    let mut cp = vec![0usize; nc + 1]; let mut ri = Vec::new(); let mut v = Vec::new();
    for &(_, c, _) in &s { cp[c+1] += 1; }
    for j in 0..nc { cp[j+1] += cp[j]; }
    for &(r, _, val) in &s { ri.push(r); v.push(val); }
    CscMatrix::try_from_csc_data(nr, nc, cp, ri, v).unwrap()
}

fn spmv_f64(a: &CscMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows()];
    for j in 0..a.ncols() {
        let xj = x[j]; if xj == 0.0 { continue; }
        for idx in a.col_offsets()[j]..a.col_offsets()[j+1] { y[a.row_indices()[idx]] += a.values()[idx] * xj; }
    }
    y
}

fn convergence_measures(g: &[f64], h: &[f64], lx: &[f64], lam: &[f64], mu: &[f64], z: &[f64], x: &[f64], f: f64, f0: f64) -> (f64, f64, f64, f64) {
    let gnorm = g.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let maxh = h.iter().fold(0.0f64, |a, &b| a.max(b.max(0.0)));
    let xnorm = x.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let znorm = z.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let lnorm = lam.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let mnorm = mu.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let feas = gnorm.max(maxh) / (1.0 + xnorm.max(znorm));
    let grad = lx.iter().fold(0.0f64, |a, &b| a.max(b.abs())) / (1.0 + lnorm.max(mnorm));
    let comp = if z.is_empty() { 0.0 } else { z.iter().zip(mu).map(|(a, b)| a * b).sum::<f64>() / (1.0 + xnorm) };
    let cost = (f - f0).abs() / (1.0 + f0.abs());
    (feas, grad, comp, cost)
}

fn step_size(z: &[f64], dz: &[f64], xi: f64) -> f64 {
    let mut a = 1.0f64;
    for k in 0..z.len() { if dz[k] < 0.0 { a = a.min(xi * z[k] / (-dz[k])); } }
    a
}

pub fn pips_with_fused_assembly<F, GH, FA, S>(
    f_fcn: F, gh_fcn: GH, mut fused_assembly: FA, x0: Vec<f64>, xmin: Vec<f64>, xmax: Vec<f64>, opt: PipsOpt, solver: &mut S,
    v5: &crate::new_opf::v5_kkt::KKTSymbolicV5,
) -> PipsResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    GH: Fn(&[f64]) -> (Vec<f64>, Vec<f64>, CscMatrix<f64>, CscMatrix<f64>),
    FA: FnMut(&[f64], &[f64], &[f64], &[f64], f64, &mut [f64]),
    S: crate::basic::solver::Solve,
{
    const XI: f64 = 0.99995; const SIGMA: f64 = 0.1; const Z0: f64 = 1.0;
    const ALPHA_MIN: f64 = 1e-8;

    let nx = x0.len(); let cm = opt.cost_mult;
    let eps = f64::EPSILON;
    let mut ieq = Vec::new(); let mut ilt = Vec::new(); let mut igt = Vec::new(); let mut ibx = Vec::new();
    for i in 0..nx {
        let lo = xmin[i]; let hi = xmax[i];
        if (hi - lo).abs() <= eps { ieq.push(i); }
        else if lo <= -1e10 && hi < 1e10 { ilt.push(i); }
        else if lo > -1e10 && hi >= 1e10 { igt.push(i); }
        else if lo > -1e10 && hi < 1e10 { ibx.push(i); }
    }
    let (ai, bi_vec, ae, be_vec) = build_linear_constraints(nx, &ieq, &ilt, &igt, &ibx, &xmin, &xmax);

    let mut x = x0.clone();
    let (f0_raw, df0) = f_fcn(&x);
    let mut f = f0_raw * cm;
    let mut df: Vec<f64> = df0.iter().map(|&v| v * cm).collect();

    let (hn, gn, dhn, dgn) = gh_fcn(&x);
    let (mut h, mut g, mut dh, mut dg) = merge_constraints(&x, &hn, &gn, &dhn, &dgn, &ai, &bi_vec, &ae, &be_vec);

    let neq = g.len(); let niq = h.len();
    let neqnln = gn.len(); let niqnln = hn.len();

    let mut lam = vec![0.0f64; neq];
    let mut z = vec![Z0; niq]; let mut mu = vec![Z0; niq];
    for k in 0..niq { if h[k] < -Z0 { z[k] = -h[k]; } mu[k] = Z0 / z[k]; }

    let mut lx = df.clone();
    if let Some(ref dg_ref) = dg { matvec_add_to(&mut lx, dg_ref, &lam); }
    if let Some(ref dh_ref) = dh { matvec_add_to(&mut lx, dh_ref, &mu); }

    let (mut feascond, mut gradcond, mut compcond, mut costcond) = convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f);
    let mut converged = feascond < opt.feastol && gradcond < opt.gradtol && compcond < opt.comptol && costcond < opt.costtol;
    let mut i = 0usize; let mut f0 = f;

    let mut total_hess = std::time::Duration::ZERO;
    let mut total_kkt = std::time::Duration::ZERO;
    let mut total_solve_sym = std::time::Duration::ZERO;
    let mut total_solve_num = std::time::Duration::ZERO;
    let mut total_gh = std::time::Duration::ZERO;

    let mut kkt_vals = vec![0.0f64; v5.row_idx.len()];

    while !converged && i < opt.max_it {
        i += 1;
        let t_start = std::time::Instant::now();
        // Fused Assembly
        fused_assembly(&x, &lam[..neqnln], &mu[..niqnln], &z[..niqnln], cm, &mut kkt_vals);
        total_hess += t_start.elapsed();

        let gap = if niq > 0 { z.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>() } else { 0.0 };
        let gamma = if niq > 0 { SIGMA * gap / niq as f64 } else { 0.0 };

        let mut dt_kkt = std::time::Duration::ZERO;
        let mut dt_solve = std::time::Duration::ZERO;
        let (dx, dlam_n, dz_n, dmu_n) = solve_kkt_fused_timed(&kkt_vals, &lx, &dh, &g, &h, &z, &mu, gamma, nx, neq, niq, niqnln, solver, v5, &mut dt_kkt, &mut dt_solve);
        total_kkt += dt_kkt;
        if i == 1 { total_solve_sym += dt_solve; } else { total_solve_num += dt_solve; }

        let alphap = step_size(&z, &dz_n, 0.99995);
        let alphad = step_size(&mu, &dmu_n, 0.99995);

        for j in 0..nx { x[j] += alphap * dx[j]; }
        for j in 0..niq { z[j] += alphap * dz_n[j]; }
        for j in 0..neq { lam[j] += alphad * dlam_n[j]; }
        for j in 0..niq { mu[j] += alphad * dmu_n[j]; }

        let t_gh_start = std::time::Instant::now();
        let (f_new, df_new) = f_fcn(&x);
        f = f_new * cm;
        df = df_new.iter().map(|&v| v * cm).collect();

        let (hn_new, gn_new, dhn_new, dgn_new) = gh_fcn(&x);
        let (h_new, g_new, dh_new, dg_new) = merge_constraints(&x, &hn_new, &gn_new, &dhn_new, &dgn_new, &ai, &bi_vec, &ae, &be_vec);
        h = h_new; g = g_new; dh = dh_new; dg = dg_new;

        lx = df.clone();
        if let Some(ref dg_ref) = dg { matvec_add_to(&mut lx, dg_ref, &lam); }
        if let Some(ref dh_ref) = dh { matvec_add_to(&mut lx, dh_ref, &mu); }
        total_gh += t_gh_start.elapsed();

        let (fc, gc, cc, cc2) = convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f0);
        feascond = fc; gradcond = gc; compcond = cc; costcond = cc2;

        if feascond < opt.feastol && gradcond < opt.gradtol && compcond < opt.comptol && costcond < opt.costtol {
            converged = true;
        } else if x.iter().any(|v| v.is_nan()) || alphap < 1e-8 || alphad < 1e-8 {
            break;
        }
        f0 = f;
    }

    if i > 0 {
        eprintln!("\nPIPS ({} iters): Hess: {:?} G/H: {:?} KKT: {:?} Solv(Sym): {:?} Solv(Num): {:?}", i, total_hess, total_gh, total_kkt, total_solve_sym, total_solve_num);
    }

    PipsResult {
        x, f: f / cm, converged, iterations: i,
        lam_eq: lam[..neqnln].to_vec(), mu_ineq: mu[..niqnln].to_vec(),
        mu_lower: vec![0.0; nx], mu_upper: vec![0.0; nx],
        message: if converged { "Converged".to_string() } else { "Failed".to_string() },
        timing: PipsTiming { hess: total_hess, gh: total_gh, kkt: total_kkt, solve_sym: total_solve_sym, solve_num: total_solve_num },
    }
}

fn solve_kkt_fused_timed<S: crate::basic::solver::Solve>(
    kkt_vals: &[f64], lx: &[f64], dh: &Option<CscMatrix<f64>>,
    g: &[f64], h: &[f64], z: &[f64], mu: &[f64], gamma: f64,
    nx: usize, neq: usize, niq: usize, niqnln: usize, solver: &mut S,
    v5: &crate::new_opf::v5_kkt::KKTSymbolicV5,
    total_kkt: &mut std::time::Duration, total_solve: &mut std::time::Duration,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t_kkt = std::time::Instant::now();
    let mut n_vec = lx.to_vec();
    if let Some(dh_ref) = dh {
        let w: Vec<f64> = (0..niq).map(|k| (mu[k] * h[k] + gamma) / z[k]).collect();
        matvec_add_to(&mut n_vec, dh_ref, &w);
    }

    // Merged Slack Penalty for linear/box constraints (not in fused assembly)
    let mut final_kkt_vals = kkt_vals.to_vec();
    if let Some(dh_ref) = dh {
        for k in niqnln..niq {
            let weight = mu[k] / z[k];
            for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k+1] {
                let r = dh_ref.row_indices()[idx];
                let v = dh_ref.values()[idx];
                // Scatter penalty into variable columns
                // Variable columns are in [0, nx)
                let s = v5.col_ptrs[r]; let e = v5.col_ptrs[r+1];
                if let Ok(pos) = v5.row_idx[s..e].binary_search(&r) {
                    final_kkt_vals[s + pos] += weight * v * v;
                }
            }
        }
    }
    *total_kkt += t_kkt.elapsed();

    let t_solve = std::time::Instant::now();
    let mut rhs = n_vec.iter().map(|&v| -v).chain(g.iter().map(|&v| -v)).collect::<Vec<_>>();
    let (mut ap, mut ai, mut ax) = (v5.col_ptrs.clone(), v5.row_idx.clone(), final_kkt_vals);
    solver.solve(&mut ap, &mut ai, &mut ax, &mut rhs, nx + neq).unwrap();
    let dx = rhs[..nx].to_vec(); let dlam = rhs[nx..].to_vec();
    let dz = if let Some(dh_ref) = dh {
        let mut tmp = (0..niq).map(|k| -h[k] - z[k]).collect::<Vec<_>>();
        for k in 0..niq { for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k+1] { tmp[k] -= dh_ref.values()[idx] * dx[dh_ref.row_indices()[idx]]; } }
        tmp
    } else { (0..niq).map(|k| -h[k] - z[k]).collect() };
    let dmu = (0..niq).map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k]).collect();
    *total_solve += t_solve.elapsed();
    (dx, dlam, dz, dmu)
}

fn matvec_add_to(out: &mut [f64], dg: &CscMatrix<f64>, v: &[f64]) {
    for j in 0..dg.ncols() {
        let vj = v[j]; if vj == 0.0 { continue; }
        for idx in dg.col_offsets()[j]..dg.col_offsets()[j+1] { out[dg.row_indices()[idx]] += dg.values()[idx] * vj; }
    }
}
