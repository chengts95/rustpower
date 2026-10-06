use super::*;

pub fn pips_with_fused_assembly<F, GH, FA, S>(
    f_fcn: F,
    gh_fcn: GH,
    mut fused_assembly: FA,
    x0: Vec<f64>,
    xmin: Vec<f64>,
    xmax: Vec<f64>,
    opt: PipsOpt,
    solver: &mut S,
    v5: &crate::new_opf::assembly::v5::symbolic::KKTSymbolicV5,
) -> PipsResult
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
    GH: Fn(&[f64]) -> (Vec<f64>, Vec<f64>, CscMatrix<f64>, CscMatrix<f64>),
    FA: FnMut(&[f64], &[f64], &[f64], &[f64], f64, &mut [f64]),
    S: crate::basic::solver::Solve,
{
    const XI: f64 = 0.99995;
    const SIGMA: f64 = 0.1;
    const Z0: f64 = 1.0;
    const ALPHA_MIN: f64 = 1e-8;

    let nx = x0.len();
    let cm = opt.cost_mult;
    let eps = f64::EPSILON;
    let mut ieq = Vec::new();
    let mut ilt = Vec::new();
    let mut igt = Vec::new();
    let mut ibx = Vec::new();
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
    let (ai, bi_vec, ae, be_vec) =
        build_linear_constraints(nx, &ieq, &ilt, &igt, &ibx, &xmin, &xmax);

    let mut x = x0.clone();
    let (f0_raw, df0) = f_fcn(&x);
    let mut f = f0_raw * cm;
    let mut df: Vec<f64> = df0.iter().map(|&v| v * cm).collect();

    let (hn, gn, dhn, dgn) = gh_fcn(&x);
    let (mut h, mut g, mut dh, mut dg) =
        merge_constraints(&x, &hn, &gn, &dhn, &dgn, &ai, &bi_vec, &ae, &be_vec);

    let neq = g.len();
    let niq = h.len();
    let neqnln = gn.len();
    let niqnln = hn.len();

    let mut lam = vec![0.0f64; neq];
    let mut z = vec![Z0; niq];
    let mut mu = vec![Z0; niq];
    for k in 0..niq {
        if h[k] < -Z0 {
            z[k] = -h[k];
        }
        mu[k] = Z0 / z[k];
    }

    let mut lx = df.clone();
    if let Some(ref dg_ref) = dg {
        matvec_add_to(&mut lx, dg_ref, &lam);
    }
    if let Some(ref dh_ref) = dh {
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
        fused_assembly(
            &x,
            &lam[..neqnln],
            &mu[..niqnln],
            &z[..niqnln],
            cm,
            &mut kkt_vals,
        );
        total_hess += t_start.elapsed();

        let gap = if niq > 0 {
            z.iter().zip(mu.iter()).map(|(a, b)| a * b).sum::<f64>()
        } else {
            0.0
        };
        let gamma = if niq > 0 {
            SIGMA * gap / niq as f64
        } else {
            0.0
        };

        let mut dt_kkt = std::time::Duration::ZERO;
        let mut dt_solve = std::time::Duration::ZERO;
        let (dx, dlam_n, dz_n, dmu_n) = solve_kkt_fused_timed(
            &kkt_vals,
            &lx,
            dh.as_ref(),
            &g,
            &h,
            &z,
            &mu,
            gamma,
            nx,
            neq,
            niq,
            niqnln,
            solver,
            v5,
            &mut dt_kkt,
            &mut dt_solve,
        );
        total_kkt += dt_kkt;
        if i == 1 {
            total_solve_sym += dt_solve;
        } else {
            total_solve_num += dt_solve;
        }

        let alphap = step_size(&z, &dz_n, XI);
        let alphad = step_size(&mu, &dmu_n, XI);

        for j in 0..nx {
            x[j] += alphap * dx[j];
        }
        for j in 0..niq {
            z[j] += alphap * dz_n[j];
        }
        for j in 0..neq {
            lam[j] += alphad * dlam_n[j];
        }
        for j in 0..niq {
            mu[j] += alphad * dmu_n[j];
        }

        let t_gh_start = std::time::Instant::now();
        let (f_new, df_new) = f_fcn(&x);
        f = f_new * cm;
        df = df_new.iter().map(|&v| v * cm).collect();

        let (hn_new, gn_new, dhn_new, dgn_new) = gh_fcn(&x);
        let (h_new, g_new, dh_new, dg_new) = merge_constraints(
            &x, &hn_new, &gn_new, &dhn_new, &dgn_new, &ai, &bi_vec, &ae, &be_vec,
        );
        h = h_new;
        g = g_new;
        dh = dh_new;
        dg = dg_new;

        lx = df.clone();
        if let Some(ref dg_ref) = dg {
            matvec_add_to(&mut lx, dg_ref, &lam);
        }
        if let Some(ref dh_ref) = dh {
            matvec_add_to(&mut lx, dh_ref, &mu);
        }
        total_gh += t_gh_start.elapsed();

        let (fc, gc, cc, cc2) = convergence_measures(&g, &h, &lx, &lam, &mu, &z, &x, f, f0);
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
        } else if x.iter().any(|v| v.is_nan()) || alphap < ALPHA_MIN || alphad < ALPHA_MIN {
            break;
        }
        f0 = f;
    }

    if i > 0 {
        eprintln!(
            "\nPIPS ({} iters): Hess: {:?} G/H: {:?} KKT: {:?} Solv(Sym): {:?} Solv(Num): {:?}",
            i, total_hess, total_gh, total_kkt, total_solve_sym, total_solve_num
        );
    }

    PipsResult {
        x,
        f: f / cm,
        converged,
        iterations: i,
        lam_eq: lam[..neqnln].to_vec(),
        mu_ineq: mu[..niqnln].to_vec(),
        mu_lower: vec![0.0; nx],
        mu_upper: vec![0.0; nx],
        message: if converged {
            "Converged".to_string()
        } else {
            "Failed".to_string()
        },
        timing: PipsTiming {
            hess: total_hess,
            gh: total_gh,
            kkt: total_kkt,
            solve_sym: total_solve_sym,
            solve_num: total_solve_num,
        },
    }
}
