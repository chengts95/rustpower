use super::*;

pub(super) fn solve_kkt_fused_timed<S: crate::basic::solver::Solve>(
    kkt_vals: &[f64],
    lx: &[f64],
    dh: Option<&CscMatrix<f64>>,
    g: &[f64],
    h: &[f64],
    z: &[f64],
    mu: &[f64],
    gamma: f64,
    nx: usize,
    neq: usize,
    niq: usize,
    niqnln: usize,
    solver: &mut S,
    v5: &crate::new_opf::assembly::v5::symbolic::KKTSymbolicV5,
    total_kkt: &mut std::time::Duration,
    total_solve: &mut std::time::Duration,
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
            for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k + 1] {
                let r = dh_ref.row_indices()[idx];
                let v = dh_ref.values()[idx];
                // Scatter penalty into variable columns
                // Variable columns are in [0, nx)
                let s = v5.col_ptrs[r];
                let e = v5.col_ptrs[r + 1];
                if let Ok(pos) = v5.row_idx[s..e].binary_search(&r) {
                    final_kkt_vals[s + pos] += weight * v * v;
                }
            }
        }
    }
    *total_kkt += t_kkt.elapsed();

    let t_solve = std::time::Instant::now();
    let mut rhs = n_vec
        .iter()
        .map(|&v| -v)
        .chain(g.iter().map(|&v| -v))
        .collect::<Vec<_>>();
    let (mut ap, mut ai, mut ax) = (v5.col_ptrs.clone(), v5.row_idx.clone(), final_kkt_vals);
    solver
        .solve(&mut ap, &mut ai, &mut ax, &mut rhs, nx + neq)
        .unwrap();
    let dx = rhs[..nx].to_vec();
    let dlam = rhs[nx..].to_vec();
    let dz = if let Some(dh_ref) = dh {
        let mut tmp = (0..niq).map(|k| -h[k] - z[k]).collect::<Vec<_>>();
        for k in 0..niq {
            for idx in dh_ref.col_offsets()[k]..dh_ref.col_offsets()[k + 1] {
                tmp[k] -= dh_ref.values()[idx] * dx[dh_ref.row_indices()[idx]];
            }
        }
        tmp
    } else {
        (0..niq).map(|k| -h[k] - z[k]).collect()
    };
    let dmu = (0..niq)
        .map(|k| -mu[k] + (gamma - mu[k] * dz[k]) / z[k])
        .collect();
    *total_solve += t_solve.elapsed();
    (dx, dlam, dz, dmu)
}
