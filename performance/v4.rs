use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::basic::benchmark_internals::*;
use rustpower::basic::ecs::{
    elements::PPNetwork,
    network::{DataOps, PowerFlow, PowerGrid},
    powerflow::systems::PowerFlowMat,
};
use rustpower::basic::jacobian_cache::JacobianCache;
use rustpower::io::pandapower::{Network, load_csv_zip};
use rustpower::lm::{KktPattern, fill_jt};
use std::time::{Duration, Instant};

fn eval_inputs(
    nb: usize,
    ybus: &CscMatrix<Complex64>,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let v: Vec<Complex64> = (0..nb)
        .map(|k| {
            let ang = 0.03 * (1.3 * k as f64).sin() - 0.01 * k as f64;
            let mag = 1.0 + 0.004 * (2.1 * k as f64).cos();
            Complex64::from_polar(mag, ang)
        })
        .collect();
    let mut ibus = vec![Complex64::new(0.0, 0.0); nb];
    for j in 0..nb {
        for p in ybus.col_offsets()[j]..ybus.col_offsets()[j + 1] {
            ibus[ybus.row_indices()[p]] += ybus.values()[p] * v[j];
        }
    }
    let scalc: Vec<Complex64> = (0..nb).map(|i| v[i] * ibus[i].conj()).collect();
    let vnorm: Vec<Complex64> = (0..nb)
        .map(|i| {
            let m = v[i].norm();
            if m > 1e-12 {
                v[i] / m
            } else {
                Complex64::new(1.0, 0.0)
            }
        })
        .collect();
    (v, vnorm, scalc)
}

fn load_ieee118_mat() -> PowerFlowMat {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let net: Network = load_csv_zip(&format!("{dir}/cases/IEEE118/data.zip")).unwrap();
    let mut pf = PowerGrid::default();
    pf.world_mut().insert_resource(PPNetwork(net));
    pf.init_pf_net();
    pf.world()
        .get_resource::<PowerFlowMat>()
        .expect("init_pf_net did not produce a PowerFlowMat resource")
        .clone()
}

fn timeit(label: &str, repeats: usize, mut f: impl FnMut()) -> Duration {
    f(); // warm-up
    let mut total = Duration::ZERO;
    let mut min = Duration::MAX;
    for _ in 0..repeats {
        let t = Instant::now();
        f();
        let d = t.elapsed();
        total += d;
        min = min.min(d);
    }
    let avg = total / repeats as u32;
    println!("    {label:<24} avg {avg:?}   min {min:?}");
    avg
}

fn two_pass_j_jt(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    npv: usize,
    npq: usize,
    j: &mut [f64],
    jt: &mut [f64],
) {
    let cache = &pat.cache;
    fill_jacobian_v4::<false>(
        ybus,
        v,
        vnorm,
        scalc,
        &pat.graph.col_starts,
        cache.pq_ends(),
        cache.active_ends(),
        cache.diag_ptrs(),
        npv,
        npq,
        j,
    );
    fill_jt::<false>(ybus, pat, j.as_ptr(), jt.as_mut_ptr());
}

fn fused_fill(
    ybus: &CscMatrix<Complex64>,
    pat: &KktPattern,
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    npv: usize,
    npq: usize,
    j: &mut [f64],
    jt: &mut [f64],
) {
    let cache = &pat.cache;
    fill_j_and_jt_exp(
        ybus,
        v,
        vnorm,
        scalc,
        &pat.graph.col_starts,
        cache.pq_ends(),
        cache.active_ends(),
        cache.diag_ptrs(),
        cache.y_trans(),
        npv,
        npq,
        j,
        jt,
    );
}

pub fn v4_vs_v3_perf_ieee118() {
    let mat = load_ieee118_mat();
    let ybus = &mat.y_bus;
    let (npv, npq) = (mat.npv, mat.npq);
    let nb = ybus.ncols();
    let pat =
        JacobianPattern2::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);
    let (v, vnorm, scalc) = eval_inputs(nb, ybus);

    let repeats = 2000;
    let mut j = vec![0.0; pat.nnz_j];
    println!("--- IEEE118 Jacobian fill: V3 (stored tables) vs V4 (inline) ---");
    let avg_v3 = timeit("V3 fill_jacobian_v3", repeats, || {
        fill_jacobian_v3(ybus, &v, &vnorm, &scalc, &pat, npv, npq, &mut j)
    });
    let v4_cache =
        JacobianCache::build_from_permuted(ybus.col_offsets(), ybus.row_indices(), npv, npq);
    let avg_v4 = timeit("V4 fill_jacobian_v4", repeats, || {
        fill_v4_block(ybus, &v4_cache, &v, &vnorm, &scalc, npv, npq, &mut j)
    });
    let ratio = avg_v3.as_secs_f64() / avg_v4.as_secs_f64();
    println!("    V3/V4 = {ratio:.3}x");
}

pub fn fused_vs_two_pass_perf_ieee118() {
    let mat = load_ieee118_mat();
    let ybus = &mat.y_bus;
    let (npv, npq) = (mat.npv, mat.npq);
    let nb = ybus.ncols();
    let pat = KktPattern::build(ybus, npv, npq);
    let (v, vnorm, scalc) = eval_inputs(nb, ybus);
    let nnz = pat.graph.nnz;

    let repeats = 2000;
    let (mut j, mut jt) = (vec![0.0; nnz], vec![0.0; nnz]);
    println!("--- IEEE118: (v4 + fill_jt) two-pass vs fused single-pass ---");
    let avg_two = timeit("two-pass v4+fill_jt", repeats, || {
        two_pass_j_jt(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j, &mut jt)
    });
    let avg_fused = timeit("fused fill_j_and_jt", repeats, || {
        fused_fill(ybus, &pat, &v, &vnorm, &scalc, npv, npq, &mut j, &mut jt)
    });
    let ratio = avg_two.as_secs_f64() / avg_fused.as_secs_f64();
    println!("    two-pass/fused = {ratio:.3}x");
}

fn fill_v4_block(
    ybus: &CscMatrix<Complex64>,
    pat: &JacobianCache,
    v: &[Complex64],
    vnorm: &[Complex64],
    scalc: &[Complex64],
    npv: usize,
    npq: usize,
    out: &mut [f64],
) {
    fill_jacobian_v4::<false>(
        ybus,
        v,
        vnorm,
        scalc,
        &pat.j_col_ptrs,
        &pat.pq_ends,
        &pat.active_ends,
        &pat.diag_ptrs,
        npv,
        npq,
        out,
    );
}
