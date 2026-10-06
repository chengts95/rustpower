//! Opt-in V3/V4 fill benchmark using their separate symbolic caches. Run alone in release mode, pinned to a CPU.
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64 as C;
use rustpower::basic::{
    benchmark_internals::{JacobianPattern2, fill_jacobian_v3, fill_jacobian_v4},
    newtonpf::newton_pf,
    solver::KLUSolver,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
struct Case {
    name: String,
    y: CscMatrix<C>,
    s: DVector<C>,
    v: DVector<C>,
    npv: usize,
    npq: usize,
}
fn zip_case(name: &str) -> Case {
    use crate::basic::ecs::{
        elements::PPNetwork,
        network::{DataOps, PowerFlow, PowerGrid},
        powerflow::systems::PowerFlowMat,
    };
    let net = crate::io::pandapower::load_csv_zip(&format!(
        "{}/cases/{name}/data.zip",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let m = grid.world().get_resource::<PowerFlowMat>().unwrap();
    Case {
        name: name.into(),
        y: m.y_bus.clone(),
        s: m.s_bus.clone(),
        v: m.v_bus_init.clone(),
        npv: m.npv,
        npq: m.npq,
    }
}
fn rte() -> Case {
    let m: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!(
            "{}/target/research/lm_audit/6515rte_dc.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    let indices = |k: &str| {
        m[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    };
    let complex = |r: &str, i: &str| {
        m[r].as_array()
            .unwrap()
            .iter()
            .zip(m[i].as_array().unwrap())
            .map(|(r, i)| C::new(r.as_f64().unwrap(), i.as_f64().unwrap()))
            .collect()
    };
    let nb = m["nb"].as_u64().unwrap() as usize;
    Case {
        name: "6515rte (DC start)".into(),
        y: CscMatrix::try_from_csc_data(
            nb,
            nb,
            indices("cp"),
            indices("ri"),
            complex("y_re", "y_im"),
        )
        .unwrap(),
        s: DVector::from_vec(complex("s_re", "s_im")),
        v: DVector::from_vec(complex("v_re", "v_im")),
        npv: m["npv"].as_u64().unwrap() as usize,
        npq: m["npq"].as_u64().unwrap() as usize,
    }
}
fn solve(c: &Case) -> (DVector<C>, usize) {
    newton_pf(
        &c.y,
        &c.s,
        &c.v,
        c.npv,
        c.npq,
        Some(1e-8),
        Some(100),
        &mut KLUSolver::default(),
        None,
    )
    .unwrap()
}
fn median(x: &[f64]) -> f64 {
    let mut a = x.to_vec();
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}
pub fn acpf_v3_vs_v4_fill() {
    let mut records = vec![];
    for c in [
        zip_case("IEEE39"),
        zip_case("IEEE118"),
        rte(),
        zip_case("pegase9241"),
    ] {
        let p = JacobianPattern2::build_from_permuted(
            c.y.col_offsets(),
            c.y.row_indices(),
            c.npv,
            c.npq,
        );
        let p4 = rustpower::basic::jacobian_cache::JacobianCache::build_from_permuted(
            c.y.col_offsets(),
            c.y.row_indices(),
            c.npv,
            c.npq,
        );
        assert_eq!(p.j_col_ptrs, p4.j_col_ptrs);
        assert_eq!(p.j_row_indices, p4.j_row_indices);
        let (sol, it) = solve(&c);
        let current = &c.y * &sol;
        let s: Vec<_> = sol
            .iter()
            .zip(current.iter())
            .map(|(v, i)| v * i.conj())
            .collect();
        let vn: Vec<_> = sol.iter().map(|v| v / v.norm()).collect();
        let mut output = vec![0.; p.nnz_j];
        let mut reference = output.clone();
        let fill = |op: bool, out: &mut [f64]| {
            if op {
                fill_jacobian_v4::<false>(
                    black_box(&c.y),
                    black_box(sol.as_slice()),
                    black_box(&vn),
                    black_box(&s),
                    &p4.j_col_ptrs,
                    &p4.pq_ends,
                    &p4.active_ends,
                    &p4.diag_ptrs,
                    c.npv,
                    c.npq,
                    black_box(out),
                );
            } else {
                fill_jacobian_v3(
                    black_box(&c.y),
                    black_box(sol.as_slice()),
                    black_box(&vn),
                    black_box(&s),
                    &p,
                    c.npv,
                    c.npq,
                    black_box(out),
                );
            }
        };
        fill(false, &mut reference);
        fill(true, &mut output);
        let maxdiff = reference
            .iter()
            .zip(&output)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f64, f64::max);
        let scale = reference
            .iter()
            .chain(&output)
            .map(|a| a.abs())
            .fold(1_f64, f64::max);
        println!(
            "{}: Jacobian max absolute difference={maxdiff:.3e}, matrix scale={scale:.3e}",
            c.name
        );
        // Diagonal corrections can cancel large terms near zero; use a matrix-scaled
        // rounding bound for the two algebraically equivalent fills.
        assert!(maxdiff < 64. * f64::EPSILON * scale);
        let fill_reps = if c.y.ncols() < 200 { 5000 } else { 200 };
        let mut fill_samples = [vec![], vec![]];
        for _ in 0..100 {
            fill(false, &mut output);
            fill(true, &mut output);
        }
        for round in 0..15 {
            for op in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let t = Instant::now();
                for _ in 0..fill_reps {
                    fill(op, &mut output);
                    black_box(&output);
                }
                fill_samples[op as usize].push(t.elapsed().as_secs_f64() * 1e6 / fill_reps as f64);
            }
        }
        let f = [median(&fill_samples[0]), median(&fill_samples[1])];
        println!(
            "{}: fill V3={:.3}us V4={:.3}us ratio={:.3}",
            c.name,
            f[0],
            f[1],
            f[1] / f[0]
        );
        records.push(json!({"case":c.name,"nnz_j":p.nnz_j,"iterations":it,
            "max_jacobian_difference":maxdiff,"jacobian_max_abs":scale,
            "fill_repeats_per_batch":fill_reps,"fill_us_v3_v4":f,"fill_us_samples_v3_v4":fill_samples}));
    }
    let path = format!(
        "{}/target/research/lm_audit/acpf_v3_v4_fill.json",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(path, serde_json::to_string_pretty(&records).unwrap()).unwrap();
}
