use rustpower::new_opf::assembly::v5::{scatter::*, symbolic::KKTSymbolicV5};
use rustpower::opf::builder::opf_data_from_network;

pub fn bench_v5_2_kkt_prep() {
    use crate::opf::pips::build_saddle_point;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    for case in ["IEEE118", "pegase9241"] {
        let path = format!("{}/cases/{}/data.zip", dir, case);
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let data = opf_data_from_network(&net);
        let nb = data.nb;
        let nx = data.nx();
        let v5 = KKTSymbolicV5::build(&data);
        let v3c = crate::new_opf::assembly::v3::symbolic::V3SymbolicCache::analyze(&data);
        let x = data.warm_x0();
        let lam = vec![0.1; 2 * nb];
        let mu = vec![0.0; 2 * data.nl];
        let cm = 1e-4;
        let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for g in 0..data.ng {
            gens_at_bus[data.gen_bus[g]].push(g);
        }
        let neqlin = v5.ieq.len();
        let iters = if case == "IEEE118" { 300 } else { 30 };

        // build merged dg once (structure invariant); included per-iter for V4/V5.0 fairness via opf_consfcn
        let mut kkt = vec![0.0f64; v5.row_idx.len()];

        // V4: v4 node fill + opf_consfcn dg + build_saddle_point
        let t = std::time::Instant::now();
        let mut sink = 0.0;
        for _ in 0..iters {
            let lxx = crate::new_opf::assembly::v4::curvature::v4_rect_numeric_fill(
                &data,
                &v3c,
                x.as_slice(),
                &lam,
                &mu,
                None,
                cm,
            );
            let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
            let mut coo = CooMatrix::<f64>::new(nx, 2 * nb + neqlin);
            for j in 0..dg.ncols() {
                for idx in dg.col_offsets()[j]..dg.col_offsets()[j + 1] {
                    coo.push(dg.row_indices()[idx], j, dg.values()[idx]);
                }
            }
            for (r, &vv) in v5.ieq.iter().enumerate() {
                coo.push(vv, 2 * nb + r, 1.0);
            }
            let dgf = CscMatrix::from(&coo);
            let k = build_saddle_point(&lxx, &Some(dgf), nx, v5.neq);
            sink += k.values()[0];
        }
        let d_v4 = t.elapsed() / iters;

        // V5.0: v4 node fill + opf_consfcn dg + transpose + fill
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let lxx = crate::new_opf::assembly::v4::curvature::v4_rect_numeric_fill(
                &data,
                &v3c,
                x.as_slice(),
                &lam,
                &mu,
                None,
                cm,
            );
            let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
            let mut coo = CooMatrix::<f64>::new(nx, 2 * nb + neqlin);
            for j in 0..dg.ncols() {
                for idx in dg.col_offsets()[j]..dg.col_offsets()[j + 1] {
                    coo.push(dg.row_indices()[idx], j, dg.values()[idx]);
                }
            }
            for (r, &vv) in v5.ieq.iter().enumerate() {
                coo.push(vv, 2 * nb + r, 1.0);
            }
            let dgf = CscMatrix::from(&coo);
            let dgt = dgf.transpose();
            v5.fill_from_merged(
                lxx.col_offsets(),
                lxx.values(),
                dgf.col_offsets(),
                dgf.values(),
                dgt.col_offsets(),
                dgt.values(),
                &mut kkt,
            );
            sink += kkt[0];
        }
        let d_v50 = t.elapsed() / iters;

        // V5.2: block operators, fully inline
        let t = std::time::Instant::now();
        for _ in 0..iters {
            fill_variable_columns(
                &v5,
                &data,
                &v3c.y_transpose_idx,
                x.as_slice(),
                &lam,
                cm,
                &mut kkt,
            );
            fill_constraint_columns(
                &v5,
                &data,
                &v3c.y_transpose_idx,
                &gens_at_bus,
                x.as_slice(),
                &mut kkt,
            );
            sink += kkt[0];
        }
        let d_v52 = t.elapsed() / iters;

        println!(
            "[{}] KKT-prep/iter — V4: {:?} | V5.0: {:?} ({:.1}x) | V5.2 inline: {:?} ({:.1}x)  (sink={:.2e})",
            case,
            d_v4,
            d_v50,
            d_v4.as_secs_f64() / d_v50.as_secs_f64(),
            d_v52,
            d_v4.as_secs_f64() / d_v52.as_secs_f64(),
            sink
        );
    }
}
