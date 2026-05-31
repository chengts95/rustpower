//! End-to-end comparison: V4 vs V5.0 vs V5.2 vs V5.3 vs V5.5 on IEEE118.
//! Run: cargo run --release --features klu --example v5_compare

use rustpower::io::pandapower::{load_csv_zip, load_opf_cfg_zip};
use rustpower::opf::builder::opf_data_from_network;
use rustpower::opf::pips::PipsOpt;
use rustpower::new_opf::NewOPFData;
use rustpower::new_opf::pips::{pips, pips_v5, pips_v5_2, pips_v5_3, pips_v5_5, pips_v5_6};

fn main() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let path = format!("{}/cases/IEEE118/data.zip", dir);
    let net = load_csv_zip(&path).unwrap();
    let mut base = opf_data_from_network(&net);
    if let Some(cfg) = load_opf_cfg_zip(&path) {
        if let Some(r) = cfg.get("ext_grid", 0) {
            base.cost_coeffs[0] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
        }
        for g in 0..54i64 {
            if let Some(r) = cfg.get("gen", g) {
                base.cost_coeffs[(1 + g) as usize] = [r.cp2_eur_per_mw2, r.cp1_eur_per_mw, r.cp0_eur];
            }
        }
    }
    let data = NewOPFData::new(base);
    let x0 = data.warm_x0();
    let (xmin, xmax) = data.bounds();
    let opt = || PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() };

    let r4 = pips(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());
    let r50 = pips_v5(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());
    let r52 = pips_v5_2(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());
    let r53 = pips_v5_3(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());
    let r55 = pips_v5_5(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());
    let r56 = pips_v5_6(&data, x0.clone(), xmin.clone(), xmax.clone(), opt());

    println!("\n=== IEEE118 end-to-end (stage totals over all iters) ===");
    for (name, r) in [("V4", &r4), ("V5.0", &r50), ("V5.2", &r52), ("V5.3", &r53), ("V5.5", &r55), ("V5.6", &r56)] {
        println!(
            "{:5}: conv={} iter={} f={:.6} | Hess {:?} G/H {:?} KKT {:?} Solv {:?}",
            name, r.converged, r.iterations, r.f,
            r.timing.hess, r.timing.gh, r.timing.kkt, r.timing.solve_sym + r.timing.solve_num,
        );
    }
    let dx = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max);
    println!("max|dx| V5.2 vs V4 = {:.3e}", dx(&r52.x, &r4.x));
    println!("max|dx| V5.3 vs V4 = {:.3e}", dx(&r53.x, &r4.x));
    println!("max|dx| V5.5 vs V4 = {:.3e}", dx(&r55.x, &r4.x));
    println!("max|dx| V5.6 vs V4 = {:.3e}", dx(&r56.x, &r4.x));
}
