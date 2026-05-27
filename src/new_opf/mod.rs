pub mod symbolic;
pub mod numeric;
pub mod v3_symbolic;
pub mod v3_numeric;
pub mod v3_numeric_fused;
pub mod v3_numeric_scalar;
pub mod math_verify;
pub mod pips;
pub mod problem;
pub mod components;

pub use problem::NewOPFData;
pub use symbolic::SymbolicCache;
pub use pips::{pips, PipsOpt, PipsResult};
pub use components::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::io::pandapower::load_csv_zip;
    use crate::opf::builder::opf_data_from_network;
    use crate::opf::PipsOpt;
    use nalgebra::DVector;
    use nalgebra_sparse::CscMatrix;
    use num_complex::Complex64;

    fn load_ieee39() -> crate::io::pandapower::Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    #[test]
    fn test_new_opf_ieee39_run() {
        let net = load_ieee39();
        let mut base_data = opf_data_from_network(&net);

        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
             for g in 0..10i64 {
                if let Some(row) = opf_cfg.get("gen", g) {
                    base_data.cost_coeffs[g as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
            }
        }

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );

        println!(
            "New OPF 39: converged={} iter={} f={:.6} msg={}",
            result.converged, result.iterations, result.f, result.message
        );

        assert!(result.converged, "New OPF should converge");
        // Pandapower reference: ~41864.13 EUR
        assert!((result.f - 41864.13).abs() < 10.0, "Result mismatch: {}", result.f);
    }

    #[test]
    fn test_new_opf_ieee118_run() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let mut base_data = opf_data_from_network(&net);

        // Load costs
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
            if let Some(row) = opf_cfg.get("ext_grid", 0) {
                base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
            }
            for g in 0..54i64 {
                if let Some(row) = opf_cfg.get("gen", g) {
                    base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
            }
        }

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );

        println!(
            "New OPF 118: converged={} iter={} f={:.6} msg={}",
            result.converged, result.iterations, result.f, result.message
        );

        assert!(result.converged, "New OPF 118 should converge");
        assert!(result.f > 120000.0 && result.f < 140000.0, "Objective out of expected range: {}", result.f);
    }

    #[test]
    fn test_baseline_opf_ieee39_run() {
        let net = load_ieee39();
        let mut base_data = opf_data_from_network(&net);
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/data.zip", dir);
        if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
             for g in 0..10i64 {
                if let Some(row) = opf_cfg.get("gen", g) {
                    base_data.cost_coeffs[g as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                }
            }
        }
        let x0 = base_data.warm_x0();
        let (xmin, xmax) = base_data.bounds();
        let result = crate::opf::pips::pips(
            |x| crate::opf::cost::opf_costfcn(&base_data, x),
            |x| {
                let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&base_data, x);
                (h, g, dh, dg)
            },
            |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );
        println!("Baseline OPF 39: converged={} iter={} f={:.6}", result.converged, result.iterations, result.f);
    }

    #[test]
    fn test_exact_hessian_comparison_pegase9241() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/pegase9241/data.zip", dir);
        if !std::path::Path::new(&path).exists() { return; }
        
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let base_data = opf_data_from_network(&net);
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&base_data);

        let x = base_data.warm_x0();
        let nx = base_data.nx();
        let lam_eq = vec![0.1; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        let h_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v3 = v3_numeric_scalar::v3_scalar_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);

        println!("Comparing V1 vs V3 Hessian ({} x {})", nx, nx);
        
        let mut diff_count = 0;
        let mut max_err: f64 = 0.0;
        
        use std::collections::HashMap;
        let mut v1_map = HashMap::new();
        for col in 0..nx {
            for idx in h_v1.col_offsets()[col]..h_v1.col_offsets()[col+1] {
                v1_map.insert((h_v1.row_indices()[idx], col), h_v1.values()[idx]);
            }
        }

        for col in 0..nx {
            for idx in h_v3.col_offsets()[col]..h_v3.col_offsets()[col+1] {
                let row = h_v3.row_indices()[idx];
                let val_v3 = h_v3.values()[idx];
                let val_v1 = *v1_map.get(&(row, col)).unwrap_or(&0.0);
                
                let err = (val_v1 - val_v3).abs();
                if err > 1e-7 {
                    if diff_count < 10 {
                        println!("Diff at ({}, {}): V1={:.6e}, V3={:.6e}, Err={:.6e}", row, col, val_v1, val_v3, err);
                    }
                    diff_count += 1;
                    max_err = max_err.max(err);
                }
            }
        }

        println!("Total Diffs: {}, Max Err: {:.6e}", diff_count, max_err);
        assert!(diff_count < 100, "Too many differences in Hessian values!");
    }

    #[test]
    fn bench_full_opf_all_cases() {
        let cases = ["IEEE39", "IEEE118", "pegase9241"];
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        println!("\n| Case | Method | f [EUR] | Iter | Total Time | Speedup |");
        println!("|---|---|---|---|---|---|");

        for case in cases {
            let path = format!("{}/cases/{}/data.zip", dir, case);
            if !std::path::Path::new(&path).exists() { continue; }
            
            let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
            let mut base_data = opf_data_from_network(&net);
            
            if case == "IEEE118" {
                if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                    if let Some(row) = opf_cfg.get("ext_grid", 0) {
                        base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                    }
                    for g in 0..54i64 {
                        if let Some(row) = opf_cfg.get("gen", g) {
                            base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                        }
                    }
                }
            } else if case == "IEEE39" {
                 if let Some(opf_cfg) = crate::io::pandapower::load_opf_cfg_zip(&path) {
                    for g in 0..10i64 {
                        if let Some(row) = opf_cfg.get("gen", g) {
                            base_data.cost_coeffs[g as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
                        }
                    }
                }
            }

            let data_v3 = NewOPFData::new(base_data.clone());
            let x0 = base_data.warm_x0();
            let (xmin, xmax) = base_data.bounds();

            let start_v1 = std::time::Instant::now();
            let res_v1 = crate::opf::pips::pips(
                |x| crate::opf::cost::opf_costfcn(&base_data, x),
                |x| {
                    let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&base_data, x);
                    (h, g, dh, dg)
                },
                |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
            );
            let dur_v1 = start_v1.elapsed();

            let start_v3 = std::time::Instant::now();
            let res_v3 = pips(
                &data_v3,
                x0, xmin, xmax,
                PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
            );
            let dur_v3 = start_v3.elapsed();

            let speedup = dur_v1.as_secs_f64() / dur_v3.as_secs_f64();
            println!("| {} | V1 | {:.2} | {} | {:?} | - |", case, res_v1.f, res_v1.iterations, dur_v1);
            println!("| {} | V3 | {:.2} | {} | {:?} | {:.2}x |", case, res_v3.f, res_v3.iterations, dur_v3, speedup);
        }
    }
}
