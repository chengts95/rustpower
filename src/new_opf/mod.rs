pub mod symbolic;
pub mod numeric;
pub mod v3_symbolic;
pub mod v3_numeric;
pub mod v3_numeric_fused;
pub mod v3_numeric_scalar;
pub mod v4_numeric_rect;
pub mod v5_kkt;
pub mod math_verify;
pub mod pips;
pub mod problem;
pub mod components;
pub mod translate;

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
        // Use the same embedded JSON case + cost mapping as opf::tests::test_opf_ieee39_run
        // so V1 and V3 solve the identical canonical pandapower case39 (ref ~41864 EUR).
        let net: crate::io::pandapower::Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let mut base_data = opf_data_from_network(&net);

        let opf_cfg = crate::io::pandapower::load_opf_cfg_json_str(
            crate::testcases::case_ieee39::IEEE_39,
        ).expect("poly_cost missing from case39 JSON");
        if let Some(row) = opf_cfg.get("ext_grid", 0) {
            base_data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
        }
        for g in 0..9i64 {
            if let Some(row) = opf_cfg.get("gen", g) {
                base_data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
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
            |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
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
        let h_v4 = v4_numeric_rect::v4_rect_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, None, cost_mult);

        println!("Comparing V1 vs V4 Hessian ({} x {})", nx, nx);
        
        // V4 accuracy vs V3
        let mut diff_v4_v3 = 0;
        let mut max_err_v4_v3: f64 = 0.0;
        for col in 0..nx {
            let start = h_v3.col_offsets()[col];
            let end = h_v3.col_offsets()[col+1];
            for i in start..end {
                let val_v3 = h_v3.values()[i];
                let val_v4 = h_v4.values()[i];
                let err = (val_v3 - val_v4).abs();
                if err > 1e-8 {
                    diff_v4_v3 += 1;
                    max_err_v4_v3 = max_err_v4_v3.max(err);
                }
            }
        }

        println!("Total v4-vs-V3 diffs: {}, Max Err: {:.6e}", diff_v4_v3, max_err_v4_v3);
        assert!(diff_v4_v3 == 0, "V4 must be mathematically identical to V3");
    }

    #[test]
    #[ignore] // run explicitly: cargo test --release run_pegase9241_v3_convergence -- --ignored --nocapture
    fn run_pegase9241_v3_convergence() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/pegase9241/data.zip", dir);
        if !std::path::Path::new(&path).exists() {
            println!("pegase9241 data not present; skipping");
            return;
        }
        let net = crate::io::pandapower::load_csv_zip(&path).unwrap();
        let base_data = opf_data_from_network(&net);
        println!("pegase9241: nb={} nl={} ng={} nx={}", base_data.nb, base_data.nl, base_data.ng, base_data.nx());

        let data = NewOPFData::new(base_data);
        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let t0 = std::time::Instant::now();
        let result = pips(
            &data,
            x0, xmin, xmax,
            PipsOpt { max_it: 200, cost_mult: 1e-4, ..Default::default() },
        );
        let dt = t0.elapsed();

        println!(
            "\n=== Corrected v3 on pegase9241 ===\nconverged={} iter={} f={:.4} msg={} time={:?}",
            result.converged, result.iterations, result.f, result.message, dt
        );
    }

    #[test]
    fn diag_hessian_breakdown_ieee118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let base_data = opf_data_from_network(&net);
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&base_data);

        let nb = base_data.nb;
        let nx = base_data.nx();
        let x = base_data.warm_x0();
        let lam_eq = vec![0.1; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        let h_v1 = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        let h_v4 = v4_numeric_rect::v4_rect_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, None, cost_mult);

        use std::collections::HashMap;
        let to_map = |m: &CscMatrix<f64>| {
            let mut map = HashMap::new();
            for col in 0..nx {
                for idx in m.col_offsets()[col]..m.col_offsets()[col+1] {
                    map.insert((m.row_indices()[idx], col), m.values()[idx]);
                }
            }
            map
        };
        let m1 = to_map(&h_v1);
        let block = |r: usize, c: usize| -> &'static str {
            let rb = if r < nb {0} else if r < 2*nb {1} else {2};
            let cb = if c < nb {0} else if c < 2*nb {1} else {2};
            match (rb, cb) {
                (0,0)=>"aa",(0,1)=>"av",(1,0)=>"va",(1,1)=>"vv",
                (2,2)=>"gen",_=>"other",
            }
        };
        let compare = |label: &str, m2: &HashMap<(usize,usize), f64>| -> usize {
            let mut keys: std::collections::HashSet<(usize,usize)> = m1.keys().cloned().collect();
            keys.extend(m2.keys().cloned());
            let mut per_block: HashMap<&str,(usize,f64)> = HashMap::new();
            let mut total = 0usize;
            for k in keys {
                let a = *m1.get(&k).unwrap_or(&0.0);
                let b = *m2.get(&k).unwrap_or(&0.0);
                let err = (a-b).abs();
                if err > 1e-7 {
                    let e = per_block.entry(block(k.0,k.1)).or_insert((0,0.0));
                    e.0 += 1; e.1 = e.1.max(err);
                    total += 1;
                }
            }
            println!("--- {} vs V1 ---", label);
            let mut blocks: Vec<_> = per_block.iter().collect();
            blocks.sort_by_key(|(k,_)| *k);
            for (blk, (cnt, mx)) in blocks {
                println!("  block {}: diffs={} maxErr={:.4e}", blk, cnt, mx);
            }
            total
        };
        let _ = compare("v4_rect", &to_map(&h_v4));

        // === Gold standard: finite-difference validaton ===
        let lag = |xv: &[f64]| -> f64 {
            let (f, _) = crate::opf::cost::opf_costfcn(&base_data, xv);
            let (g, hh, _, _) = crate::opf::constraints::opf_consfcn(&base_data, xv);
            let mut s = cost_mult * f;
            for i in 0..2 * nb { s += lam_eq[i] * g[i]; }
            for j in 0..2 * base_data.nl { s += mu_ineq[j] * hh[j]; }
            s
        };
        let hstep = 1e-5;
        let d2 = |i: usize, k: usize| -> f64 {
            let mut a = x.clone(); a[i] += hstep; a[k] += hstep;
            let mut b = x.clone(); b[i] += hstep; b[k] -= hstep;
            let mut c = x.clone(); c[i] -= hstep; c[k] += hstep;
            let mut d = x.clone(); d[i] -= hstep; d[k] -= hstep;
            (lag(&a) - lag(&b) - lag(&c) + lag(&d)) / (4.0 * hstep * hstep)
        };

        let m_v4 = to_map(&h_v4);
        const ATOL: f64 = 0.5;
        const RTOL: f64 = 5e-4;
        let mut worst_v4 = (0usize, 0usize, 0.0f64, 0.0f64); 
        for (&(r, c), &val_v4) in m_v4.iter() {
            if r > c { continue; }
            if r >= 2 * nb && c >= 2 * nb { continue; } 
            let fd = d2(r, c);
            let tol = ATOL + RTOL * fd.abs();
            let (a4, q4) = ((fd - val_v4).abs(), (fd - val_v4).abs() / tol);
            if q4 > worst_v4.3 { worst_v4 = (r, c, a4, q4); }
        }
        println!("  FD gold standard for V4: worst ratio={:.2e} abs={:.4e} at ({},{})", worst_v4.3, worst_v4.2, worst_v4.0, worst_v4.1);
        assert!(worst_v4.3 < 1.0, "v4 Hessian disagrees with finite differences");
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
                |x, l, m, _z, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: false, ..Default::default() },
            );
            let dur_v1 = start_v1.elapsed();

            let start_v3 = std::time::Instant::now();
            let res_v3 = pips(
                &data_v3,
                x0.clone(), xmin.clone(), xmax.clone(),
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: false, ..Default::default() },
            );
            let dur_v3 = start_v3.elapsed();

            // V4 Benchmark
            let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&data_v3);
            let start_v4 = std::time::Instant::now();
            let res_v4 = crate::opf::pips::pips(
                |x| crate::opf::cost::opf_costfcn(&data_v3, x),
                |x| {
                    let (g, h, dg, dh) = crate::opf::constraints::opf_consfcn(&data_v3, x);
                    (h, g, dh, dg)
                },
                |x, lam_eq, mu_ineq, z_ineq, cost_mult| {
                    v4_numeric_rect::v4_rect_numeric_fill(&data_v3, &v3_cache, x, lam_eq, mu_ineq, Some(z_ineq), cost_mult)
                },
                x0, xmin, xmax,
                PipsOpt { max_it: 150, cost_mult: 1e-4, merged_slacks: true, ..Default::default() },
            );
            let dur_v4 = start_v4.elapsed();

            let speedup3 = dur_v1.as_secs_f64() / dur_v3.as_secs_f64();
            let speedup4 = dur_v1.as_secs_f64() / dur_v4.as_secs_f64();
            println!("| {} | V1 | {:.2} | {} | {:?} | - |", case, res_v1.f, res_v1.iterations, dur_v1);
            println!("| {} | V3 | {:.2} | {} | {:?} | {:.2}x |", case, res_v3.f, res_v3.iterations, dur_v3, speedup3);
            println!("| {} | V4 | {:.2} | {} | {:?} | {:.2}x |", case, res_v4.f, res_v4.iterations, dur_v4, speedup4);
        }
    }
}
