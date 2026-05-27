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
            |x| crate::opf::constraints::opf_consfcn(&base_data, x),
            |x, l, m, c| crate::opf::hessian::opf_hessfcn(&base_data, x, l, m, c),
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );
        println!("Baseline OPF 39: converged={} iter={} f={:.6}", result.converged, result.iterations, result.f);
    }

    #[test]
    fn bench_hessian_118() {
        let net = crate::io::pandapower::load_csv_zip(&format!("{}/cases/IEEE118/data.zip", std::env::var("CARGO_MANIFEST_DIR").unwrap())).unwrap();
        let base_data = opf_data_from_network(&net);
        let v3_cache = v3_symbolic::V3SymbolicCache::analyze(&base_data);

        let x = base_data.warm_x0();
        let lam_eq = vec![0.1; 2 * base_data.nb];
        let mu_ineq = vec![0.05; 2 * base_data.nl];
        let cost_mult = 1e-4;

        // Warm up
        for _ in 0..5 {
            let _ = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
            let _ = v3_numeric::v3_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        }

        let start_base = std::time::Instant::now();
        for _ in 0..100 {
            let _ = crate::opf::hessian::opf_hessfcn(&base_data, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        }
        let dur_base = start_base.elapsed() / 100;

        let start_new = std::time::Instant::now();
        for _ in 0..100 {
            let _ = v3_numeric::v3_numeric_fill(&base_data, &v3_cache, x.as_slice(), &lam_eq, &mu_ineq, cost_mult);
        }
        let dur_new = start_new.elapsed() / 100;

        println!("--- IEEE 118 Hessian Assembly Performance ---");
        println!("Baseline (Old Path): {:?}", dur_base);
        println!("V3 (Revolutionary Path): {:?}", dur_new);
        println!("Hessian Speedup: {:.2}x", dur_base.as_secs_f64() / dur_new.as_secs_f64());
    }
}
