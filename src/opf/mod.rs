pub mod problem;
pub mod cost;
pub mod constraints;
pub mod hessian;
pub mod pips;
pub mod builder;

pub use problem::OPFData;
pub use pips::{pips, PipsOpt, PipsResult};
pub use builder::opf_data_from_network;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::pandapower::load_csv_zip;
    use std::env;
    #[allow(unused_imports)]
    use serde_json;

    fn load_ieee118() -> crate::io::pandapower::Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        load_csv_zip(&path).unwrap()
    }

    #[test]
    fn test_opf_ieee118_build() {
        let net = load_ieee118();
        let data = opf_data_from_network(&net);
        assert_eq!(data.nb, 118);
        assert!(data.nl > 0);
        assert!(data.ng > 0);
        println!("nb={} nl={} ng={} nx={}", data.nb, data.nl, data.ng, data.nx());
    }

    fn load_ieee39() -> crate::io::pandapower::Network {
        serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap()
    }

    #[test]
    fn test_opf_ieee39_run() {
        let net = load_ieee39();
        let mut data = opf_data_from_network(&net);

        // Apply real poly_cost from the embedded case39 JSON (cp2=0.01, cp1=0.3, cp0=0.2 EUR/MW)
        let opf_cfg = crate::io::pandapower::load_opf_cfg_json_str(
            crate::testcases::case_ieee39::IEEE_39,
        ).expect("poly_cost missing from case39 JSON");
        // Generator order: ext_grid[0] → cost_coeffs[0], gen[0..8] → cost_coeffs[1..9]
        if let Some(row) = opf_cfg.get("ext_grid", 0) {
            data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
        }
        for g in 0..9i64 {
            if let Some(row) = opf_cfg.get("gen", g) {
                data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
            }
        }

        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            |x| cost::opf_costfcn(&data, x),
            |x| {
                let (g, h, dg, dh) = constraints::opf_consfcn(&data, x);
                (h, g, dh, dg)
            },
            |x, lam_eq, mu_ineq, cm| hessian::opf_hessfcn(&data, x, lam_eq, mu_ineq, cm),
            x0, xmin, xmax,
            PipsOpt { max_it: 100, cost_mult: 1e-4, ..Default::default() },
        );
        println!("IEEE39 OPF: converged={} iter={} f={:.4} EUR", result.converged, result.iterations, result.f);
        assert!(result.converged, "IEEE39 OPF should converge with warm start");
        // pandapower reference: ~41872 EUR
        assert!(result.f > 35000.0 && result.f < 50000.0, "Objective out of expected range: {}", result.f);
    }

    #[test]
    fn test_opf_ieee118_run() {
        let net = load_ieee118();
        let mut data = opf_data_from_network(&net);

        // Load costs from the same zip file
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let opf_cfg = crate::io::pandapower::load_opf_cfg_zip(&path)
            .expect("poly_cost.csv missing from IEEE118 zip");

        // Apply costs: ext_grid[0], then gen[0..53]
        if let Some(row) = opf_cfg.get("ext_grid", 0) {
            data.cost_coeffs[0] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
        }
        for g in 0..54i64 {
            if let Some(row) = opf_cfg.get("gen", g) {
                data.cost_coeffs[(1 + g) as usize] = [row.cp2_eur_per_mw2, row.cp1_eur_per_mw, row.cp0_eur];
            }
        }

        let x0 = data.warm_x0();
        let (xmin, xmax) = data.bounds();

        let result = pips(
            |x| cost::opf_costfcn(&data, x),
            |x| {
                let (g, h, dg, dh) = constraints::opf_consfcn(&data, x);
                (h, g, dh, dg)
            },
            |x, lam_eq, mu_ineq, cm| hessian::opf_hessfcn(&data, x, lam_eq, mu_ineq, cm),
            x0, xmin, xmax,
            PipsOpt { max_it: 150, cost_mult: 1e-4, ..Default::default() },
        );

        println!(
            "OPF 118: converged={} iter={} f={:.6} msg={}",
            result.converged, result.iterations, result.f, result.message
        );

        assert!(result.converged, "IEEE118 OPF should converge");
        // Pandapower reference: ~129704 EUR
        assert!(result.f > 120000.0 && result.f < 140000.0, "Objective out of expected range: {}", result.f);
    }

}
