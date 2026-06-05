use super::problem::OPFData;

/// Evaluate total generation cost, gradient w.r.t. x = [Va;Vm;Pg;Qg].
///
/// cost(Pg_g) = c2·(Pg_g·baseMVA)² + c1·(Pg_g·baseMVA) + c0
/// Only Pg (and optionally Qg) variables affect the cost.
pub fn opf_costfcn(data: &OPFData, x: &[f64]) -> (f64, Vec<f64>) {
    let nx = data.nx();
    let mut f = 0.0f64;
    let mut df = vec![0.0f64; nx];

    let base = data.base_mva;
    let pg_off = 2 * data.nb;

    for g in 0..data.ng {
        let pg_pu = x[pg_off + g];
        let pg_mw = pg_pu * base;
        let [c2, c1, c0] = data.cost_coeffs[g];
        f += c2 * pg_mw * pg_mw + c1 * pg_mw + c0;
        // df/dPg_pu = df/dPg_MW * dPg_MW/dPg_pu = (2*c2*Pg_MW + c1) * base
        df[pg_off + g] = (2.0 * c2 * pg_mw + c1) * base;
    }

    (f, df)
}

/// Second derivative of cost w.r.t. Pg (diagonal, in x-space).
/// d²f/dPg_g² = 2*c2_g * baseMVA²
pub fn opf_cost_d2f(data: &OPFData) -> Vec<f64> {
    let base = data.base_mva;
    data.cost_coeffs
        .iter()
        .map(|&[c2, _, _]| 2.0 * c2 * base * base)
        .collect()
}
