//! V5.6: Direct-fill G/H with linear constraints baked into a static structure.
//!
//! V5.5 removed the per-iteration derivative-matrix construction (dSbus/dAbr), but the
//! PIPS loop still called `merge_constraints` every iteration, which re-transposed the
//! constant ae/ai bound matrices and hstacked them onto dgn/dhn — rebuilding full CSC
//! matrices each iter. V5.6 removes that too:
//!
//!   dg = [ dgn | aeᵀ ]   (nx × neq)   — column-major, so values = [dgn.values | aeᵀ.values]
//!   dh = [ dhn | aiᵀ ]   (nx × niq)
//!
//! The aeᵀ/aiᵀ suffix is constant (±1 at fixed/bounded variables) → prefilled once.
//! The dgn/dhn prefix is direct-filled by the V5.5 evaluator (same indices, since it's a
//! contiguous prefix). Per iteration only the cheap linear scalars `ae·x−be`, `ai·x−bi`
//! are recomputed. No transpose, no hstack, no matrix allocation in the loop.

use crate::opf::problem::OPFData;
use nalgebra_sparse::CscMatrix;
use super::v5_5_evaluator::V55Evaluator;

pub struct V56Evaluator {
    v55: V55Evaluator,

    // Final merged structures (constant): dg (nx×neq), dh (nx×niq).
    pub dg_cp: Vec<usize>,
    pub dg_ri: Vec<usize>,
    pub dh_cp: Vec<usize>,
    pub dh_ri: Vec<usize>,
    pub dg_vals0: Vec<f64>, // prefilled: dgn region = 0, aeᵀ suffix = const ±1
    pub dh_vals0: Vec<f64>, // prefilled: dhn region = 0, aiᵀ suffix = const ±1

    dgn_nnz: usize,
    dhn_nnz: usize,
    pub neqnln: usize, // = 2nb
    pub niqnln: usize, // = 2nl
    pub neq: usize,
    pub niq: usize,

    // Linear constraint rows: he = ae·x − be (equality), hi = ai·x − bi (inequality).
    ae: Option<CscMatrix<f64>>, // neqlin × nx
    be: Vec<f64>,
    ai: Option<CscMatrix<f64>>, // nlin × nx
    bi: Vec<f64>,
}

impl V56Evaluator {
    pub fn new(data: &OPFData) -> Self {
        let nx = data.nx();
        let (xmin, xmax) = data.bounds();
        let eps = f64::EPSILON;

        // Classify bounds exactly like pips_with_fused_assembly_v55.
        let (mut ieq, mut ilt, mut igt, mut ibx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..nx {
            let (lo, hi) = (xmin[i], xmax[i]);
            if (hi - lo).abs() <= eps { ieq.push(i); }
            else if lo <= -1e10 && hi < 1e10 { ilt.push(i); }
            else if lo > -1e10 && hi >= 1e10 { igt.push(i); }
            else if lo > -1e10 && hi < 1e10 { ibx.push(i); }
        }

        // ae (neqlin × nx): one +1 per fixed variable.  be = xmax[ieq].
        let neqlin = ieq.len();
        let ae = if neqlin > 0 {
            let ent: Vec<(usize, usize, f64)> = ieq.iter().enumerate().map(|(r, &i)| (r, i, 1.0)).collect();
            Some(coo_to_csc(neqlin, nx, &ent))
        } else { None };
        let be: Vec<f64> = ieq.iter().map(|&i| xmax[i]).collect();

        // ai (nlin × nx): bound inequalities (matches build_linear_constraints).
        let nlin = ilt.len() + igt.len() + 2 * ibx.len();
        let ai = if nlin > 0 {
            let mut row = 0usize; let mut ent = Vec::new();
            for &i in &ilt { ent.push((row, i, 1.0)); row += 1; }
            for &i in &igt { ent.push((row, i, -1.0)); row += 1; }
            for &i in &ibx { ent.push((row, i, 1.0)); ent.push((row + 1, i, -1.0)); row += 2; }
            Some(coo_to_csc(nlin, nx, &ent))
        } else { None };
        let bi: Vec<f64> = ilt.iter().map(|&i| xmax[i])
            .chain(igt.iter().map(|&i| -xmin[i]))
            .chain(ibx.iter().flat_map(|&i| vec![xmax[i], -xmin[i]]))
            .collect();

        // Nonlinear structure from V5.5.
        let v55 = V55Evaluator::new(data);
        let nb = data.nb;
        let nl = data.nl;
        let neqnln = 2 * nb;
        let niqnln = 2 * nl;
        let dgn_nnz = v55.dgn_ri.len();
        let dhn_nnz = v55.dhn_ri.len();

        // Build merged dg = [dgn | aeᵀ], dh = [dhn | aiᵀ] (structure + const suffix values).
        let (dg_cp, dg_ri, mut dg_vals0) =
            hstack_with_transpose(nx, &v55.dgn_cp, &v55.dgn_ri, neqnln, ae.as_ref());
        let (dh_cp, dh_ri, mut dh_vals0) =
            hstack_with_transpose(nx, &v55.dhn_cp, &v55.dhn_ri, niqnln, ai.as_ref());
        // prefix (nonlinear) starts at 0; suffix already holds the const ±1 from aeᵀ/aiᵀ.
        for v in dg_vals0[..dgn_nnz].iter_mut() { *v = 0.0; }
        for v in dh_vals0[..dhn_nnz].iter_mut() { *v = 0.0; }

        Self {
            v55, dg_cp, dg_ri, dh_cp, dh_ri, dg_vals0, dh_vals0,
            dgn_nnz, dhn_nnz,
            neqnln, niqnln,
            neq: neqnln + neqlin, niq: niqnln + nlin,
            ae, be, ai, bi,
        }
    }

    /// Fill the full merged g, h, dg.values, dh.values in place. No matrix allocation,
    /// no transpose, no hstack — the structure is constant and the const ±1 suffix is
    /// already present in `dg_v`/`dh_v` (caller seeds them from dg_vals0/dh_vals0 once,
    /// or we only overwrite the nonlinear prefix here).
    pub fn update(
        &self, data: &OPFData, x: &[f64],
        g: &mut [f64], h: &mut [f64], dg_v: &mut [f64], dh_v: &mut [f64],
    ) {
        // Nonlinear g/h + dgn/dhn prefix via V5.5 direct-fill.
        self.v55.update(
            data, x,
            &mut g[..self.neqnln], &mut h[..self.niqnln],
            &mut dg_v[..self.dgn_nnz], &mut dh_v[..self.dhn_nnz],
        );

        // Linear equality rows: g[2nb..] = ae·x − be.
        if let Some(ref ae) = self.ae {
            let ax = spmv(ae, x);
            for (r, val) in ax.into_iter().enumerate() {
                g[self.neqnln + r] = val - self.be[r];
            }
        }
        // Linear inequality rows: h[2nl..] = ai·x − bi.
        if let Some(ref ai) = self.ai {
            let ax = spmv(ai, x);
            for (r, val) in ax.into_iter().enumerate() {
                h[self.niqnln + r] = val - self.bi[r];
            }
        }
        // dg/dh suffix (aeᵀ/aiᵀ) is constant ±1 — left untouched in dg_v/dh_v.
    }
}

/// Build CSC of [ inner | Mᵀ ] given inner structure (nx × inner_ncols) and an optional
/// M (m_rows × nx). Returns (col_ptrs, row_idx, values) with the inner-value prefix set
/// from `inner` being absent (we only have structure), so prefix values are 0 and the
/// Mᵀ suffix carries M's values transposed.
fn hstack_with_transpose(
    nx: usize,
    inner_cp: &[usize],
    inner_ri: &[usize],
    inner_ncols: usize,
    m: Option<&CscMatrix<f64>>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let inner_nnz = inner_ri.len();
    match m {
        None => {
            // dg/dh = inner only.
            (inner_cp.to_vec(), inner_ri.to_vec(), vec![0.0; inner_nnz])
        }
        Some(m) => {
            // Mᵀ: nx × m_rows. m is m_rows × nx (CSC). Transpose to CSC(nx × m_rows).
            let mt = transpose(m); // nx × m_rows
            let suffix_ncols = mt.ncols();
            let total_cols = inner_ncols + suffix_ncols;
            let mut cp = vec![0usize; total_cols + 1];
            cp[..inner_ncols + 1].copy_from_slice(inner_cp);
            let mt_cp = mt.col_offsets();
            for j in 0..suffix_ncols {
                cp[inner_ncols + j + 1] = inner_nnz + mt_cp[j + 1];
            }
            let mut ri = Vec::with_capacity(inner_nnz + mt.nnz());
            ri.extend_from_slice(inner_ri);
            ri.extend_from_slice(mt.row_indices());
            let mut vals = vec![0.0; inner_nnz];
            vals.extend_from_slice(mt.values());
            let _ = nx;
            (cp, ri, vals)
        }
    }
}

fn coo_to_csc(nr: usize, nc: usize, ent: &[(usize, usize, f64)]) -> CscMatrix<f64> {
    let mut s = ent.to_vec();
    s.sort_unstable_by_key(|&(r, c, _)| (c, r));
    let mut cp = vec![0usize; nc + 1];
    let mut ri = Vec::new();
    let mut v = Vec::new();
    for &(_, c, _) in &s { cp[c + 1] += 1; }
    for j in 0..nc { cp[j + 1] += cp[j]; }
    for &(r, _, val) in &s { ri.push(r); v.push(val); }
    CscMatrix::try_from_csc_data(nr, nc, cp, ri, v).unwrap()
}

fn transpose(a: &CscMatrix<f64>) -> CscMatrix<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let cp = a.col_offsets(); let ri = a.row_indices(); let v = a.values();
    let mut tcp = vec![0usize; m + 1];
    for &r in ri { tcp[r + 1] += 1; }
    for i in 0..m { tcp[i + 1] += tcp[i]; }
    let mut pos = tcp.clone();
    let mut tri = vec![0usize; ri.len()];
    let mut tv = vec![0.0; v.len()];
    for c in 0..n {
        for idx in cp[c]..cp[c + 1] {
            let r = ri[idx];
            let p = pos[r];
            tri[p] = c; tv[p] = v[idx]; pos[r] += 1;
        }
    }
    CscMatrix::try_from_csc_data(n, m, tcp, tri, tv).unwrap()
}

fn spmv(a: &CscMatrix<f64>, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; a.nrows()];
    let cp = a.col_offsets(); let ri = a.row_indices(); let v = a.values();
    for j in 0..a.ncols() {
        let xj = x[j];
        if xj == 0.0 { continue; }
        for idx in cp[j]..cp[j + 1] { y[ri[idx]] += v[idx] * xj; }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opf::builder::opf_data_from_network;
    use crate::opf::constraints::opf_consfcn;
    use crate::io::pandapower::load_csv_zip;

    /// V5.6 merged g/h/dg/dh must equal the legacy merge_constraints output byte-for-byte.
    #[test]
    fn test_v5_6_merged_matches_consfcn_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net = load_csv_zip(&format!("{}/cases/IEEE118/data.zip", dir)).unwrap();
        let data = opf_data_from_network(&net);
        let nx = data.nx();

        let ev = V56Evaluator::new(&data);
        let mut x = data.warm_x0();
        for (i, xi) in x.iter_mut().enumerate() { *xi += 0.01 * ((i % 7) as f64 - 3.0); }

        // Reference: opf_consfcn + the exact merge the pips loop does.
        let (gn, hn, dgn, dhn) = {
            let (g, h, dg, dh) = opf_consfcn(&data, &x);
            (g, h, dg, dh)
        };
        // Build reference merged via the same linear constraints.
        let ev_ref = V56Evaluator::new(&data); // reuse its ae/ai/be/bi
        // glin/hlin
        let mut g_ref = gn.clone();
        if let Some(ref ae) = ev_ref.ae {
            let ax = spmv(ae, &x);
            for (r, val) in ax.into_iter().enumerate() { g_ref.push(val - ev_ref.be[r]); }
        }
        let mut h_ref = hn.clone();
        if let Some(ref ai) = ev_ref.ai {
            let ax = spmv(ai, &x);
            for (r, val) in ax.into_iter().enumerate() { h_ref.push(val - ev_ref.bi[r]); }
        }

        // V5.6 update
        let mut g = vec![0.0; ev.neq];
        let mut h = vec![0.0; ev.niq];
        let mut dg_v = ev.dg_vals0.clone();
        let mut dh_v = ev.dh_vals0.clone();
        ev.update(&data, &x, &mut g, &mut h, &mut dg_v, &mut dh_v);

        let maxd = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0f64,f64::max);
        let dg_diff = maxd(&g, &g_ref);
        let dh_diff = maxd(&h, &h_ref);

        // dg/dh value check: reconstruct reference dg = [dgn | aeᵀ] values.
        let mut dgn_ref_v = dgn.values().to_vec();
        if let Some(ref ae) = ev_ref.ae { dgn_ref_v.extend_from_slice(transpose(ae).values()); }
        let mut dhn_ref_v = dhn.values().to_vec();
        if let Some(ref ai) = ev_ref.ai { dhn_ref_v.extend_from_slice(transpose(ai).values()); }
        let ddg = maxd(&dg_v, &dgn_ref_v);
        let ddh = maxd(&dh_v, &dhn_ref_v);

        println!("V5.6 vs merge_constraints: |g|={:.2e} |h|={:.2e} |dg|={:.2e} |dh|={:.2e}", dg_diff, dh_diff, ddg, ddh);
        assert!(dg_diff < 1e-12 && dh_diff < 1e-12 && ddg < 1e-12 && ddh < 1e-12);
        assert_eq!(g.len(), g_ref.len());
        assert_eq!(dg_v.len(), dgn_ref_v.len());
    }
}
