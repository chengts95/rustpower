//! V5.5: Direct-Fill Jacobian Evaluator (PF-style)
//!
//! Eliminates `opf_consfcn`'s per-iteration matrix construction entirely. Like the PF
//! `fill_jacobian_v3`, this computes the dSbus/dAbr scalars INLINE over the Ybus/Yf/Yt
//! CSC structure and writes them straight into the preallocated dgn/dhn value arrays at
//! prestored indices — no intermediate `dS_dVa`/`dS_dVm`/`dAbr` matrices, no allocation,
//! no `merge_constraints` concatenation.
//!
//! Structure facts (verified against `basic::dsbus_dv` / `basic::dsbr_dv`):
//!   - dSbus_dVa, dSbus_dVm share Ybus's CSC structure exactly (col = variable bus j,
//!     row = equation bus i, nnz index == Ybus nnz index).
//!   - dSf/dVx share Yf's structure (the k==f[l] "self" term merges into the existing
//!     Yf[l,f[l]] entry); likewise dSt with Yt. dAbr keeps that same structure.

use crate::opf::problem::OPFData;
use num_complex::Complex64;
use crate::opf::constraints::opf_consfcn;

pub struct V55Evaluator {
    pub dgn_cp: Vec<usize>,
    pub dgn_ri: Vec<usize>,
    pub dhn_cp: Vec<usize>,
    pub dhn_ri: Vec<usize>,

    // Per Ybus-nnz → dgn value index (P uses .re, Q uses .im).
    va_to_dgn_p: Vec<usize>,
    va_to_dgn_q: Vec<usize>,
    vm_to_dgn_p: Vec<usize>,
    vm_to_dgn_q: Vec<usize>,
    // Per generator → dgn value index for the -Cg coupling (P, Q).
    cg_to_dgn_p: Vec<usize>,
    cg_to_dgn_q: Vec<usize>,

    // Per Yf-nnz / Yt-nnz → dhn value index.
    daf_va_to_dhn: Vec<usize>,
    daf_vm_to_dhn: Vec<usize>,
    dat_va_to_dhn: Vec<usize>,
    dat_vm_to_dhn: Vec<usize>,
}

impl V55Evaluator {
    pub fn new(data: &OPFData) -> Self {
        let x0 = data.warm_x0();
        // One reference call to obtain the (constant) dgn/dhn CSC structure.
        let (_, _, dgn, dhn) = opf_consfcn(data, &x0);

        let nb = data.nb;
        let nl = data.nl;
        let ng = data.ng;

        let find_dgn = |col: usize, row: usize| -> usize {
            let r = dgn.col_offsets()[col]..dgn.col_offsets()[col + 1];
            r.start + dgn.row_indices()[r.clone()].binary_search(&row).unwrap()
        };
        let find_dhn = |col: usize, row: usize| -> usize {
            let r = dhn.col_offsets()[col]..dhn.col_offsets()[col + 1];
            r.start + dhn.row_indices()[r.clone()].binary_search(&row).unwrap()
        };

        // ── dgn maps, indexed by Ybus nnz (col = variable bus k, row = equation bus i) ──
        let y_cp = data.ybus.col_offsets();
        let y_ri = data.ybus.row_indices();
        let ynnz = data.ybus.nnz();
        let mut va_to_dgn_p = vec![0usize; ynnz];
        let mut va_to_dgn_q = vec![0usize; ynnz];
        let mut vm_to_dgn_p = vec![0usize; ynnz];
        let mut vm_to_dgn_q = vec![0usize; ynnz];
        for k in 0..nb {
            for idx in y_cp[k]..y_cp[k + 1] {
                let i = y_ri[idx];                  // equation bus
                va_to_dgn_p[idx] = find_dgn(i, k);          // dgn[P_eq_i, Va_k]
                va_to_dgn_q[idx] = find_dgn(nb + i, k);     // dgn[Q_eq_i, Va_k]
                vm_to_dgn_p[idx] = find_dgn(i, nb + k);     // dgn[P_eq_i, Vm_k]
                vm_to_dgn_q[idx] = find_dgn(nb + i, nb + k);// dgn[Q_eq_i, Vm_k]
            }
        }

        let mut cg_to_dgn_p = vec![0usize; ng];
        let mut cg_to_dgn_q = vec![0usize; ng];
        for g in 0..ng {
            let bus = data.gen_bus[g];
            cg_to_dgn_p[g] = find_dgn(bus, 2 * nb + g);          // dgn[P_eq_bus, Pg_g]
            cg_to_dgn_q[g] = find_dgn(nb + bus, 2 * nb + ng + g);// dgn[Q_eq_bus, Qg_g]
        }

        // ── dhn maps, indexed by Yf/Yt nnz (col = bus k, row = branch l) ──
        let map_branch = |ybr: &nalgebra_sparse::CscMatrix<Complex64>, is_vm: bool, col_off: usize| -> Vec<usize> {
            let cp = ybr.col_offsets();
            let ri = ybr.row_indices();
            let mut m = vec![0usize; ybr.nnz()];
            for k in 0..nb {
                for idx in cp[k]..cp[k + 1] {
                    let l = ri[idx];
                    let r = if is_vm { nb + k } else { k };
                    m[idx] = find_dhn(col_off + l, r);
                }
            }
            m
        };
        let daf_va_to_dhn = map_branch(&data.yf, false, 0);
        let daf_vm_to_dhn = map_branch(&data.yf, true, 0);
        let dat_va_to_dhn = map_branch(&data.yt, false, nl);
        let dat_vm_to_dhn = map_branch(&data.yt, true, nl);

        Self {
            dgn_cp: dgn.col_offsets().to_vec(), dgn_ri: dgn.row_indices().to_vec(),
            dhn_cp: dhn.col_offsets().to_vec(), dhn_ri: dhn.row_indices().to_vec(),
            va_to_dgn_p, va_to_dgn_q, vm_to_dgn_p, vm_to_dgn_q,
            cg_to_dgn_p, cg_to_dgn_q,
            daf_va_to_dhn, daf_vm_to_dhn, dat_va_to_dhn, dat_vm_to_dhn,
        }
    }

    /// Direct-fill update: computes g, h and the dgn/dhn value arrays in place, fully
    /// inline. The only matvecs are Ybus·V, Yf·V, Yt·V — each O(nnz) and also required
    /// for g/h, so nothing is wasted. No matrices are allocated.
    pub fn update(
        &self, data: &OPFData, x: &[f64],
        g: &mut [f64], h: &mut [f64], dgn_v: &mut [f64], dhn_v: &mut [f64],
    ) {
        let nb = data.nb;
        let nl = data.nl;
        let j = Complex64::i();

        let v = data.v_from_x(x);
        let vs = v.as_slice();
        let mut vnorm = vec![Complex64::new(0.0, 0.0); nb];
        for i in 0..nb { vnorm[i] = vs[i] / vs[i].norm().max(1e-9); }

        // ── g: power-balance mismatch  mis = V·conj(Ybus·V) − Sbus ──
        let ibus = &data.ybus * &v;
        let ibus_s = ibus.as_slice();
        let sbus = data.sbus_from_x(x);
        for i in 0..nb {
            let s = vs[i] * ibus_s[i].conj();
            g[i] = s.re - sbus[i].re;
            g[nb + i] = s.im - sbus[i].im;
        }

        // ── dgn: dSbus/dVa, dSbus/dVm inline over Ybus structure ──
        let y_cp = data.ybus.col_offsets();
        let y_ri = data.ybus.row_indices();
        let y_v = data.ybus.values();
        for k in 0..nb {
            for idx in y_cp[k]..y_cp[k + 1] {
                let i = y_ri[idx];
                let y_ik = y_v[idx];
                let (dsa, dsm) = if i == k {
                    (j * vs[i] * (ibus_s[i] - y_ik * vs[i]).conj(),
                     vs[i] * (y_ik * vnorm[i]).conj() + ibus_s[i].conj() * vnorm[i])
                } else {
                    (j * vs[i] * (-y_ik * vs[k]).conj(),
                     vs[i] * (y_ik * vnorm[k]).conj())
                };
                dgn_v[self.va_to_dgn_p[idx]] = dsa.re;
                dgn_v[self.va_to_dgn_q[idx]] = dsa.im;
                dgn_v[self.vm_to_dgn_p[idx]] = dsm.re;
                dgn_v[self.vm_to_dgn_q[idx]] = dsm.im;
            }
        }
        // generator coupling: dP_eq/dPg = -1, dQ_eq/dQg = -1 (from -Cg)
        for g_i in 0..data.ng {
            let val = -data.cg.values()[g_i];
            dgn_v[self.cg_to_dgn_p[g_i]] = val;
            dgn_v[self.cg_to_dgn_q[g_i]] = val;
        }

        // ── h + dhn: branch apparent-power limits, inline over Yf/Yt structure ──
        let i_f = &data.yf * &v;
        let i_t = &data.yt * &v;
        let flow_max = data.flow_max_sq();
        let mut s_f = vec![Complex64::new(0.0, 0.0); nl];
        let mut s_t = vec![Complex64::new(0.0, 0.0); nl];
        for l in 0..nl {
            s_f[l] = vs[data.f_buses[l]] * i_f[l].conj();
            s_t[l] = vs[data.t_buses[l]] * i_t[l].conj();
            h[l] = s_f[l].norm_sqr() - flow_max[l];
            h[nl + l] = s_t[l].norm_sqr() - flow_max[l];
        }

        self.fill_branch(&data.yf, &data.f_buses, vs, &vnorm, i_f.as_slice(), &s_f,
                         &self.daf_va_to_dhn, &self.daf_vm_to_dhn, dhn_v);
        self.fill_branch(&data.yt, &data.t_buses, vs, &vnorm, i_t.as_slice(), &s_t,
                         &self.dat_va_to_dhn, &self.dat_vm_to_dhn, dhn_v);
    }

    /// Inline dAbr fill for one end (from or to). dA/dV = 2·Re(conj(S)·dS/dV).
    #[allow(clippy::too_many_arguments)]
    fn fill_branch(
        &self, ybr: &nalgebra_sparse::CscMatrix<Complex64>, bus: &[usize],
        vs: &[Complex64], vnorm: &[Complex64], i_br: &[Complex64], s_br: &[Complex64],
        va_map: &[usize], vm_map: &[usize], dhn_v: &mut [f64],
    ) {
        let j = Complex64::i();
        let cp = ybr.col_offsets();
        let ri = ybr.row_indices();
        let yv = ybr.values();
        for k in 0..ybr.ncols() {
            for idx in cp[k]..cp[k + 1] {
                let l = ri[idx];
                let bl = bus[l];
                let y = yv[idx];
                // dS/dVa[l,k] and dS/dVm[l,k] (self term k==bus[l] merges in)
                let mut dsva = vs[bl] * (j * y * vs[k]).conj();
                let mut dsvm = vs[bl] * (y * vnorm[k]).conj();
                if k == bl {
                    dsva += j * vs[bl] * i_br[l].conj();
                    dsvm += i_br[l].conj() * vnorm[bl];
                }
                let sc = s_br[l].conj();
                dhn_v[va_map[idx]] = 2.0 * (sc * dsva).re;
                dhn_v[vm_map[idx]] = 2.0 * (sc * dsvm).re;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opf::builder::opf_data_from_network;
    use crate::io::pandapower::load_csv_zip;

    /// V5.5 direct-fill must reproduce opf_consfcn's g, h, dgn, dhn byte-for-byte.
    #[test]
    fn test_v5_5_direct_fill_matches_consfcn_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net = load_csv_zip(&format!("{}/cases/IEEE118/data.zip", dir)).unwrap();
        let data = opf_data_from_network(&net);

        let ev = V55Evaluator::new(&data);
        // perturb away from x0 to exercise all terms
        let mut x = data.warm_x0();
        for (i, xi) in x.iter_mut().enumerate() { *xi += 0.01 * ((i % 7) as f64 - 3.0); }

        let (g_ref, h_ref, dgn_ref, dhn_ref) = opf_consfcn(&data, &x);

        let mut g = vec![0.0; g_ref.len()];
        let mut h = vec![0.0; h_ref.len()];
        let mut dgn_v = vec![0.0; dgn_ref.values().len()];
        let mut dhn_v = vec![0.0; dhn_ref.values().len()];
        ev.update(&data, &x, &mut g, &mut h, &mut dgn_v, &mut dhn_v);

        let maxd = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x,y)|(x-y).abs()).fold(0.0f64,f64::max);
        let dg = maxd(&g, &g_ref);
        let dh = maxd(&h, &h_ref);
        let ddg = maxd(&dgn_v, dgn_ref.values());
        let ddh = maxd(&dhn_v, dhn_ref.values());
        println!("V5.5 vs opf_consfcn: |g|={:.2e} |h|={:.2e} |dgn|={:.2e} |dhn|={:.2e}", dg, dh, ddg, ddh);
        assert!(dg < 1e-12 && dh < 1e-12 && ddg < 1e-12 && ddh < 1e-12,
            "V5.5 direct-fill mismatch: g={:.2e} h={:.2e} dgn={:.2e} dhn={:.2e}", dg, dh, ddg, ddh);
    }
}
