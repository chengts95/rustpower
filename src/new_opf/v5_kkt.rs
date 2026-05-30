//! V5: Symbolic KKT construction directly from Ybus structure.
//!
//! The full KKT sparsity is a pure function of the Ybus structure + the gen→bus map +
//! the fixed-variable (linear-equality) set. This module builds the KKT CSC skeleton
//! (`col_ptrs`, `row_idx`) once, in canonical (row-ascending) order, with NO COO build,
//! NO sort, NO scatter lookup tables. The numeric kernel (separate) then streams values
//! into the preallocated `values` array with a single advancing pointer.
//!
//! KKT layout (natural variable order, identity bus permutation for now):
//!   x = [θ(nb), Vm(nb), Pg(ng), Qg(ng)]   nx = 2nb + 2ng
//!   eq = [P(nb), Q(nb), linear-eq(neqlin)]   neq = 2nb + neqlin
//!   K  = [ M  dg ; dgᵀ 0 ]   dim = nx + neq
//!
//! Per-column row pattern (all derived from N(j) = Ybus column j neighbors):
//!   θ_j  : {i}(Haa) {nb+i}(Hva) {nx+i}(dgPᵀ) {nx+nb+i}(dgQᵀ) [+lineq]
//!   Vm_j : {i}(Hav) {nb+i}(Hvv) {nx+i}            {nx+nb+i}    [+lineq]
//!   Pg_g : {2nb+g}(cost diag) {nx+bus(g)}
//!   Qg_g : {2nb+ng+g}(diag)   {nx+nb+bus(g)}
//!   P_eq j (col nx+j)      : {k}{nb+k} (k∈N(j))  {2nb+g : bus(g)=j}
//!   Q_eq j (col nx+nb+j)   : {k}{nb+k}           {2nb+ng+g : bus(g)=j}
//!   lin-eq r (col nx+2nb+r): {ieq[r]}

use crate::opf::problem::OPFData;

pub struct KKTSymbolicV5 {
    pub dim: usize,
    pub nx: usize,
    pub neq: usize,
    /// Fixed-variable indices (xmin == xmax) → linear equality constraints.
    pub ieq: Vec<usize>,
    pub col_ptrs: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub gens_at_bus: Vec<Vec<usize>>,
    /// For each branch l, the 16 indices into KKT values for the 4x4 Hessian block
    /// [θf,θf thf,vmf ...] in variable-column order.
    pub br_to_kkt: Vec<[usize; 16]>,
}

impl KKTSymbolicV5 {
    /// Build from an OPFData in natural (unpermuted) variable/bus order.
    pub fn build(data: &OPFData) -> Self {
        let nx = data.nx();
        let (xmin, xmax) = data.bounds();
        let fixed: Vec<usize> = (0..nx)
            .filter(|&i| (xmax[i] - xmin[i]).abs() <= f64::EPSILON)
            .collect();
        Self::from_parts(
            data.nb,
            data.ng,
            data.nl,
            data.f_buses.as_slice(),
            data.t_buses.as_slice(),
            data.ybus.col_offsets(),
            data.ybus.row_indices(),
            &data.gen_bus,
            &fixed,
        )
    }

    /// Build the KKT skeleton from raw structural parts. Works for any bus/gen ordering
    /// (natural or permuted): `y_cp`/`y_ri` is the (possibly permuted) Ybus structure,
    /// `gen_bus[g]` is the (possibly permuted) bus index of generator `g`, and
    /// `fixed_vars` is the ascending list of fixed variable indices (→ linear eqs).
    pub fn from_parts(
        nb: usize,
        ng: usize,
        nl: usize,
        f_buses: &[usize],
        t_buses: &[usize],
        y_cp: &[usize],
        y_ri: &[usize],
        gen_bus: &[usize],
        fixed_vars: &[usize],
    ) -> Self {
        let nx = 2 * nb + 2 * ng;

        let ieq: Vec<usize> = fixed_vars.to_vec();
        let neqlin = ieq.len();
        let neq = 2 * nb + neqlin;
        let dim = nx + neq;

        // var → linear-eq column offset r (usize::MAX = not fixed)
        let mut var_to_lineq = vec![usize::MAX; nx];
        for (r, &v) in ieq.iter().enumerate() {
            var_to_lineq[v] = r;
        }

        // gens attached at each bus, ascending g
        let mut gens_at_bus: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for g in 0..ng {
            gens_at_bus[gen_bus[g]].push(g);
        }

        let mut col_ptrs = vec![0usize; dim + 1];
        let mut row_idx: Vec<usize> = Vec::new();

        // ── variable columns ──────────────────────────────────────────────────
        // θ_j  (c = j, j < nb)
        for j in 0..nb {
            let nbr = &y_ri[y_cp[j]..y_cp[j + 1]];
            for &i in nbr { row_idx.push(i); }              // Haa
            for &i in nbr { row_idx.push(nb + i); }         // Hva
            for &i in nbr { row_idx.push(nx + i); }         // dgPᵀ
            for &i in nbr { row_idx.push(nx + nb + i); }    // dgQᵀ
            if var_to_lineq[j] != usize::MAX {
                row_idx.push(nx + 2 * nb + var_to_lineq[j]);
            }
            col_ptrs[j + 1] = row_idx.len();
        }
        // Vm_j (c = nb + j, j < nb)
        for j in 0..nb {
            let nbr = &y_ri[y_cp[j]..y_cp[j + 1]];
            for &i in nbr { row_idx.push(i); }              // Hav
            for &i in nbr { row_idx.push(nb + i); }         // Hvv
            for &i in nbr { row_idx.push(nx + i); }         // dgPᵀ
            for &i in nbr { row_idx.push(nx + nb + i); }    // dgQᵀ
            if var_to_lineq[nb + j] != usize::MAX {
                row_idx.push(nx + 2 * nb + var_to_lineq[nb + j]);
            }
            col_ptrs[nb + j + 1] = row_idx.len();
        }
        // Pg_g (c = 2*nb + g)
        for g in 0..ng {
            let bus = gen_bus[g];
            row_idx.push(2 * nb + g);            // cost diag
            row_idx.push(nx + bus);              // coupling to P_eq_bus
            if var_to_lineq[2 * nb + g] != usize::MAX {
                row_idx.push(nx + 2 * nb + var_to_lineq[2 * nb + g]);
            }
            col_ptrs[2 * nb + g + 1] = row_idx.len();
        }
        // Qg_g (c = 2*nb + ng + g)
        for g in 0..ng {
            let bus = gen_bus[g];
            row_idx.push(2 * nb + ng + g);       // structural diag
            row_idx.push(nx + nb + bus);         // coupling to Q_eq_bus
            if var_to_lineq[2 * nb + ng + g] != usize::MAX {
                row_idx.push(nx + 2 * nb + var_to_lineq[2 * nb + ng + g]);
            }
            col_ptrs[2 * nb + ng + g + 1] = row_idx.len();
        }

        // ── constraint columns ────────────────────────────────────────────────
        // P_eq_i (c = nx + i, i < nb)
        for i in 0..nb {
            let nbr = &y_ri[y_cp[i]..y_cp[i + 1]];
            for &k in nbr { row_idx.push(k); }              // dP/dθ_k
            for &k in nbr { row_idx.push(nb + k); }         // dP/dVm_k
            for &g in &gens_at_bus[i] {
                row_idx.push(2 * nb + g);                   // dP/dPg
            }
            col_ptrs[nx + i + 1] = row_idx.len();
        }
        // Q_eq_i (c = nx + nb + i)
        for i in 0..nb {
            let nbr = &y_ri[y_cp[i]..y_cp[i + 1]];
            for &k in nbr { row_idx.push(k); }              // dQ/dθ_k
            for &k in nbr { row_idx.push(nb + k); }         // dQ/dVm_k
            for &g in &gens_at_bus[i] {
                row_idx.push(2 * nb + ng + g);              // dQ/dQg
            }
            col_ptrs[nx + nb + i + 1] = row_idx.len();
        }
        // lin-eq r (c = nx + 2*nb + r)
        for r in 0..neqlin {
            row_idx.push(ieq[r]);
            col_ptrs[nx + 2 * nb + r + 1] = row_idx.len();
        }

        let mut find_k = |r: usize, c: usize| -> usize {
            let s = col_ptrs[c]; let e = col_ptrs[c+1];
            row_idx[s..e].binary_search(&r).map(|p| s + p).expect("KKT element missing")
        };

        let mut br_to_kkt = vec![[0usize; 16]; nl];
        for l in 0..nl {
            let f = f_buses[l]; let t = t_buses[l];
            let buses = [f, t];
            let vars = [0, nb]; // θ offset, Vm offset
            let mut ptrs = [0usize; 16];
            for ni in 0..2 { // from bus (f or t)
                for nj in 0..2 { // to variable (θ or Vm)
                    let c_bus = buses[ni];
                    let c_var_off = vars[nj];
                    let col = c_bus + c_var_off;
                    
                    let nbr_range = y_cp[c_bus]..y_cp[c_bus+1];
                    let deg = nbr_range.len();
                    
                    for row_node_idx in 0..2 { // to row bus (f or t)
                        let r_bus = buses[row_node_idx];
                        let r_pos = y_ri[nbr_range.clone()].binary_search(&r_bus).expect("Branch neighbor missing");
                        
                        for row_var_idx in 0..2 { // to row variable (θ or Vm)
                            let r_var_off = vars[row_var_idx];
                            // Row index in KKT column `col`:
                            // θ_j column has: [Haa | Hva | dgP | dgQ]
                            // Vm_j column has: [Hav | Hvv | dgP | dgQ]
                            let kkt_row_pos = col_ptrs[col] + (row_var_idx * deg) + r_pos;
                            
                            // Map [ni, nj, row_node_idx, row_var_idx] to 0..16
                            // Order: [θf,θf θf,vmf θf,θt θf,vmt | vmf,θf ...]
                            // ni*8 + nj*4 + row_node_idx*2 + row_var_idx
                            ptrs[ni*8 + nj*4 + row_node_idx*2 + row_var_idx] = kkt_row_pos;
                        }
                    }
                }
            }
            br_to_kkt[l] = ptrs;
        }

        Self { dim, nx, neq, ieq, col_ptrs, row_idx, gens_at_bus, br_to_kkt }
    }

    /// Optimized streaming fill. Writes numerical values directly into `kkt_vals`
    /// at the locations determined by the symbolic skeleton.
    ///
    /// Values are provided as standard CSC parts from separate matrices (legacy V4).
    #[allow(clippy::too_many_arguments)]
    pub fn fill(
        &self,
        lxx_cp: &[usize], lxx_v: &[f64],
        dg_cp: &[usize], dg_v: &[f64],
        dgt_cp: &[usize], dgt_v: &[f64],
        kkt_vals: &mut [f64],
    ) {
        let nx = self.nx;
        let nb = (self.neq - self.ieq.len()) / 2;

        // var → linear-eq presence (value is always 1.0)
        let mut is_fixed = vec![false; nx];
        for &v in &self.ieq { is_fixed[v] = true; }

        let mut ptr = 0usize;

        // ── variable columns c ∈ [0, nx): M run, then dgᵀ run, then optional lineq ──
        for c in 0..nx {
            for idx in lxx_cp[c]..lxx_cp[c + 1] {
                kkt_vals[ptr] = lxx_v[idx];
                ptr += 1;
            }
            for idx in dgt_cp[c]..dgt_cp[c + 1] {
                kkt_vals[ptr] = dgt_v[idx];
                ptr += 1;
            }
            if is_fixed[c] {
                kkt_vals[ptr] = 1.0;
                ptr += 1;
            }
        }

        // ── constraint columns: P_eq (col nx+j) and Q_eq (col nx+nb+j) = dg columns ──
        for j in 0..2 * nb {
            for idx in dg_cp[j]..dg_cp[j + 1] {
                kkt_vals[ptr] = dg_v[idx];
                ptr += 1;
            }
        }

        // ── linear-equality columns: single unit entry each ──
        for _ in 0..self.ieq.len() {
            kkt_vals[ptr] = 1.0;
            ptr += 1;
        }

        debug_assert_eq!(ptr, self.row_idx.len(), "V5 fill wrote wrong nnz count");
    }

    /// Streaming fill for the PIPS merged-slack path. Here `dg`/`dg_t` are the **merged**
    /// equality Jacobian (nx × neq, columns = [P | Q | linear-eq]) and its transpose
    /// (neq × nx). Because the linear-equality columns/rows are already present in the
    /// merged matrices (unit `ae` entries), no separate lineq handling is needed: each
    /// variable column is `lxx col ++ dgᵀ col`, each constraint column is `dg col`.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_from_merged(
        &self,
        lxx_cp: &[usize], lxx_v: &[f64],
        dg_cp: &[usize], dg_v: &[f64],
        dgt_cp: &[usize], dgt_v: &[f64],
        kkt_vals: &mut [f64],
    ) {
        let nx = self.nx;
        let neq = self.neq;
        let mut ptr = 0usize;
        for c in 0..nx {
            for idx in lxx_cp[c]..lxx_cp[c + 1] { kkt_vals[ptr] = lxx_v[idx]; ptr += 1; }
            for idx in dgt_cp[c]..dgt_cp[c + 1] { kkt_vals[ptr] = dgt_v[idx]; ptr += 1; }
        }
        for j in 0..neq {
            for idx in dg_cp[j]..dg_cp[j + 1] { kkt_vals[ptr] = dg_v[idx]; ptr += 1; }
        }
        debug_assert_eq!(ptr, self.row_idx.len(), "V5 fill_from_merged wrote wrong nnz count");
    }
}

/// [PQ | PV | ext] bus ordering for OPF (reuses the same classification as
/// `warm_x0`/new_pf). Returns `map`: orig bus index → new internal index.
/// Generators live only on PV/ext buses, so this clusters all generator buses
/// into the contiguous tail `[npq, nb)`.
pub fn opf_bus_order(data: &OPFData) -> Vec<usize> {
    let nb = data.nb;
    let mut is_pv = vec![false; nb];
    for &b in &data.gen_bus {
        if b != data.ref_bus {
            is_pv[b] = true;
        }
    }
    let mut order: Vec<usize> = Vec::with_capacity(nb);
    for b in 0..nb { if b != data.ref_bus && !is_pv[b] { order.push(b); } } // PQ
    for b in 0..nb { if is_pv[b] { order.push(b); } }                       // PV
    for b in 0..nb { if b == data.ref_bus { order.push(b); } }              // ext/slack
    let mut map = vec![0usize; nb];
    for (new, &orig) in order.iter().enumerate() { map[orig] = new; }
    map
}

/// Produce the permuted structural parts for `from_parts` given a bus `map`
/// (orig bus → new internal idx): permuted Ybus structure, generator buses (gens
/// reordered ascending by new bus → contiguous tail), and fixed-variable list.
pub fn permute_for_v5(
    data: &OPFData,
    map: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    use nalgebra_sparse::{CooMatrix, CscMatrix};
    let nb = data.nb;
    let ng = data.ng;
    let nx = data.nx();
    let yb = &data.ybus;

    // permuted Ybus structure (values dummy; CscMatrix::from canonicalizes/sorts)
    let mut coo = CooMatrix::<f64>::new(nb, nb);
    for j in 0..nb {
        for idx in yb.col_offsets()[j]..yb.col_offsets()[j + 1] {
            let i = yb.row_indices()[idx];
            coo.push(map[i], map[j], 1.0);
        }
    }
    let yp = CscMatrix::from(&coo);
    let y_cp = yp.col_offsets().to_vec();
    let y_ri = yp.row_indices().to_vec();

    // generators sorted by new bus → gen g' is contiguous in the tail
    let mut gen_order: Vec<usize> = (0..ng).collect();
    gen_order.sort_by_key(|&g| map[data.gen_bus[g]]);
    let gen_bus_new: Vec<usize> = gen_order.iter().map(|&g| map[data.gen_bus[g]]).collect();
    let mut inv_gen = vec![0usize; ng];
    for (newg, &orig) in gen_order.iter().enumerate() { inv_gen[orig] = newg; }

    // map a natural variable index → new variable index
    let var_new = |v: usize| -> usize {
        if v < nb { map[v] }
        else if v < 2 * nb { nb + map[v - nb] }
        else if v < 2 * nb + ng { 2 * nb + inv_gen[v - 2 * nb] }
        else { 2 * nb + ng + inv_gen[v - 2 * nb - ng] }
    };

    let (xmin, xmax) = data.bounds();
    let mut fixed_new = Vec::new();
    for v in 0..nx {
        if (xmax[v] - xmin[v]).abs() <= f64::EPSILON {
            fixed_new.push(var_new(v));
        }
    }
    fixed_new.sort_unstable();

    (y_cp, y_ri, gen_bus_new, fixed_new)
}

/// Precomputed transpose of `dg` (nx×2nb → 2nb×nx) as a reusable structure + a
/// source-index map, so the per-iteration transpose is a single sequential pass
/// `dgt_vals[i] = dg_vals[src[i]]` into a reused buffer — no allocation, no sort.
///
/// `dg`'s structure is iteration-invariant (Ybus + gen topology), so this is built once.
pub struct DgTransposeCache {
    pub col_ptrs: Vec<usize>, // dg_t (2nb×nx) column pointers, length nx+1
    pub row_idx: Vec<usize>,  // dg_t row indices (= constraint indices), ascending per col
    pub src: Vec<usize>,      // dgt nnz i  ←  dg nnz src[i]
    pub nnz: usize,
}

impl DgTransposeCache {
    /// Build from a representative `dg` (nx×2nb). Standard counting transpose; because we
    /// scan `dg` in constraint-column order, each dg_t column's rows come out ascending.
    pub fn analyze(dg: &nalgebra_sparse::CscMatrix<f64>) -> Self {
        let nx = dg.nrows();
        let nnz = dg.nnz();
        let dg_cp = dg.col_offsets();
        let dg_ri = dg.row_indices();

        let mut col_ptrs = vec![0usize; nx + 1];
        for &r in dg_ri { col_ptrs[r + 1] += 1; }
        for i in 0..nx { col_ptrs[i + 1] += col_ptrs[i]; }

        let mut row_idx = vec![0usize; nnz];
        let mut src = vec![0usize; nnz];
        let mut pos = col_ptrs.clone();
        for con in 0..dg.ncols() {
            for idx in dg_cp[con]..dg_cp[con + 1] {
                let var = dg_ri[idx];
                let p = pos[var];
                row_idx[p] = con; // dg_t row = constraint index
                src[p] = idx;
                pos[var] += 1;
            }
        }
        Self { col_ptrs, row_idx, src, nnz }
    }

    /// Sequential transpose into a reused buffer (length `nnz`): no allocation.
    #[inline]
    pub fn apply(&self, dg_vals: &[f64], dgt_vals: &mut [f64]) {
        for i in 0..self.nnz {
            dgt_vals[i] = dg_vals[self.src[i]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opf::builder::opf_data_from_network;
    use crate::opf::pips::build_saddle_point;
    use crate::io::pandapower::load_csv_zip;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    /// V5 symbolic skeleton structure must be byte-identical to what the legacy
    /// `build_saddle_point` produces (same nonzero set ⇒ same canonical CSC).
    #[test]
    fn test_v5_skeleton_matches_build_saddle_point_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net = load_csv_zip(&format!("{}/cases/IEEE118/data.zip", dir)).unwrap();
        let data = opf_data_from_network(&net);

        let v5 = KKTSymbolicV5::build(&data);
        let nx = data.nx();
        let nb = data.nb;

        // === Build the reference KKT structure the legacy way ===
        // M structure = v4 Lxx template (real fill at warm_x0, dummy multipliers).
        let x = data.warm_x0();
        let lam = vec![0.1; 2 * nb];
        let mu = vec![0.05; 2 * data.nl];
        let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);
        let m = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
            &data, &v3c, x.as_slice(), &lam, &mu, None, 1e-4,
        );

        // dg = [dgn | aeᵀ], nx × (2nb + neqlin)
        let (_, _, dgn, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
        let neqlin = v5.ieq.len();
        let mut dg_coo = CooMatrix::<f64>::new(nx, 2 * nb + neqlin);
        for j in 0..dgn.ncols() {
            for idx in dgn.col_offsets()[j]..dgn.col_offsets()[j + 1] {
                dg_coo.push(dgn.row_indices()[idx], j, dgn.values()[idx]);
            }
        }
        for (r, &v) in v5.ieq.iter().enumerate() {
            dg_coo.push(v, 2 * nb + r, 1.0);
        }
        let dg = CscMatrix::from(&dg_coo);

        let neq = 2 * nb + neqlin;
        let ref_kkt = build_saddle_point(&m, &Some(dg), nx, neq);

        // === Compare structure ===
        assert_eq!(v5.dim, ref_kkt.nrows(), "dim mismatch");
        assert_eq!(v5.col_ptrs.len(), ref_kkt.col_offsets().len(), "col_ptrs length");
        assert_eq!(&v5.col_ptrs, ref_kkt.col_offsets(), "col_ptrs differ");
        assert_eq!(v5.row_idx.len(), ref_kkt.row_indices().len(), "nnz differ");
        assert_eq!(&v5.row_idx, ref_kkt.row_indices(), "row_indices differ");

        println!(
            "V5 KKT skeleton matches build_saddle_point: dim={}, nnz={}, neqlin={}",
            v5.dim, v5.row_idx.len(), neqlin
        );
    }

    /// V5 streaming `fill` must produce values byte-identical to `build_saddle_point`.
    #[test]
    fn test_v5_fill_matches_build_saddle_point_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let net = load_csv_zip(&format!("{}/cases/IEEE118/data.zip", dir)).unwrap();
        let data = opf_data_from_network(&net);

        let v5 = KKTSymbolicV5::build(&data);
        let nx = data.nx();
        let nb = data.nb;

        let x = data.warm_x0();
        let lam = vec![0.1; 2 * nb];
        let mu = vec![0.05; 2 * data.nl];
        let z = vec![0.7; 2 * data.nl];
        let cm = 1e-4;

        let v3c = crate::new_opf::v3_symbolic::V3SymbolicCache::analyze(&data);
        // M includes the nonlinear branch slack penalty (z provided), exactly as the
        // merged-slack solve path feeds build_saddle_point.
        let lxx = crate::new_opf::v4_numeric_rect::v4_rect_numeric_fill(
            &data, &v3c, x.as_slice(), &lam, &mu, Some(&z), cm,
        );
        let (_, _, dg, _) = crate::opf::constraints::opf_consfcn(&data, x.as_slice());
        let dg_t = dg.transpose();

        // V5 streaming fill (slice API)
        let mut v5_vals = vec![0.0f64; v5.row_idx.len()];
        v5.fill(
            lxx.col_offsets(), lxx.values(),
            dg.col_offsets(), dg.values(),
            dg_t.col_offsets(), dg_t.values(),
            &mut v5_vals,
        );

        // Cached transpose must reproduce dg.transpose() exactly.
        let tcache = DgTransposeCache::analyze(&dg);
        assert_eq!(&tcache.col_ptrs, dg_t.col_offsets(), "DgTransposeCache col_ptrs differ");
        assert_eq!(&tcache.row_idx, dg_t.row_indices(), "DgTransposeCache row_idx differ");
        let mut dgt_buf = vec![0.0f64; tcache.nnz];
        tcache.apply(dg.values(), &mut dgt_buf);
        for (a, b) in dgt_buf.iter().zip(dg_t.values()) {
            assert!((a - b).abs() < 1e-15, "cached transpose value differs");
        }

        // Reference: build_saddle_point with the same M and dg(+ae)
        let neqlin = v5.ieq.len();
        let mut dg_coo = CooMatrix::<f64>::new(nx, 2 * nb + neqlin);
        for j in 0..dg.ncols() {
            for idx in dg.col_offsets()[j]..dg.col_offsets()[j + 1] {
                dg_coo.push(dg.row_indices()[idx], j, dg.values()[idx]);
            }
        }
        for (r, &v) in v5.ieq.iter().enumerate() {
            dg_coo.push(v, 2 * nb + r, 1.0);
        }
        let dg_full = CscMatrix::from(&dg_coo);
        let ref_kkt = build_saddle_point(&lxx, &Some(dg_full), nx, v5.neq);

        // Element-wise compare (structures already proven identical)
        assert_eq!(v5_vals.len(), ref_kkt.values().len());
        let mut max_diff = 0.0f64;
        for (a, b) in v5_vals.iter().zip(ref_kkt.values()) {
            max_diff = max_diff.max((a - b).abs());
        }
        println!("V5 fill vs build_saddle_point: nnz={}, max_diff={:.3e}", v5_vals.len(), max_diff);
        assert!(max_diff < 1e-12, "V5 fill values differ (max_diff={:.3e})", max_diff);
    }
}
