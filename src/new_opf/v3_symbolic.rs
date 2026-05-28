use nalgebra_sparse::{CscMatrix, CooMatrix};
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// V3 Symbolic Cache: Uses direct indexing based on CSC structure.
pub struct V3SymbolicCache {
    pub nb: usize,
    pub nx: usize,
    pub y_to_lxx: Vec<[usize; 4]>,
    pub br_to_y_indices: Vec<[usize; 4]>,
    pub br_to_lxx: Vec<[usize; 16]>,
    pub br_to_yf_idx: Vec<[usize; 2]>,
    pub br_to_yt_idx: Vec<[usize; 2]>,
    pub y_transpose_idx: Vec<usize>,
    pub lxx_diag_ptrs: Vec<usize>,
    pub lxx_va_diag_ptrs: Vec<usize>,
    pub lxx_av_diag_ptrs: Vec<usize>,
    pub lxx_cp: Vec<usize>,
    pub lxx_ri: Vec<usize>,
}

impl V3SymbolicCache {
    pub fn analyze(data: &OPFData) -> Self {
        let nb = data.nb;
        let nx = data.nx();
        let ybus = &data.ybus;
        let nl = data.nl;
        let y_cp = ybus.col_offsets();
        let y_ri = ybus.row_indices();

        let mut y_transpose_idx = vec![0usize; ybus.nnz()];
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] {
                let i = y_ri[idx];
                let range = y_cp[i]..y_cp[i+1];
                let transpose_idx = y_ri[range.clone()].binary_search(&j)
                    .map(|pos| range.start + pos).expect("Transpose missing");
                y_transpose_idx[idx] = transpose_idx;
            }
        }

        let mut l_cp = vec![0usize; nx + 1];
        let mut l_ri = Vec::new();
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(y_ri[idx]); }
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(nb + y_ri[idx]); }
            l_cp[j + 1] = l_ri.len();
        }
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(y_ri[idx]); }
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(nb + y_ri[idx]); }
            l_cp[nb + j + 1] = l_ri.len();
        }
        for g in 0..2 * data.ng {
            l_ri.push(2 * nb + g);
            l_cp[2 * nb + g + 1] = l_ri.len();
        }

        let mut y_to_lxx = vec![[0usize; 4]; ybus.nnz()];
        for j in 0..nb {
            let nnz_j = y_cp[j+1] - y_cp[j];
            for off in 0..nnz_j {
                let idx = y_cp[j] + off;
                y_to_lxx[idx] = [l_cp[j]+off, l_cp[nb+j]+off, l_cp[j]+nnz_j+off, l_cp[nb+j]+nnz_j+off];
            }
        }

        let find_l = |r: usize, c: usize| -> usize {
            let range = l_cp[c]..l_cp[c+1];
            l_ri[range.clone()].binary_search(&r).map(|pos| range.start + pos).unwrap()
        };
        let find_sparse = |mat: &CscMatrix<Complex64>, r: usize, c: usize| -> usize {
            let range = mat.col_offsets()[c]..mat.col_offsets()[c + 1];
            mat.row_indices()[range.clone()].binary_search(&r).map(|pos| range.start + pos).unwrap()
        };

        let mut br_to_lxx = vec![[0usize; 16]; nl];
        let mut br_to_yf_idx = vec![[0usize; 2]; nl];
        let mut br_to_yt_idx = vec![[0usize; 2]; nl];
        for l in 0..nl {
            let f = data.f_buses[l]; let t = data.t_buses[l];
            br_to_yf_idx[l] = [ find_sparse(&data.yf, l, f), find_sparse(&data.yf, l, t) ];
            br_to_yt_idx[l] = [ find_sparse(&data.yt, l, f), find_sparse(&data.yt, l, t) ];
            let nodes = [f, t];
            let mut ptrs = [0usize; 16];
            for ni in 0..2 { for nj in 0..2 {
                let r = nodes[ni]; let c = nodes[nj]; let base = (ni * 2 + nj) * 4;
                ptrs[base+0]=find_l(r,c); ptrs[base+1]=find_l(r,nb+c); ptrs[base+2]=find_l(nb+r,c); ptrs[base+3]=find_l(nb+r,nb+c);
            }}
            br_to_lxx[l] = ptrs;
        }

        Self { 
            nb, nx, y_to_lxx, br_to_y_indices: vec![], br_to_lxx, br_to_yf_idx, br_to_yt_idx, 
            y_transpose_idx, lxx_diag_ptrs: (0..nx).map(|j| find_l(j, j)).collect(),
            lxx_va_diag_ptrs: (0..nb).map(|i| find_l(nb + i, i)).collect(),
            lxx_av_diag_ptrs: (0..nb).map(|i| find_l(i, nb + i)).collect(),
            lxx_cp: l_cp, lxx_ri: l_ri 
        }
    }
}

pub struct KKTSymbolicCache {
    pub dim: usize,
    pub kkt_skeleton: CscMatrix<f64>,
    pub lxx_to_kkt: Vec<usize>,
    pub dg_to_kkt: Vec<usize>,
    pub dgt_to_kkt: Vec<usize>,
}

impl KKTSymbolicCache {
    pub fn fill_kkt(&self, kkt_vals: &mut [f64], lxx: &CscMatrix<f64>, dgn: &CscMatrix<f64>) {
        kkt_vals.fill(0.0);
        
        // 1. Fill Lxx
        let lxx_v = lxx.values();
        for (idx, &v) in lxx_v.iter().enumerate() {
            kkt_vals[self.lxx_to_kkt[idx]] = v;
        }
        
        // 2. Fill dg and dgt
        let dg_v = dgn.values();
        for (idx, &v) in dg_v.iter().enumerate() {
            kkt_vals[self.dg_to_kkt[idx]] = v;
            kkt_vals[self.dgt_to_kkt[idx]] = v;
        }
    }

    pub fn analyze(lxx_cache: &V3SymbolicCache, data: &OPFData) -> Self {
        let nx = lxx_cache.nx;
        let x0 = vec![1.0; nx];
        let (_, gn, _, dgn) = crate::opf::constraints::opf_consfcn(data, &x0);
        let (xmin, xmax) = data.bounds();
        let mut ieq = Vec::new();
        for i in 0..nx { if (xmax[i] - xmin[i]).abs() <= f64::EPSILON { ieq.push(i); } }
        let neqlin = ieq.len();
        let neqnln = gn.len();
        let neq = neqnln + neqlin;

        let ae = if neqlin > 0 {
            let mut s = ieq.iter().enumerate().map(|(r, &i)| (r, i, 1.0)).collect::<Vec<_>>();
            s.sort_unstable_by_key(|&(_, c, _)| c);
            let mut cp = vec![0usize; nx + 1];
            for &(_, c, _) in &s { cp[c+1] += 1; }
            for j in 0..nx { cp[j+1] += cp[j]; }
            let mut ri = Vec::new(); let mut v = Vec::new();
            for &(r, _, val) in &s { ri.push(r); v.push(val); }
            Some(CscMatrix::try_from_csc_data(neqlin, nx, cp, ri, v).unwrap())
        } else { None };

        let dg_final = match ae {
            Some(r) => {
                let rt = r.transpose();
                let mut cp = vec![0usize; dgn.ncols() + rt.ncols() + 1];
                cp[..dgn.ncols()+1].copy_from_slice(dgn.col_offsets());
                let nnza = dgn.nnz();
                for j in 0..rt.ncols() { cp[dgn.ncols()+j+1] = nnza + rt.col_offsets()[j+1]; }
                let ri = [dgn.row_indices(), rt.row_indices()].concat();
                let v = [dgn.values(), rt.values()].concat();
                CscMatrix::try_from_csc_data(nx, dgn.ncols() + rt.ncols(), cp, ri, v).unwrap()
            }
            None => dgn.clone(),
        };

        let dummy_lxx = CscMatrix::try_from_csc_data(nx, nx, lxx_cache.lxx_cp.clone(), lxx_cache.lxx_ri.clone(), vec![1.0; lxx_cache.lxx_ri.len()]).unwrap();
        let kkt = crate::opf::pips::build_saddle_point(&dummy_lxx, &Some(dg_final.clone()), nx, neq);
        let kkt_cp = kkt.col_offsets();
        let kkt_ri = kkt.row_indices();
        let find_k = |r: usize, c: usize| -> usize {
            let s = kkt_cp[c]; let e = kkt_cp[c+1];
            kkt_ri[s..e].binary_search(&r).map(|p| s + p).expect("KKT element missing")
        };

        let lxx_to_kkt = (0..nx).flat_map(|j| (lxx_cache.lxx_cp[j]..lxx_cache.lxx_cp[j+1]).map(move |off| find_k(lxx_cache.lxx_ri[off], j))).collect();
        let mut dg_to_kkt = Vec::with_capacity(dgn.nnz());
        let mut dgt_to_kkt = Vec::with_capacity(dgn.nnz());
        let dgn_cp = dgn.col_offsets();
        let dgn_ri = dgn.row_indices();
        for j in 0..dgn.ncols() {
            for idx in dgn_cp[j]..dgn_cp[j+1] {
                let var_i = dgn_ri[idx];
                dg_to_kkt.push(find_k(nx + j, var_i));
                dgt_to_kkt.push(find_k(var_i, nx + j));
            }
        }
        Self { dim: kkt.nrows(), kkt_skeleton: kkt, lxx_to_kkt, dg_to_kkt, dgt_to_kkt }
    }
}
