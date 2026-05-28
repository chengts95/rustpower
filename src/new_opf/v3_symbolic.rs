use nalgebra_sparse::{CscMatrix, CooMatrix};
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// V3 Symbolic Cache: Specialized for Full OPF System
pub struct V3SymbolicCache {
    pub nb: usize,
    pub nx: usize,
    pub y_to_lxx: Vec<[usize; 4]>,
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
            nb, nx, y_to_lxx, br_to_lxx, br_to_yf_idx, br_to_yt_idx, 
            y_transpose_idx, 
            lxx_diag_ptrs: (0..nx).map(|j| find_l(j, j)).collect(),
            lxx_va_diag_ptrs: (0..nb).map(|i| find_l(nb + i, i)).collect(),
            lxx_av_diag_ptrs: (0..nb).map(|i| find_l(i, nb + i)).collect(),
            lxx_cp: l_cp, lxx_ri: l_ri 
        }
    }
}

pub struct KKTSymbolicCache {
    pub dim: usize,
    pub kkt_cp: Vec<usize>,
    pub kkt_ri: Vec<usize>,
    pub lxx_to_kkt: Vec<usize>,
    pub dg_to_kkt: Vec<usize>,
    pub dgt_to_kkt: Vec<usize>,
}

impl KKTSymbolicCache {
    pub fn analyze(lxx_cache: &V3SymbolicCache, data: &OPFData) -> Self {
        let nx = lxx_cache.nx;
        let x_dummy = vec![1.0; nx];
        let (_, _, dg_ref, _) = crate::opf::constraints::opf_consfcn(data, &x_dummy);
        
        let neq = dg_ref.nrows(); // Use the REAL number of equations from consfcn
        let dim = nx + neq;

        let mut k_coo = CooMatrix::<f64>::new(dim, dim);
        
        // 1. Hessian Block [0..nx, 0..nx]
        for j in 0..nx {
            for idx in lxx_cache.lxx_cp[j]..lxx_cache.lxx_cp[j+1] {
                k_coo.push(lxx_cache.lxx_ri[idx], j, 0.0);
            }
        }
        // 2. Jacobian Blocks
        for j in 0..nx {
            for idx in dg_ref.col_offsets()[j]..dg_ref.col_offsets()[j+1] {
                let r = dg_ref.row_indices()[idx];
                k_coo.push(nx + r, j, 0.0); // Bottom-Left
                k_coo.push(j, nx + r, 0.0); // Top-Right
            }
        }

        let kkt = CscMatrix::from(&k_coo);
        let find_k = |r: usize, c: usize| -> usize {
            let s = kkt.col_offsets()[c]; let e = kkt.col_offsets()[c+1];
            kkt.row_indices()[s..e].binary_search(&r).map(|p| s + p).unwrap()
        };

        // 3. Precise Bijective Mapping
        let lxx_to_kkt = (0..nx).flat_map(|j| {
            (lxx_cache.lxx_cp[j]..lxx_cache.lxx_cp[j+1]).map(move |off| find_k(lxx_cache.lxx_ri[off], j))
        }).collect();

        let mut dg_to_kkt = vec![0usize; dg_ref.nnz()];
        let mut dgt_to_kkt = vec![0usize; dg_ref.nnz()];
        for j in 0..nx {
            for idx in dg_ref.col_offsets()[j]..dg_ref.col_offsets()[j+1] {
                let r = dg_ref.row_indices()[idx];
                dg_to_kkt[idx] = find_k(nx + r, j);
                dgt_to_kkt[idx] = find_k(j, nx + r);
            }
        }

        Self { dim, kkt_cp: kkt.col_offsets().to_vec(), kkt_ri: kkt.row_indices().to_vec(), lxx_to_kkt, dg_to_kkt, dgt_to_kkt }
    }
}
