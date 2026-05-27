use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// V3 Symbolic Cache: Uses direct indexing based on CSC structure.
/// Fuses node and branch mappings, allowing O(1) extraction of parameters.
pub struct V3SymbolicCache {
    pub nb: usize,
    pub nx: usize,
    pub y_to_lxx: Vec<[usize; 4]>,
    pub br_to_y_indices: Vec<[usize; 4]>,
    pub br_to_lxx: Vec<[usize; 16]>,
    
    /// Mapping from Branch index `l` to its 2 positions in Yf.values: [(l,f), (l,t)].
    pub br_to_yf_idx: Vec<[usize; 2]>,
    /// Mapping from Branch index `l` to its 2 positions in Yt.values: [(l,f), (l,t)].
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

        // --- 1. Compute Ybus Transpose Mapping ---
        let mut y_transpose_idx = vec![0usize; ybus.nnz()];
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] {
                let i = y_ri[idx];
                let range = y_cp[i]..y_cp[i+1];
                let transpose_idx = y_ri[range.clone()].binary_search(&j)
                    .map(|pos| range.start + pos)
                    .expect("Structural NNZ transpose missing in Ybus!");
                y_transpose_idx[idx] = transpose_idx;
            }
        }

        // --- 2. Build Lxx Template (Sparsity pattern only) ---
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

        // --- 3. Direct Index Mapping: Ybus -> Lxx ---
        let mut y_to_lxx = vec![[0usize; 4]; ybus.nnz()];
        for j in 0..nb {
            let nnz_j = y_cp[j+1] - y_cp[j];
            for (off, _) in (y_cp[j]..y_cp[j+1]).enumerate() {
                let idx = y_cp[j] + off;
                y_to_lxx[idx] = [
                    l_cp[j] + off,           // aa
                    l_cp[nb + j] + off,      // av
                    l_cp[j] + nnz_j + off,   // va
                    l_cp[nb + j] + nnz_j + off, // vv
                ];
            }
        }

        // --- 4. Branch Mapping & Parameter Extraction ---
        let mut br_to_y_indices = vec![[0usize; 4]; nl];
        let mut br_to_lxx = vec![[0usize; 16]; nl];
        let mut br_to_yf_idx = vec![[0usize; 2]; nl];
        let mut br_to_yt_idx = vec![[0usize; 2]; nl];

        let find_l = |r: usize, c: usize| -> usize {
            let range = l_cp[c]..l_cp[c+1];
            l_ri[range.clone()].binary_search(&r).map(|pos| range.start + pos).expect("Lxx missing structural element!")
        };
        
        let find_sparse = |mat: &CscMatrix<Complex64>, r: usize, c: usize| -> usize {
            let range = mat.col_offsets()[c]..mat.col_offsets()[c + 1];
            mat.row_indices()[range.clone()].binary_search(&r)
                .map(|pos| range.start + pos)
                .expect("Matrix missing structural element!")
        };

        for l in 0..nl {
            let f = data.f_buses[l];
            let t = data.t_buses[l];
            
            // Ybus mapping
            br_to_y_indices[l] = [ 
                find_sparse(ybus, f, f), 
                find_sparse(ybus, f, t), 
                find_sparse(ybus, t, f), 
                find_sparse(ybus, t, t) 
            ];
            
            // Yf/Yt extraction mapping
            br_to_yf_idx[l] = [
                find_sparse(&data.yf, l, f),
                find_sparse(&data.yf, l, t)
            ];
            br_to_yt_idx[l] = [
                find_sparse(&data.yt, l, f),
                find_sparse(&data.yt, l, t)
            ];

            // Lxx mapping
            let nodes = [f, t];
            let mut ptrs = [0usize; 16];
            for ni in 0..2 {
                for nj in 0..2 {
                    let r = nodes[ni];
                    let c = nodes[nj];
                    let base = (ni * 2 + nj) * 4;
                    ptrs[base + 0] = find_l(r, c);           // aa
                    ptrs[base + 1] = find_l(r, nb + c);      // av
                    ptrs[base + 2] = find_l(nb + r, c);      // va
                    ptrs[base + 3] = find_l(nb + r, nb + c); // vv
                }
            }
            br_to_lxx[l] = ptrs;
        }

        let lxx_diag_ptrs = (0..nx).map(|j| find_l(j, j)).collect();
        let lxx_va_diag_ptrs = (0..nb).map(|i| find_l(nb + i, i)).collect();
        let lxx_av_diag_ptrs = (0..nb).map(|i| find_l(i, nb + i)).collect();

        Self { 
            nb, nx, y_to_lxx, br_to_y_indices, br_to_lxx, 
            br_to_yf_idx, br_to_yt_idx,
            y_transpose_idx, lxx_diag_ptrs, lxx_va_diag_ptrs, lxx_av_diag_ptrs, lxx_cp: l_cp, lxx_ri: l_ri 
        }
    }
}
