use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use crate::opf::problem::OPFData;

/// V3 Symbolic Cache: Uses direct indexing based on CSC structure.
/// Fuses node and branch mappings, eliminating the need for Yf, Yt matrices.
pub struct V3SymbolicCache {
    pub nb: usize,
    pub nx: usize,
    pub y_to_lxx: Vec<[usize; 4]>,
    pub br_to_y_indices: Vec<[usize; 4]>,
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
        // Columns j < nb: theta variables
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(y_ri[idx]); }
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(nb + y_ri[idx]); }
            l_cp[j + 1] = l_ri.len();
        }
        // Columns nb..2*nb: Vm variables
        for j in 0..nb {
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(y_ri[idx]); }
            for idx in y_cp[j]..y_cp[j+1] { l_ri.push(nb + y_ri[idx]); }
            l_cp[nb + j + 1] = l_ri.len();
        }
        // Columns 2*nb..nx: Gen variables
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

        // --- 4. Branch Mapping: Branch -> Ybus Indices ---
        let mut br_to_y_indices = vec![[0usize; 4]; nl];
        for l in 0..nl {
            let f = data.f_buses[l];
            let t = data.t_buses[l];
            let find_y = |r: usize, c: usize| -> usize {
                let range = y_cp[c]..y_cp[c+1];
                y_ri[range.clone()].binary_search(&r).map(|pos| range.start + pos).expect("Ybus missing branch topology!")
            };
            br_to_y_indices[l] = [ find_y(f, f), find_y(f, t), find_y(t, f), find_y(t, t) ];
        }

        let find_l = |r: usize, c: usize| -> usize {
            let range = l_cp[c]..l_cp[c+1];
            l_ri[range.clone()].binary_search(&r).map(|pos| range.start + pos).expect("Lxx missing structural element!")
        };

        let lxx_diag_ptrs = (0..nx).map(|j| find_l(j, j)).collect();
        let lxx_va_diag_ptrs = (0..nb).map(|i| find_l(nb + i, i)).collect();
        let lxx_av_diag_ptrs = (0..nb).map(|i| find_l(i, nb + i)).collect();

        Self { nb, nx, y_to_lxx, br_to_y_indices, y_transpose_idx, lxx_diag_ptrs, lxx_va_diag_ptrs, lxx_av_diag_ptrs, lxx_cp: l_cp, lxx_ri: l_ri }
    }
}
