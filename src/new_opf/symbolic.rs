use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use crate::opf::problem::OPFData;

pub struct SymbolicCache {
    /// Pointers to diagonal elements in Ybus. NNZ index for (i, i).
    pub y_diag_ptrs: Vec<usize>,

    /// Mapping from Ybus NNZ index to the positions in the full Hessian Lxx.
    /// y_to_lxx[idx] = [ptr_aa, ptr_av, ptr_va, ptr_vv]
    pub y_to_lxx: Vec<[usize; 4]>,

    /// Pointers to diagonal elements in the full Hessian Lxx (nx x nx).
    pub h_diag_ptrs: Vec<usize>,

    /// Mapping from Ybus NNZ index to the positions in the transposed Jacobian dg.
    /// y_to_dg[idx] = [ptr_va_p, ptr_vm_p, ptr_va_q, ptr_vm_q]
    pub y_to_dg: Vec<[usize; 4]>,

    /// Pointers to Pg and Qg terms in dg.
    pub pg_dg_ptrs: Vec<usize>,
    pub qg_dg_ptrs: Vec<usize>,

    /// Mapping from Ybus NNZ index `idx` to the index of its transpose `(j, i)`.
    pub y_transpose_idx: Vec<usize>,

    /// Mapping from branch index `l` to its 16 positions in Lxx.
    /// Order: [ff_aa, ff_av, ff_va, ff_vv, ft_aa, ft_av, ft_va, ft_vv,
    ///         tf_aa, tf_av, tf_va, tf_vv, tt_aa, tt_av, tt_va, tt_vv]
    pub br_to_lxx: Vec<[usize; 16]>,

    /// Mapping from branch index `l` to its positions in dh (flow constraints).
    /// dh is 2nl x nx. (transposed nx x 2nl in PIPS)
    /// We'll implement this if needed.
    pub br_to_dh: Vec<[usize; 8]>,

    /// Pre-allocated CSC template for the full Hessian Lxx.
    pub lxx_template: CscMatrix<f64>,
    
    /// Pre-allocated CSC template for the Jacobian dg (nx x 2nb).
    pub dg_template: CscMatrix<f64>,
}

impl SymbolicCache {
    pub fn analyze(data: &OPFData) -> Self {
        let nb = data.nb;
        let nl = data.nl;
        let ng = data.ng;
        let nx = data.nx();
        let ybus = &data.ybus;

        // --- 1. Build Hessian Template ---
        let mut l_coo_ri: Vec<usize> = Vec::with_capacity(ybus.nnz() * 4 + 2 * ng);
        let mut l_coo_ci: Vec<usize> = Vec::with_capacity(ybus.nnz() * 4 + 2 * ng);

        for j in 0..nb {
            for idx in ybus.col_offsets()[j]..ybus.col_offsets()[j+1] {
                let i = ybus.row_indices()[idx];
                l_coo_ri.push(i);      l_coo_ci.push(j);      // aa
                l_coo_ri.push(i);      l_coo_ci.push(nb + j); // av
                l_coo_ri.push(nb + i); l_coo_ci.push(j);      // va
                l_coo_ri.push(nb + i); l_coo_ci.push(nb + j); // vv
            }
        }
        for g in 0..ng {
            l_coo_ri.push(2 * nb + g);      l_coo_ci.push(2 * nb + g);      // Pg
            l_coo_ri.push(2 * nb + ng + g); l_coo_ci.push(2 * nb + ng + g); // Qg
        }
        let lxx_template = coo_to_csc_f64(nx, nx, &l_coo_ri, &l_coo_ci);

        // --- 2. Build Jacobian Template ---
        let mut g_coo_ri: Vec<usize> = Vec::with_capacity(ybus.nnz() * 4 + 2 * ng);
        let mut g_coo_ci: Vec<usize> = Vec::with_capacity(ybus.nnz() * 4 + 2 * ng);
        for j in 0..nb {
            for idx in ybus.col_offsets()[j]..ybus.col_offsets()[j+1] {
                let i = ybus.row_indices()[idx];
                g_coo_ri.push(i);      g_coo_ci.push(j);      // Va in P_eq j
                g_coo_ri.push(nb + i); g_coo_ci.push(j);      // Vm in P_eq j
                g_coo_ri.push(i);      g_coo_ci.push(nb + j); // Va in Q_eq j
                g_coo_ri.push(nb + i); g_coo_ci.push(nb + j); // Vm in Q_eq j
            }
        }
        for g in 0..ng {
            let bus = data.gen_bus[g];
            g_coo_ri.push(2 * nb + g);      g_coo_ci.push(bus);      // Pg in P_eq bus
            g_coo_ri.push(2 * nb + ng + g); g_coo_ci.push(nb + bus); // Qg in Q_eq bus
        }
        let dg_template = coo_to_csc_f64(nx, 2 * nb, &g_coo_ri, &g_coo_ci);

        // --- 3. Compute Direct Mappings ---
        let find_nnz_f64 = |mat: &CscMatrix<f64>, r: usize, c: usize| -> usize {
            let range = mat.col_offsets()[c]..mat.col_offsets()[c + 1];
            mat.row_indices()[range.clone()].binary_search(&r)
                .map(|pos| range.start + pos)
                .expect("Structural NNZ missing in template!")
        };
        
        let find_nnz_cx = |mat: &CscMatrix<Complex64>, r: usize, c: usize| -> usize {
            let range = mat.col_offsets()[c]..mat.col_offsets()[c + 1];
            mat.row_indices()[range.clone()].binary_search(&r)
                .map(|pos| range.start + pos)
                .expect("Structural NNZ missing in Ybus!")
        };

        let mut y_to_lxx = vec![[0usize; 4]; ybus.nnz()];
        let mut y_to_dg = vec![[0usize; 4]; ybus.nnz()];
        let mut y_transpose_idx = vec![0usize; ybus.nnz()];

        for j in 0..nb {
            for idx in ybus.col_offsets()[j]..ybus.col_offsets()[j+1] {
                let i = ybus.row_indices()[idx];
                y_to_lxx[idx] = [
                    find_nnz_f64(&lxx_template, i, j),           // aa
                    find_nnz_f64(&lxx_template, i, nb + j),      // av
                    find_nnz_f64(&lxx_template, nb + i, j),      // va
                    find_nnz_f64(&lxx_template, nb + i, nb + j), // vv
                ];
                y_to_dg[idx] = [
                    find_nnz_f64(&dg_template, i, j),      // Va_P
                    find_nnz_f64(&dg_template, nb + i, j), // Vm_P
                    find_nnz_f64(&dg_template, i, nb + j), // Va_Q
                    find_nnz_f64(&dg_template, nb + i, nb + j), // Vm_Q
                ];
                y_transpose_idx[idx] = find_nnz_cx(ybus, j, i);
            }
        }

        let pg_dg_ptrs: Vec<usize> = (0..ng).map(|g| find_nnz_f64(&dg_template, 2 * nb + g, data.gen_bus[g])).collect();
        let qg_dg_ptrs: Vec<usize> = (0..ng).map(|g| find_nnz_f64(&dg_template, 2 * nb + ng + g, nb + data.gen_bus[g])).collect();

        let h_diag_ptrs: Vec<usize> = (0..nx).map(|i| find_nnz_f64(&lxx_template, i, i)).collect();

        let mut y_diag_ptrs = vec![0usize; nb];
        for j in 0..nb {
            let start = ybus.col_offsets()[j];
            let end = ybus.col_offsets()[j + 1];
            for idx in start..end {
                if ybus.row_indices()[idx] == j {
                    y_diag_ptrs[j] = idx;
                    break;
                }
            }
        }

        // --- 4. Branch Mapping ---
        let mut br_to_lxx = vec![[0usize; 16]; nl];
        for l in 0..nl {
            let f = data.f_buses[l];
            let t = data.t_buses[l];
            let nodes = [f, t];
            let mut ptrs = [0usize; 16];
            for ni in 0..2 {
                for nj in 0..2 {
                    let r = nodes[ni];
                    let c = nodes[nj];
                    let base = (ni * 2 + nj) * 4;
                    ptrs[base + 0] = find_nnz_f64(&lxx_template, r, c);           // aa
                    ptrs[base + 1] = find_nnz_f64(&lxx_template, r, nb + c);      // av
                    ptrs[base + 2] = find_nnz_f64(&lxx_template, nb + r, c);      // va
                    ptrs[base + 3] = find_nnz_f64(&lxx_template, nb + r, nb + c); // vv
                }
            }
            br_to_lxx[l] = ptrs;
        }

        Self {
            y_diag_ptrs,
            y_to_lxx,
            h_diag_ptrs,
            y_to_dg,
            pg_dg_ptrs,
            qg_dg_ptrs,
            y_transpose_idx,
            br_to_lxx,
            br_to_dh: Vec::new(),
            lxx_template,
            dg_template,
        }
    }
}

fn coo_to_csc_f64(nrows: usize, ncols: usize, ri: &[usize], ci: &[usize]) -> CscMatrix<f64> {
    let mut entries: Vec<(usize, usize, f64)> = ri.iter().zip(ci.iter()).map(|(&r, &c)| (r, c, 0.0)).collect();
    entries.sort_unstable_by_key(|&(r, c, _)| (c, r));
    entries.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    
    let mut c_cp = vec![0usize; ncols + 1];
    let mut c_ri: Vec<usize> = Vec::with_capacity(entries.len());
    let c_v: Vec<f64> = vec![0.0; entries.len()];
    
    for &(r, c, _) in &entries {
        c_cp[c + 1] += 1;
        c_ri.push(r);
    }
    for j in 0..ncols {
        c_cp[j + 1] += c_cp[j];
    }
    CscMatrix::try_from_csc_data(nrows, ncols, c_cp, c_ri, c_v).unwrap()
}
