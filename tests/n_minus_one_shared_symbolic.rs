//! Concurrent KLU sharing probe: structure and Symbolic are borrowed from one
//! owner; Common, Numeric, values and RHS are private to each scoped worker.
#![cfg(any(feature = "klu", feature = "klu_dyn"))]
use rustpower::prelude::{
    dcpf::DcpfModel, ecs::powerflow::systems::PowerFlowMat, jacobian_cache::JacobianCache, *,
};
use rustpower_sol_klu::KLUSolver;

struct NumericWorker(KLUSolver);
impl Drop for NumericWorker {
    fn drop(&mut self) {
        // Borrowed Symbolic belongs to the parent. Legacy Drop remains responsible
        // for this worker's own Numeric/Common; the parent frees Symbolic once.
        self.0.symbolic = std::ptr::null_mut();
    }
}
fn analyze(p: &[usize], i: &[usize]) -> KLUSolver {
    let mut owner = KLUSolver::default();
    // KLU ABI uses int64_t*: prepare owned input buffers before publication.
    let mut p: Vec<i64> = p.iter().map(|&v| v as i64).collect();
    let mut i: Vec<i64> = i.iter().map(|&v| v as i64).collect();
    unsafe {
        assert_eq!(
            owner.solve_sym(p.as_mut_ptr(), i.as_mut_ptr(), p.len() as i64 - 1),
            0
        );
    }
    assert!(!owner.symbolic.is_null());
    owner
}
fn exercise(owner: &KLUSolver, p: &[i64], i: &[i64], worker: usize) {
    let n = p.len() - 1;
    let mut numeric = NumericWorker(KLUSolver::default());
    numeric.0.symbolic = owner.symbolic;
    let mut values = vec![0.; i.len()];
    let mut rhs = vec![0.; n];
    for job in 0..24 {
        rhs.fill(0.);
        // Use both actual PF sparsity patterns with nonsingular test values;
        // this isolates FFI sharing from PF convergence and numerical physics.
        for col in 0..n {
            for slot in p[col] as usize..p[col + 1] as usize {
                let row = i[slot] as usize;
                values[slot] = if row == col {
                    (n + 1) as f64 + worker as f64 + job as f64 * 0.1
                } else {
                    -0.25
                };
                rhs[row] += values[slot]; // A * ones
            }
        }
        unsafe {
            // System klu.h documents these CSC/Symbolic inputs as unmodified.
            // Cast only at FFI; shared Rust slices are never made mutable refs.
            let status = if job == 0 {
                numeric.0.factor(
                    p.as_ptr() as *mut i64,
                    i.as_ptr() as *mut i64,
                    values.as_mut_ptr(),
                )
            } else {
                numeric.0.refactor(
                    p.as_ptr() as *mut i64,
                    i.as_ptr() as *mut i64,
                    values.as_mut_ptr(),
                    n as i64,
                )
            };
            assert_eq!(status, 0);
            assert!(!numeric.0.numeric.is_null());
            assert_eq!(numeric.0.symbolic, owner.symbolic);
            assert_eq!(numeric.0.solve(rhs.as_mut_ptr(), n as i64, 1), 0);
        }
        assert!(rhs.iter().all(|&v| (v - 1.).abs() < 1e-10));
    }
}
#[test]
fn four_workers_borrow_the_same_dc_and_ac_symbolics_without_copying() {
    let net = serde_json::from_str::<rustpower::io::pandapower::Network>(
        rustpower::testcases::case_ieee39::IEEE_39,
    )
    .unwrap();
    let mut grid = PowerGrid::default();
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let mat = grid.world().resource::<PowerFlowMat>();
    let ac = JacobianCache::build_from_permuted(
        mat.y_bus.col_offsets(),
        mat.y_bus.row_indices(),
        mat.npv,
        mat.npq,
    );
    let dc = DcpfModel::from_ybus(
        &mat.y_bus,
        &mat.v_bus_init,
        mat.npv + mat.npq,
        &mat.to_perm,
        &vec![0.; mat.y_bus.nrows()],
    );
    let ac_owner = analyze(&ac.j_col_ptrs, &ac.j_row_indices);
    let dc_owner = analyze(&dc.col_ptrs, &dc.row_indices);
    // One immutable pattern per matrix, shared by all workers. No per-worker clone.
    let ac_p: Vec<i64> = ac.j_col_ptrs.iter().map(|&x| x as i64).collect();
    let ac_i: Vec<i64> = ac.j_row_indices.iter().map(|&x| x as i64).collect();
    let dc_p: Vec<i64> = dc.col_ptrs.iter().map(|&x| x as i64).collect();
    let dc_i: Vec<i64> = dc.row_indices.iter().map(|&x| x as i64).collect();
    std::thread::scope(|scope| {
        for worker in 0..4 {
            let (a, d, ap, ai, dp, di) = (&ac_owner, &dc_owner, &ac_p, &ac_i, &dc_p, &dc_i);
            scope.spawn(move || {
                exercise(d, dp, di, worker);
                exercise(a, ap, ai, worker);
            });
        }
    });
    // Parent retains ownership until every worker and its Numeric have dropped.
}
