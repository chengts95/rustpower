//! Allocation ledger for one shared base and one representative worker.
use rustpower::{n1::*, prelude::*};
use std::mem::size_of;
fn main() -> Result<(), String> {
    let net = rustpower::io::pandapower::load_csv_zip(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/cases/pegase9241/data.zip"
    ))
    .map_err(|e| e.to_string())?;
    let mut main_world = PowerGrid::default();
    main_world.world_mut().insert_resource(PPNetwork(net));
    main_world.init_pf_net();
    let p = prepare_n1(main_world.world_mut())?;
    let base = p.base().baseline();
    let mut worker = N1Worker::new_ac(base);
    let mut patches = outage::WorkerPatchState::default();
    let mut converged = false;
    for index in p.active_branches() {
        let task = outage::OutageTask::for_ac_branch(base, p.base().catalog(), index, index)?;
        patches.apply(&mut worker, base, &task)?;
        if worker.solve_ac_current(base, 50, 1e-10).converged {
            converged = true;
            break;
        }
    }
    assert!(converged);
    let shared = base.allocation_stats();
    let w = worker.allocation_stats();
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({
  "buses":base.v_init().len(),"y_nnz":base.y_bus().nnz(),"dc_nnz":base.dc_model().values.len(),"j_nnz":base.j_pattern().j_row_indices.len(),
  "shared_pattern_bytes":shared.pattern_bytes,"shared_baseline_value_bytes":shared.baseline_value_bytes,"shared_symbolic_current_bytes":shared.symbolic_bytes,
  "shared_branch_soa_capacity":p.base().catalog().capacity(),
  "worker_vector_bytes":w.vector_bytes,"worker_dc_current_bytes":w.dc.current_bytes,"worker_ac_current_bytes":w.ac.current_bytes,
  "worker_dc_peak_bytes":w.dc.peak_bytes,"worker_ac_peak_bytes":w.ac.peak_bytes,
  "dc_l_nnz":w.dc.l_nonzeros,"dc_u_nnz":w.dc.u_nonzeros,"ac_l_nnz":w.ac.l_nonzeros,"ac_u_nnz":w.ac.u_nonzeros,
  "dc_block_bytes":w.dc.factor_block_bytes,"ac_block_bytes":w.ac.factor_block_bytes,
  "dc_permanent_workspace_bytes":w.dc.permanent_workspace_bytes,"ac_permanent_workspace_bytes":w.ac.permanent_workspace_bytes,
  "group_voltage_bytes_batch8":base.v_init().len()*size_of::<num_complex::Complex64>()*8,
  "watch_hit_size":size_of::<rustpower::n1::watch::WatchHit>(),
  "worker_known_bytes_without_outputs":w.vector_bytes+w.dc.current_bytes+w.ac.current_bytes,
  "scope":"Representative successful AC-only contingency factorization; excludes source ECS, App metadata, thread stacks, queues, outputs and allocator retained pages. factor block/workspace bytes are subsets of KLU current bytes."
 })).unwrap());
    Ok(())
}
