//! Proof of fixed-pattern numerical reuse, independent of the in_service gateway.
#![cfg(any(feature = "klu", feature = "klu_dyn"))]
use bevy_ecs::prelude::*;
use rustpower::{
    prelude::ecs::powerflow::{structure_update::FullRebuildEvent, systems::PowerFlowMat},
    prelude::*,
    prelude::{
        ecs::elements::{bus::OutOfService, line::LineID},
        newtonpf::NewtonCache,
        solver::KLUSolver,
    },
    testcases::case_ieee39::IEEE_39,
};

#[test]
fn branch_outage_and_restore_reuse_klu_symbolic_with_fixed_csc() {
    let net = serde_json::from_str::<rustpower::io::pandapower::Network>(IEEE_39).unwrap();
    let mut grid = PowerGrid::default();
    grid.app_mut()
        .add_plugins(rustpower::prelude::ecs::powerflow::structure_update::StructureUpdatePlugin);
    grid.world_mut().insert_resource(PPNetwork(net));
    grid.init_pf_net();
    let mat = grid.world().resource::<PowerFlowMat>();
    let base = mat.y_bus.clone();
    let s = mat.s_bus.clone();
    let v = mat.v_bus_init.clone();
    let (npv, npq) = (mat.npv, mat.npq);
    let mut solver = KLUSolver::default();
    let mut cache = NewtonCache::default();
    let baseline = rustpower::prelude::newton_pf(
        &base,
        &s,
        &v,
        npv,
        npq,
        Some(1e-10),
        Some(50),
        &mut solver,
        Some(&mut cache),
    )
    .unwrap()
    .0;
    let symbolic = solver.0.symbolic;
    assert!(!symbolic.is_null());
    let j_offsets = cache.j_pattern.as_ref().unwrap().j_col_ptrs.as_ptr();
    let line = grid
        .world_mut()
        .query::<(Entity, &LineID)>()
        .iter(grid.world())
        .find(|(_, id)| id.0 == 0)
        .unwrap()
        .0;
    grid.world_mut().entity_mut(line).insert(OutOfService);
    grid.world_mut().write_message(FullRebuildEvent);
    grid.run_pf();
    assert!(
        grid.world()
            .resource::<rustpower::prelude::ecs::powerflow::structure_update::LastStructureAction>()
            .full_rebuild
    );
    let outage = grid.world().resource::<PowerFlowMat>();
    assert_eq!((npv, npq), (outage.npv, outage.npq));
    assert_eq!(outage.s_bus, s);
    let mut frozen = base.clone();
    for col in 0..base.ncols() {
        let lo = outage.y_bus.col_offsets()[col];
        let hi = outage.y_bus.col_offsets()[col + 1];
        let rows = &outage.y_bus.row_indices()[lo..hi];
        for &row in rows {
            assert!(
                base.row_indices()[base.col_offsets()[col]..base.col_offsets()[col + 1]]
                    .binary_search(&row)
                    .is_ok()
            );
        }
        for slot in base.col_offsets()[col]..base.col_offsets()[col + 1] {
            frozen.values_mut()[slot] = rows
                .binary_search(&base.row_indices()[slot])
                .map_or(num_complex::Complex64::new(0., 0.), |k| {
                    outage.y_bus.values()[lo + k]
                });
        }
    }
    assert_eq!(base.col_offsets(), frozen.col_offsets());
    assert_eq!(base.row_indices(), frozen.row_indices());
    let mut reference_solver = KLUSolver::default();
    let reference = rustpower::prelude::newton_pf(
        &outage.y_bus,
        &s,
        &v,
        npv,
        npq,
        Some(1e-10),
        Some(50),
        &mut reference_solver,
        None,
    )
    .unwrap()
    .0;
    let actual = rustpower::prelude::newton_pf(
        &frozen,
        &s,
        &v,
        npv,
        npq,
        Some(1e-10),
        Some(50),
        &mut solver,
        Some(&mut cache),
    )
    .unwrap()
    .0;
    assert!(
        (&actual - reference)
            .iter()
            .map(|z| z.norm())
            .fold(0.0_f64, f64::max)
            < 1e-9
    );
    assert_eq!(symbolic, solver.0.symbolic);
    assert_eq!(
        j_offsets,
        cache.j_pattern.as_ref().unwrap().j_col_ptrs.as_ptr()
    );
    let restored = rustpower::prelude::newton_pf(
        &base,
        &s,
        &v,
        npv,
        npq,
        Some(1e-10),
        Some(50),
        &mut solver,
        Some(&mut cache),
    )
    .unwrap()
    .0;
    assert!(
        (&restored - baseline)
            .iter()
            .map(|z| z.norm())
            .fold(0.0_f64, f64::max)
            < 1e-9
    );
    assert_eq!(symbolic, solver.0.symbolic);
    assert_eq!(
        j_offsets,
        cache.j_pattern.as_ref().unwrap().j_col_ptrs.as_ptr()
    );
}
