#![cfg(feature = "experimental_n1")]
use bevy_ecs::prelude::*;
use nalgebra::{DVector, Matrix2};
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;
use rustpower::{
    n1::postprocess::{BranchCatalog, BranchKind, PostprocessKernel},
    prelude::{
        branch::BranchFlowResult,
        ecs::{
            elements::{
                BusMatrixIndex, FromBusMatrixIndex, PFCommonData, ToBusMatrixIndex,
                bus::{OutOfService, VNominal},
                line::{LineID, LineParams},
                switch::NodeAggRes,
                trans::{Port4MatPatch, TransformerDevice, TransformerID},
                units::Pair,
            },
            powerflow::systems::PowerFlowMat,
        },
    },
};
fn base_world() -> World {
    let mut world = World::new();
    world.insert_resource(PFCommonData {
        sbase: 100.0,
        f_hz: 50.0,
        wbase: 2.0 * std::f64::consts::PI * 50.0,
    });
    for index in 0..2 {
        world.spawn((
            BusMatrixIndex(index),
            VNominal(Pair(110.0, Default::default())),
        ));
    }
    world.insert_resource(PowerFlowMat {
        y_bus: CscMatrix::try_from_csc_data(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![Complex64::new(0., 0.); 4],
        )
        .unwrap(),
        s_bus: DVector::zeros(2),
        v_bus_init: DVector::from_element(2, Complex64::new(1., 0.)),
        npq: 1,
        npv: 0,
        to_perm: vec![1, 0],
        from_perm: vec![1, 0],
    });
    world
}
fn params() -> LineParams {
    LineParams {
        r_ohm_per_km: 0.1,
        x_ohm_per_km: 0.2,
        g_us_per_km: 0.,
        c_nf_per_km: 0.,
        length_km: 1.,
        df: 1.,
        parallel: 1,
        max_i_ka: 2.,
    }
}
fn patch() -> Matrix2<Complex64> {
    let y = Complex64::new(1., -3.);
    Matrix2::new(y, -y, -y, y)
}
fn add_line(world: &mut World, id: i64, scale: f64, off: bool) -> Entity {
    let mut line = world.spawn((
        LineID(id),
        params(),
        FromBusMatrixIndex(0),
        ToBusMatrixIndex(1),
        Port4MatPatch(patch().scale(scale)),
    ));
    if off {
        line.insert(OutOfService);
    }
    line.id()
}
fn all_zero(r: &BranchFlowResult) {
    assert_eq!(r.p_from_mw, 0.);
    assert_eq!(r.q_from_mvar, 0.);
    assert_eq!(r.p_to_mw, 0.);
    assert_eq!(r.q_to_mvar, 0.);
    assert_eq!(r.pl_mw, 0.);
    assert_eq!(r.ql_mvar, 0.);
    assert_eq!(r.i_from_ka, 0.);
    assert_eq!(r.i_to_ka, 0.);
    assert_eq!(r.i_ka, 0.);
    assert_eq!(r.vm_from_pu, 0.);
    assert_eq!(r.vm_to_pu, 0.);
    assert_eq!(r.va_from_degree, 0.);
    assert_eq!(r.va_to_degree, 0.);
    assert_eq!(r.loading_percent, 0.);
}
#[test]
fn parallel_identity_outage_mask_and_reuse_preserve_shared_catalog() {
    let mut world = base_world();
    add_line(&mut world, 103, 2., false);
    add_line(&mut world, 100, 1., false);
    add_line(&mut world, 700, 1., true);
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    assert_eq!(
        catalog
            .identities()
            .iter()
            .map(|id| id.id)
            .collect::<Vec<_>>(),
        [100, 103, 700]
    );
    assert!(
        catalog
            .identities()
            .iter()
            .all(|id| id.kind == BranchKind::Line)
    );
    assert_eq!(
        (
            catalog.branch(0).unwrap().from_bus,
            catalog.branch(0).unwrap().to_bus
        ),
        (1, 0)
    );
    let voltage = [Complex64::new(0.97, -0.05), Complex64::new(1.02, 0.02)];
    let mut out = vec![];
    catalog.compute(&voltage, None, 100., &mut out).unwrap();
    let base = out.clone();
    assert!(base[0].i_ka > 0.);
    assert!((base[1].p_from_mw - 2. * base[0].p_from_mw).abs() < 1e-9);
    all_zero(&base[2]);
    catalog.compute(&voltage, Some(0), 100., &mut out).unwrap();
    all_zero(&out[0]);
    assert!((out[1].p_from_mw - base[1].p_from_mw).abs() < 1e-12);
    assert!(catalog.branch(0).unwrap().in_service);
    catalog.compute(&voltage, None, 100., &mut out).unwrap();
    assert!((out[0].p_from_mw - base[0].p_from_mw).abs() < 1e-12);
    let current = patch() * nalgebra::Vector2::new(voltage[1], voltage[0]);
    let sf = voltage[1] * current[0].conj() * 110. * 110.;
    assert!((out[0].p_from_mw - sf.re).abs() < 1e-9);
    assert!((out[0].i_from_ka - current[0].norm() * 110. / 3_f64.sqrt()).abs() < 1e-10);
    assert!((out[0].loading_percent - out[0].i_ka / 2. * 100.).abs() < 1e-10);
}
#[test]
fn transformer_uses_current_based_loading_and_parallel_rating() {
    let mut world = base_world();
    let dev = TransformerDevice {
        df: 1.,
        i0_percent: 0.,
        pfe_kw: 0.,
        vk_percent: 10.,
        vkr_percent: 0.5,
        shift_degree: 0.,
        sn_mva: 40.,
        vn_hv_kv: 110.,
        vn_lv_kv: 20.,
        max_loading_percent: None,
        parallel: 2,
        tap: None,
    };
    world.spawn((
        TransformerID(901),
        dev,
        FromBusMatrixIndex(0),
        ToBusMatrixIndex(1),
        Port4MatPatch(patch()),
    ));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let voltage = [Complex64::new(0.85, -0.04), Complex64::new(1.03, 0.01)];
    let mut out = vec![];
    catalog.compute(&voltage, None, 100., &mut out).unwrap();
    let current = patch() * nalgebra::Vector2::new(voltage[1], voltage[0]);
    let expected = current[0].norm().max(current[1].norm()) * 20. * 20. / 80. * 100.;
    assert!((out[0].loading_percent - expected).abs() < 1e-9);
    assert!((out[0].i_from_ka - current[0].norm() * 400. / (3_f64.sqrt() * 110.)).abs() < 1e-10);
    assert!((out[0].i_to_ka - current[1].norm() * 20. / 3_f64.sqrt()).abs() < 1e-10);
    catalog.compute(&voltage, Some(0), 100., &mut out).unwrap();
    all_zero(&out[0]);
}
#[test]
fn merged_bus_mapping_and_input_errors_are_explicit() {
    let mut world = base_world();
    world.spawn((BusMatrixIndex(2), VNominal(Pair(110.0, Default::default()))));
    world.insert_resource(NodeAggRes {
        original_to_merged: vec![0, 0, 1],
        ..Default::default()
    });
    let line = add_line(&mut world, 100, 1., false);
    world.entity_mut(line).insert(ToBusMatrixIndex(2));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    assert_eq!(
        (
            catalog.branch(0).unwrap().from_bus,
            catalog.branch(0).unwrap().to_bus
        ),
        (1, 0)
    );
    let mut out = vec![];
    assert!(catalog.compute(&[], None, 100., &mut out).is_err());
    assert!(
        catalog
            .compute(&[Complex64::new(1., 0.); 2], Some(1), 100., &mut out)
            .is_err()
    );
    assert!(
        catalog
            .compute(&[Complex64::new(1., 0.); 2], None, 200., &mut out)
            .is_err()
    );
}

fn assert_results_close(a: &BranchFlowResult, b: &BranchFlowResult) {
    let aa = [
        a.p_from_mw,
        a.q_from_mvar,
        a.p_to_mw,
        a.q_to_mvar,
        a.pl_mw,
        a.ql_mvar,
        a.i_from_ka,
        a.i_to_ka,
        a.i_ka,
        a.vm_from_pu,
        a.vm_to_pu,
        a.va_from_degree,
        a.va_to_degree,
        a.loading_percent,
    ];
    let bb = [
        b.p_from_mw,
        b.q_from_mvar,
        b.p_to_mw,
        b.q_to_mvar,
        b.pl_mw,
        b.ql_mvar,
        b.i_from_ka,
        b.i_to_ka,
        b.i_ka,
        b.vm_from_pu,
        b.vm_to_pu,
        b.va_from_degree,
        b.va_to_degree,
        b.loading_percent,
    ];
    for (x, y) in aa.into_iter().zip(bb) {
        assert!(
            (x - y).abs() <= 1e-10 * (1.0 + x.abs().max(y.abs())),
            "{x} != {y}"
        );
    }
}
#[test]
fn soa_portable_and_selected_avx_kernel_match_for_phase_and_outage() {
    let mut world = base_world();
    add_line(&mut world, 100, 1.0, false);
    let dev = TransformerDevice {
        df: 1.,
        i0_percent: 0.,
        pfe_kw: 0.,
        vk_percent: 10.,
        vkr_percent: 0.5,
        shift_degree: 17.,
        sn_mva: 40.,
        vn_hv_kv: 110.,
        vn_lv_kv: 20.,
        max_loading_percent: None,
        parallel: 2,
        tap: None,
    };
    // Asymmetric off-diagonal complex values exercise ordering and phase terms.
    let shifted = Matrix2::new(
        Complex64::new(1., -3.),
        Complex64::new(-1.2, 2.8),
        Complex64::new(-0.7, 3.2),
        Complex64::new(1.1, -2.9),
    );
    world.spawn((
        TransformerID(901),
        dev,
        FromBusMatrixIndex(0),
        ToBusMatrixIndex(1),
        Port4MatPatch(shifted),
    ));
    let catalog = BranchCatalog::from_world(&mut world).unwrap();
    let coefficients = catalog.admittances();
    assert!(coefficients.iter().all(|v| v.len() == catalog.len()));
    assert_eq!(catalog.from_buses(), &[1, 1]);
    assert_eq!(catalog.to_buses(), &[0, 0]);
    let voltage = [Complex64::new(0.91, -0.08), Complex64::new(1.03, 0.04)];
    let (mut portable, mut selected) = (vec![], vec![]);
    for outage in [None, Some(0), Some(1)] {
        catalog
            .compute_with_kernel(&voltage, outage, PostprocessKernel::Portable, &mut portable)
            .unwrap();
        catalog
            .compute_with_kernel(&voltage, outage, catalog.kernel(), &mut selected)
            .unwrap();
        for (a, b) in portable.iter().zip(&selected) {
            assert_results_close(a, b);
        }
        if PostprocessKernel::Avx2Fma.available() {
            catalog
                .compute_with_kernel(&voltage, outage, PostprocessKernel::Avx2Fma, &mut selected)
                .unwrap();
            for (a, b) in portable.iter().zip(&selected) {
                assert_results_close(a, b);
            }
        }
        if let Some(i) = outage {
            all_zero(&selected[i]);
        }
    }
}
