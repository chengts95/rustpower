use std::collections::HashSet;

use bevy_ecs::{prelude::*, system::RunSystemOnce};
use nalgebra::*;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num_complex::Complex64;
use num_traits::One;

use super::{elements::*, network::PowerFlowMat};

/// Creates the permutation matrix for reordering buses in the power flow network.
///
/// This function creates the permutation matrix for reordering buses in the power flow network based on PV nodes, PQ nodes, and external grid nodes.
///
/// # Arguments
///
/// * `pv` - A reference to a slice containing PV node indices.
/// * `pq` - A reference to a slice containing PQ node indices.
/// * `ext` - A reference to a slice containing external grid node indices.
/// * `nodes` - The total number of nodes in the power flow network.
///
/// # Returns
///
/// The permutation matrix for reordering buses in the power flow network as a COO (Coordinate) matrix.
fn create_premute_mat(pv: &[i64], pq: &[i64], ext: &[i64], nodes: usize) -> CooMatrix<i64> {
    let row_indices = DVector::from_fn(nodes, |i, _| i);
    let mut col_indices = DVector::from_fn(nodes, |i, _| i);
    let values = DVector::from_element(nodes, 1);

    let n_bus = pv.len() + pq.len();
    for i in 0..pv.len() {
        //let temp = col_indices[i];
        col_indices[i] = pv[i] as usize;
        //col_indices[pv[i] as usize] = temp;
    }
    for i in pv.len()..n_bus {
        //let temp = col_indices[i];
        col_indices[i] = pq[i - pv.len()] as usize;
        //col_indices[pv[i] as usize] = temp;
    }
    for i in n_bus..nodes {
        col_indices[i] = ext[i - n_bus] as usize;
    }
    let t = unsafe {
        CooMatrix::try_from_triplets(
            nodes,
            nodes,
            row_indices.data.into(),
            col_indices.data.into(),
            values.data.into(),
        )
        .unwrap_unchecked()
    };
    t
}

fn create_y_bus(
    common: Res<PFCommonData>,
    node_lookup: Res<NodeLookup>,
    y_br: Query<(&Admittance, &Port2, &VBase)>,
) -> (CsrMatrix<Complex64>, CsrMatrix<Complex64>) {
    let nodes = node_lookup.0.len();
    let branches = y_br.iter();
    let s_base = common.sbase;
    let mut diag_admit = CsrMatrix::identity(branches.len());
    let admit_br = diag_admit.values_mut();
    let mut incidence_matrix = CooMatrix::new(nodes, branches.len());
    for (idx, (ad, topo, vbase)) in branches.enumerate() {
        admit_br[idx] = ad.0 * (vbase.0 * vbase.0) / s_base;
        if topo.0[0] >= 0 {
            incidence_matrix.push(topo.0[0] as usize, idx as usize, Complex::one());
        }
        if topo.0[1] >= 0 {
            incidence_matrix.push(topo.0[1] as usize, idx as usize, -Complex::one());
        }
    }

    let incidence_matrix = CsrMatrix::from(&incidence_matrix);
    let ybus = &incidence_matrix * (diag_admit * incidence_matrix.transpose());
    (incidence_matrix, ybus)
}
pub fn init_states(world: &mut World) {
    let (_inci_mat, y_bus) = world.run_system_once(create_y_bus);
    let cfg = world.run_system_once(init_bus_status);
    let y_bus =  y_bus.transpose_as_csc();
    let s_bus =  cfg.s_bus;
    let v_bus_init = cfg.v_bus_init;
    world.insert_resource(PowerFlowMat {
        reorder: cfg.reorder,
        y_bus,
        s_bus,
        v_bus_init,
        npv: cfg.npv,
        npq: cfg.npq,
    });
}
struct SystemBusStatus {
    reorder: CsrMatrix<Complex64>,
    s_bus: DVector<Complex64>,
    v_bus_init: DVector<Complex64>,
    npv: usize,
    npq: usize,
}
fn init_bus_status(
    node_lookup: Res<NodeLookup>,
   // node_mapping: Option<Res<NodeMapping>>,
    common: Res<PFCommonData>,
    q: Query<&NodeType>,
) -> SystemBusStatus {
    let nodes = node_lookup.0.len();
    let mut pq_set = HashSet::new();
    let mut pv_set = HashSet::new();
    let mut ext_set = HashSet::new();
    let mut sbus: DVector<Complex64> = DVector::zeros(nodes);
    let mut vbus: DVector<Complex64> = DVector::from_element(nodes, Complex64::one());
    let s_base = common.sbase;
    q.iter().for_each(|node| match node {
        NodeType::PQ(pq) => {
            sbus[pq.bus as usize] -= pq.s;
            pq_set.insert(pq.bus);
        }
        NodeType::PV(pv) => {
            sbus[pv.bus as usize] += pv.p;
            vbus[pv.bus as usize] = Complex64::new(pv.v, 0.0);
            pv_set.insert(pv.bus);
        }
        NodeType::EXT(ext) => {
            vbus[ext.bus as usize] = Complex64::from_polar(ext.v, ext.phase);
            ext_set.insert(ext.bus);
        }
        NodeType::AUX(_aux_node) => {}
    });
    let pv_ext: HashSet<_> = pv_set.union(&ext_set).collect();
    let mut pv_only: Vec<_> = pv_set.difference(&ext_set).map(|x| *x).collect();
    let mut pq_only: Vec<_> = node_lookup
        .0
        .keys()
        .collect::<HashSet<_>>()
        .difference(&pv_ext)
        .map(|x| **x)
        .collect();
    let npv = pv_only.len();
    let npq = pq_only.len();
    let mut exts: Vec<_> = ext_set.into_iter().collect();
    // if let Some(mapping) = node_mapping {

    //     //let pq_map: Vec<_> = pq_set.iter().map(|x| mapping.0[*x as usize]).collect();
    // }
    pv_only.sort();
    pq_only.sort();
    exts.sort();

    let reorder = create_premute_mat(
        pv_only.as_slice(),
        pq_only.as_slice(),
        exts.as_slice(),
        nodes,
    );
    let from = CsrMatrix::from(&reorder);
    let reorder: CsrMatrix<Complex64> = CsrMatrix::try_from_pattern_and_values(
        from.pattern().clone(),
        Vec::from_iter(from.values().iter().map(|x| Complex64::new(*x as f64, 0.0))),
    )
    .unwrap();
    sbus.scale_mut(1.0/s_base);
    SystemBusStatus {
        reorder: reorder,
        s_bus: sbus,
        v_bus_init: vbus,
        npv: npv,
        npq: npq,
    }
}
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::{
//         basic::new_ecs::network::{DataOps, PowerFlow, PowerGrid},
//         io::pandapower::{ecs_net_conv::*, Network},
//         prelude::test_ieee39,
//     };

// }
