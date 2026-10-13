use bevy_ecs::{prelude::*, system::RunSystemOnce};
use nalgebra::*;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use num_complex::Complex64;
use num_traits::One;

use crate::basic::ecs::elements::*;

use super::init::*;
// /// Resource that wraps the power flow network (PFNetwork).
// #[derive(Debug, Resource, Clone, serde::Serialize, serde::Deserialize)]
// pub struct ResPFNetwork(pub PFNetwork);

/// Resource that holds the power flow configuration options, such as the initial voltage guess,
/// maximum iterations, and tolerance for convergence.
#[derive(Debug, Default, Resource, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerFlowConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_it: Option<usize>, // Maximum number of iterations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tol: Option<f64>, // Tolerance for convergence
}

/// Resource for storing the results of power flow calculation, including the final voltage vector,
/// number of iterations taken, and whether the solution converged.
#[derive(Debug, Default, Resource, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerFlowResult {
    pub v: DVector<Complex64>, // Final voltage vector after convergence
    pub iterations: usize,     // Number of iterations taken
    pub converged: bool,       // Convergence status
}

/// Resource holding various matrices required for power flow calculations, including the reordered
/// matrix, admittance matrix (Y-bus), and the power injection vector (S-bus).
#[derive(Debug, Resource, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerFlowMat {
    pub y_bus: CscMatrix<Complex<f64>>, // Y-bus admittance matrix
    pub s_bus: DVector<Complex64>,      // S-bus power injections
    pub v_bus_init: DVector<Complex64>, // V-bus power injections
    pub npv: usize,                     // Number of PV buses
    pub npq: usize,                     // Number of PQ buses
    pub to_perm: Vec<usize>,            // original → reordered
    pub from_perm: Vec<usize>,          // reordered → original
}
impl PowerFlowMat {
    pub fn reorder_index(&self, orig: usize) -> usize {
        self.to_perm[orig]
    }

    pub fn inverse_index(&self, perm: usize) -> usize {
        self.from_perm[perm]
    }
}
/// Creates a permutation matrix for reordering buses in the power flow network.
///
/// This function constructs a permutation matrix based on the indices of PV nodes, PQ nodes, and external grid nodes.
/// The resulting permutation matrix can be used to reorder buses in the network for computational efficiency.
///
/// # Arguments
///
/// * `pv` - A slice containing the indices of PV nodes.
/// * `pq` - A slice containing the indices of PQ nodes.
/// * `ext` - A slice containing the indices of external grid nodes.
/// * `nodes` - The total number of nodes in the power flow network.
///
/// # Returns
///
/// A permutation matrix for reordering buses in the power flow network as a COO (Coordinate) matrix.
///
/// # Panics
///
/// This function will panic if the indices provided in `pv`, `pq`, or `ext` are out of bounds.
/// Creates a permutation matrix based on PV, PQ, and EXT nodes.
#[allow(dead_code)]
pub(crate) fn create_permutation_matrix(
    pq: &[i64],
    pv: &[i64],
    ext: &[i64],
    nodes: usize,
) -> CsrMatrix<Complex64> {
    let mut p = vec![0; nodes];
    let n_bus = pq.len() + pv.len();
    for i in 0..pq.len() {
        p[i] = pq[i] as usize;
    }
    for i in pq.len()..n_bus {
        p[i] = pv[i - pq.len()] as usize;
    }
    for i in n_bus..nodes {
        p[i] = ext[i - n_bus] as usize;
    }

    crate::basic::sparse::utils::csr_permutation(nodes, &p)
}

/// Creates the Y-bus matrix for the power flow network.
///
/// This function constructs the admittance (Y-bus) matrix and the incidence matrix for the power flow network
/// based on the provided branch admittances, network topology, and voltage bases.
///
/// # Arguments
///
/// * `common` - A resource containing common power flow data (e.g., base power).
/// * `node_lookup` - A resource containing the node lookup table.
/// * `y_br` - A query providing access to branch admittances, topology, and voltage bases.
///
/// # Returns
///
/// A tuple containing:
/// - The incidence matrix as a CSR (Compressed Sparse Row) matrix.
/// - The Y-bus matrix as a CSR matrix.
pub(crate) fn create_y_bus(
    common: Res<PFCommonData>,
    node_lookup: Res<NodeLookup>,
    buses: Query<(&BusMatrixIndex, &VNominal)>,
    y_br: Query<(&Admittance, &Port2MatrixIndex, &VBase)>,
    lines_and_switches: Query<
        (&Port4MatPatch, &FromBusMatrixIndex, &ToBusMatrixIndex),
        (Without<OutOfService>, Without<TransformerDevice>),
    >,
    trafos: Query<
        (
            &Port4MatPatch,
            &TransformerDevice,
            &FromBusMatrixIndex,
            &ToBusMatrixIndex,
        ),
        Without<OutOfService>,
    >,
) -> (CsrMatrix<Complex64>, CscMatrix<Complex64>) {
    let nodes = node_lookup.len();
    let inv_s_base = 1.0 / common.sbase;

    // Cache bus z_base = (Vn^2) / S_base array using contiguous slice iteration
    let mut bus_z_base = vec![inv_s_base; nodes];
    if let Ok(chunks) = buses.contiguous_iter() {
        for (id_slice, vnom_slice) in chunks {
            let len = id_slice.len();
            for i in 0..len {
                let idx = id_slice[i].0;
                if idx < nodes {
                    let vn = vnom_slice[i].0.0;
                    bus_z_base[idx] = (vn * vn) * inv_s_base;
                }
            }
        }
    } else {
        for (id, vnom) in buses.iter() {
            let idx = id.0;
            if idx < nodes {
                let vn = vnom.0.0;
                bus_z_base[idx] = (vn * vn) * inv_s_base;
            }
        }
    }

    let mut coo = CooMatrix::new(nodes, nodes);

    // 1. Stamp Lines and Switches using contiguous slice iteration
    if let Ok(chunks) = lines_and_switches.contiguous_iter() {
        for (patch_slice, from_slice, to_slice) in chunks {
            let len = patch_slice.len();
            for i in 0..len {
                let f = from_slice[i].0;
                let t = to_slice[i].0;
                let z_base = bus_z_base[f];
                let p = patch_slice[i].0.scale(z_base);
                coo.push(f, f, p[(0, 0)]);
                coo.push(t, t, p[(1, 1)]);
                coo.push(f, t, p[(0, 1)]);
                coo.push(t, f, p[(1, 0)]);
            }
        }
    } else {
        for (patch, from, to) in lines_and_switches.iter() {
            let f = from.0;
            let t = to.0;
            let z_base = bus_z_base[f];
            let p = patch.0.scale(z_base);
            coo.push(f, f, p[(0, 0)]);
            coo.push(t, t, p[(1, 1)]);
            coo.push(f, t, p[(0, 1)]);
            coo.push(t, f, p[(1, 0)]);
        }
    }

    // 2. Stamp Transformers using contiguous slice iteration
    if let Ok(chunks) = trafos.contiguous_iter() {
        for (patch_slice, dev_slice, from_slice, to_slice) in chunks {
            let len = patch_slice.len();
            for i in 0..len {
                let vn = dev_slice[i].vn_lv_kv;
                let z_base = (vn * vn) * inv_s_base;
                let p = patch_slice[i].0.scale(z_base);
                let f = from_slice[i].0;
                let t = to_slice[i].0;
                coo.push(f, f, p[(0, 0)]);
                coo.push(t, t, p[(1, 1)]);
                coo.push(f, t, p[(0, 1)]);
                coo.push(t, f, p[(1, 0)]);
            }
        }
    } else {
        for (patch, dev, from, to) in trafos.iter() {
            let vn = dev.vn_lv_kv;
            let z_base = (vn * vn) * inv_s_base;
            let p = patch.0.scale(z_base);
            let f = from.0;
            let t = to.0;
            coo.push(f, f, p[(0, 0)]);
            coo.push(t, t, p[(1, 1)]);
            coo.push(f, t, p[(0, 1)]);
            coo.push(t, f, p[(1, 0)]);
        }
    }

    // Ground is explicit; the resolved port caches contain native matrix offsets.
    for (ad, port, base) in y_br.iter() {
        let y = ad.0 * ((base.0 * base.0) * inv_s_base);
        let [from, to] = port.0;
        if let Some(f) = from {
            coo.push(f, f, y);
        }
        if let Some(t) = to {
            coo.push(t, t, y);
        }
        if let (Some(f), Some(t)) = (from, to) {
            coo.push(f, t, -y);
            coo.push(t, f, -y);
        }
    }

    let y_csc = CscMatrix::from(&coo);
    (CsrMatrix::zeros(0, 0), y_csc)
}

/// Initializes the power flow calculation states and inserts necessary resources into the world.
///
/// This function should be called once at the beginning to set up the initial system state for power flow calculations.
///
/// # Arguments
///
/// * `world` - A mutable reference to the ECS world.
///
/// # Side Effects
///
/// Inserts a `PowerFlowMat` resource into the world, containing matrices and vectors required for power flow analysis.
/// Resource holding the original unpermuted Ybus admittance matrix.
#[derive(Debug, Resource, Clone)]
pub struct OriginalYBus(pub CscMatrix<Complex64>);

pub fn init_states(world: &mut World) {
    crate::basic::ecs::elements::topology::prepare_topology(world).expect("invalid bus topology");
    // Case archives omit result buffers; recreate them when preparing the model.
    let lines: Vec<_> = world
        .query_filtered::<Entity, (
            With<LineParams>,
            Without<crate::basic::ecs::post_processing::LineResultData>,
        )>()
        .iter(world)
        .collect();
    for e in lines {
        world
            .entity_mut(e)
            .insert(crate::basic::ecs::post_processing::LineResultData::default());
    }
    let trafos: Vec<_> = world
        .query_filtered::<Entity, (
            With<TransformerDevice>,
            Without<crate::basic::ecs::post_processing::TrafoResultData>,
        )>()
        .iter(world)
        .collect();
    for e in trafos {
        world
            .entity_mut(e)
            .insert(crate::basic::ecs::post_processing::TrafoResultData::default());
    }
    crate::basic::ecs::elements::switch::prepare_switches(world);
    let (_incidence_matrix, y_bus) = world.run_system_once(create_y_bus).unwrap();
    world.insert_resource(OriginalYBus(y_bus.clone()));
    insert_bus_states(world, y_bus);
}

/// Bus-type changes preserve the physical admittance; only its solver ordering changes.
pub fn repartition_states(world: &mut World) {
    if let Some(original) = world.get_resource::<OriginalYBus>() {
        let y_bus = original.0.clone();
        insert_bus_states(world, y_bus);
    } else {
        init_states(world);
    }
}

fn insert_bus_states(world: &mut World, y_bus: CscMatrix<Complex64>) {
    let cfg = world.run_system_once(init_bus_status).unwrap();
    let s_bus = cfg.s_bus;
    let v_bus_init = cfg.v_bus_init;
    world.insert_resource(PowerFlowMat {
        y_bus,
        s_bus,
        v_bus_init,
        npv: cfg.npv,
        npq: cfg.npq,
        to_perm: cfg.to_perm,
        from_perm: cfg.from_perm,
    });
    crate::basic::ecs::elements::switch::reduce_switch_nodes(world);
}

/// Holds the system bus status, including permutation indices, power injections, initial voltages, and counts of PV and PQ buses.
pub(crate) struct SystemBusStatus {
    to_perm: Vec<usize>,
    from_perm: Vec<usize>,
    /// The complex power injections at each bus.
    s_bus: DVector<Complex64>,
    /// The initial voltage vector for each bus.
    v_bus_init: DVector<Complex64>,
    /// The number of PV buses.
    npv: usize,
    /// The number of PQ buses.
    npq: usize,
}

/// Initializes the bus status, including bus types and initial conditions.
///
/// This function collects bus information from the ECS world and prepares the necessary data structures for power flow analysis.
///
/// # Arguments
///
/// * `node_lookup` - A resource containing the node lookup table.
/// * `common` - A resource containing common power flow data (e.g., base power).
/// * `q` - A query providing access to node types.
///
/// # Returns
///
/// A `SystemBusStatus` struct containing the initialized bus statuses.
pub(crate) fn init_bus_status(
    node_lookup: Res<NodeLookup>,
    pq: Query<(&BusMatrixIndex, &PQBus)>,
    pv: Query<(&BusMatrixIndex, &PVBus), Without<SlackBus>>,
    ext: Query<(&BusMatrixIndex, &SlackBus)>,
    sbus: Query<(&BusMatrixIndex, &SBusInjPu, Option<&super::qlim::QLimited>)>,
    vbus: Query<(&BusMatrixIndex, &VBusPu)>,
) -> SystemBusStatus {
    let nodes = node_lookup.len();
    // Initialize power injections and voltage vectors
    let mut s_bus = DVector::zeros(nodes);
    let mut v_bus_init = DVector::from_element(nodes, Complex64::one());
    let mut pq_only: Vec<_> = pq.iter().map(|x| x.0.0).collect();
    let mut pv_only: Vec<_> = pv.iter().map(|x| x.0.0).collect();
    let mut exts: Vec<_> = ext.iter().map(|x| x.0.0).collect();

    sbus.iter().for_each(|(bus_id, s, bound)| {
        let idx = bus_id.0;
        s_bus[idx] = s.0 + Complex64::new(0.0, bound.map_or(0.0, |q| q.q_pu));
    });
    vbus.iter().for_each(|(bus_id, s)| {
        let idx = bus_id.0;
        v_bus_init[idx] = s.0;
    });

    let npv = pv_only.len();
    let npq = pq_only.len();

    // Sort the bus indices for consistent ordering
    pv_only.sort_unstable();
    pq_only.sort_unstable();
    exts.sort_unstable();

    let mut to_perm = vec![0; nodes];
    let mut from_perm = Vec::with_capacity(nodes);
    from_perm.extend(pq_only.iter().map(|&x| x as usize));
    from_perm.extend(pv_only.iter().map(|&x| x as usize));
    from_perm.extend(exts.iter().map(|&x| x as usize));

    for (new_idx, &original_idx) in from_perm.iter().enumerate() {
        to_perm[original_idx] = new_idx;
    }

    SystemBusStatus {
        to_perm,
        from_perm,
        s_bus,
        v_bus_init,
        npv,
        npq,
    }
}
