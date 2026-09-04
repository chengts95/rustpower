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
    buses: Query<&VNominal>,
    y_br: Query<(&Admittance, &Port2, &VBase)>,
    lines_and_switches: Query<
        (&Port4MatPatch, &FromBus, &ToBus),
        (Without<OutOfService>, Without<TransformerDevice>),
    >,
    trafos: Query<(&Port4MatPatch, &TransformerDevice, &FromBus, &ToBus), Without<OutOfService>>,
) -> (CsrMatrix<Complex64>, CscMatrix<Complex64>) {
    let nodes = node_lookup.len();
    let s_base = common.sbase;

    // Cache bus nominal voltage array indexed by bus id
    let mut bus_vn = vec![1.0; nodes];
    for (bus_idx, entity) in node_lookup.iter() {
        if let Ok(vnom) = buses.get(entity) {
            bus_vn[bus_idx as usize] = vnom.0.0;
        }
    }

    let mut coo = CooMatrix::new(nodes, nodes);

    // 1. Stamp Lines and Switches (V_base from connecting bus)
    for (patch, from, to) in lines_and_switches.iter() {
        let f = from.0;
        let t = to.0;
        let vn = if f >= 0 && (f as usize) < nodes {
            bus_vn[f as usize]
        } else if t >= 0 && (t as usize) < nodes {
            bus_vn[t as usize]
        } else {
            1.0
        };
        let p = patch.0.scale((vn * vn) / s_base);
        if f >= 0 {
            coo.push(f as usize, f as usize, p[(0, 0)]);
        }
        if t >= 0 {
            coo.push(t as usize, t as usize, p[(1, 1)]);
        }
        if f >= 0 && t >= 0 {
            coo.push(f as usize, t as usize, p[(0, 1)]);
            coo.push(t as usize, f as usize, p[(1, 0)]);
        }
    }

    // 2. Stamp Transformers (V_base from transformer device vn_lv_kv)
    for (patch, dev, from, to) in trafos.iter() {
        let vn = dev.vn_lv_kv;
        let p = patch.0.scale((vn * vn) / s_base);
        let f = from.0;
        let t = to.0;
        if f >= 0 {
            coo.push(f as usize, f as usize, p[(0, 0)]);
        }
        if t >= 0 {
            coo.push(t as usize, t as usize, p[(1, 1)]);
        }
        if f >= 0 && t >= 0 {
            coo.push(f as usize, t as usize, p[(0, 1)]);
            coo.push(t as usize, f as usize, p[(1, 0)]);
        }
    }

    // 3. Add ground shunts (EShunt) directly to diagonal
    for (ad, topo, vbase) in y_br.iter() {
        let y_pu = ad.0 * (vbase.0 * vbase.0) / s_base;
        if topo.0[0] >= 0 && topo.0[1] < 0 {
            coo.push(topo.0[0] as usize, topo.0[0] as usize, y_pu);
        } else if topo.0[1] >= 0 && topo.0[0] < 0 {
            coo.push(topo.0[1] as usize, topo.0[1] as usize, y_pu);
        } else if topo.0[0] >= 0 && topo.0[1] >= 0 {
            let idx0 = topo.0[0] as usize;
            let idx1 = topo.0[1] as usize;
            coo.push(idx0, idx0, y_pu);
            coo.push(idx1, idx1, y_pu);
            coo.push(idx0, idx1, -y_pu);
            coo.push(idx1, idx0, -y_pu);
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
    let (_incidence_matrix, y_bus) = world.run_system_once(create_y_bus).unwrap();
    world.insert_resource(OriginalYBus(y_bus.clone()));
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
    pq: Query<(&BusID, &PQBus)>,
    pv: Query<(&BusID, &PVBus), Without<SlackBus>>,
    ext: Query<(&BusID, &SlackBus)>,
    sbus: Query<(&BusID, &SBusInjPu)>,
    vbus: Query<(&BusID, &VBusPu)>,
) -> SystemBusStatus {
    let nodes = node_lookup.len();
    // Initialize power injections and voltage vectors
    let mut s_bus = DVector::zeros(nodes);
    let mut v_bus_init = DVector::from_element(nodes, Complex64::one());
    let mut pq_only: Vec<_> = pq.iter().map(|x| x.0.0).collect();
    let mut pv_only: Vec<_> = pv.iter().map(|x| x.0.0).collect();
    let mut exts: Vec<_> = ext.iter().map(|x| x.0.0).collect();

    sbus.iter().for_each(|(bus_id, s)| {
        let idx = bus_id.0 as usize;
        s_bus[idx] = s.0;
    });
    vbus.iter().for_each(|(bus_id, s)| {
        let idx = bus_id.0 as usize;
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
