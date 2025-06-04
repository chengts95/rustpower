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
    pub reorder: CsrMatrix<Complex<f64>>, // Reordering matrix
    pub y_bus: CscMatrix<Complex<f64>>,   // Y-bus admittance matrix
    pub s_bus: DVector<Complex64>,        // S-bus power injections
    pub v_bus_init: DVector<Complex64>,   // V-bus power injections
    pub npv: usize,                       // Number of PV buses
    pub npq: usize,                       // Number of PQ buses
    pub to_perm: Vec<usize>,              // original → reordered
    pub from_perm: Vec<usize>,            // reordered → original
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
///

pub(crate) fn create_permutation_matrix(
    pv: &[i64],
    pq: &[i64],
    ext: &[i64],
    nodes: usize,
) -> CooMatrix<i64> {
    let row_indices: Vec<usize> = (0..nodes).collect();
    let mut col_indices: Vec<usize> = (0..nodes).collect();
    let values = vec![1; nodes];

    let n_bus = pv.len() + pq.len();
    for i in 0..pv.len() {
        col_indices[i] = pv[i] as usize;
    }
    for i in pv.len()..n_bus {
        col_indices[i] = pq[i - pv.len()] as usize;
    }
    for i in n_bus..nodes {
        col_indices[i] = ext[i - n_bus] as usize;
    }

    CooMatrix::try_from_triplets(nodes, nodes, row_indices, col_indices, values)
        .expect("Failed to create permutation matrix")
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
    y_br: Query<(&Admittance, &Port2, &VBase)>,
) -> (CsrMatrix<Complex64>, CsrMatrix<Complex64>) {
    let nodes = node_lookup.len();
    let branches = y_br.iter();
    let s_base = common.sbase;

    // Initialize diagonal admittance matrix for branches
    let mut diag_admit = CsrMatrix::identity(branches.len());
    let admit_br = diag_admit.values_mut();

    // Initialize incidence matrix in COO format
    let mut incidence_matrix = CooMatrix::new(nodes, branches.len());

    for (idx, (ad, topo, vbase)) in branches.enumerate() {
        // Compute branch admittance in per-unit system
        admit_br[idx] = ad.0 * (vbase.0 * vbase.0) / s_base;

        // Build incidence matrix
        if topo.0[0] >= 0 {
            incidence_matrix.push(topo.0[0] as usize, idx, Complex64::one());
        }
        if topo.0[1] >= 0 {
            incidence_matrix.push(topo.0[1] as usize, idx, -Complex64::one());
        }
    }

    // Convert incidence matrix to CSR format
    let incidence_matrix = CsrMatrix::from(&incidence_matrix);

    // Compute Y-bus matrix: Y = A * diag(admittance) * A^T
    let y_bus = &incidence_matrix * (diag_admit * incidence_matrix.transpose());

    (incidence_matrix, y_bus)
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
pub fn init_states(world: &mut World) {
    let (_incidence_matrix, y_bus) = world.run_system_once(create_y_bus).unwrap();
    let cfg = world.run_system_once(init_bus_status).unwrap();
    let y_bus = y_bus.transpose_as_csc();
    let s_bus = cfg.s_bus;
    let v_bus_init = cfg.v_bus_init;
    let mut to_perm = vec![0; v_bus_init.len()]; // 原 → 新
    let mut from_perm = vec![0; v_bus_init.len()]; // 新 → 原
    // println!(
    //     "Power flow system initialized with {} buses, {} PV buses, and {} PQ buses.",
    //     v_bus_init.len(),
    //     cfg.npv,
    //     cfg.npq
    // );
    for (new_idx, &original_idx) in cfg.reorder.col_indices().iter().enumerate() {
        to_perm[original_idx] = new_idx;
        from_perm[new_idx] = original_idx;
    }
    world.insert_resource(PowerFlowMat {
        reorder: cfg.reorder,
        y_bus,
        s_bus,
        v_bus_init,
        npv: cfg.npv,
        npq: cfg.npq,
        to_perm,
        from_perm,
    });
}

/// Holds the system bus status, including reorder matrix, power injections, initial voltages, and counts of PV and PQ buses.
pub(crate) struct SystemBusStatus {
    /// The permutation matrix for reordering buses.
    reorder: CsrMatrix<Complex64>,
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

    // Create permutation matrix for bus reordering
    let reorder_coo = create_permutation_matrix(
        pv_only.as_slice(),
        pq_only.as_slice(),
        exts.as_slice(),
        nodes,
    );
    let reorder_csr = CsrMatrix::from(&reorder_coo);
    let reorder: CsrMatrix<Complex64> = CsrMatrix::try_from_pattern_and_values(
        reorder_csr.pattern().clone(),
        reorder_csr
            .values()
            .iter()
            .map(|&x| Complex64::new(x as f64, 0.0))
            .collect(),
    )
    .expect("Failed to create complex permutation matrix");

    SystemBusStatus {
        reorder,
        s_bus,
        v_bus_init,
        npv,
        npq,
    }
}
