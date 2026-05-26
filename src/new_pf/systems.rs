use bevy_ecs::prelude::*;
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use crate::basic::ecs::elements::{Line, Transformer, Admittance, Port2, VBase, PFCommonData, BusID, BusType, PPNetwork};

/// Specialized component for the new data path: stores the 2x2 primitive block of a branch.
#[derive(Component, Debug, Clone)]
pub struct PrimitiveY2x2(pub [Complex64; 4]); // [yff, yft, ytf, ytt]

/// Resource that holds the assembled network operators.
#[derive(Resource, Default)]
pub struct NetworkOperators {
    pub ybus: Option<CscMatrix<Complex64>>,
    pub yf: Option<CscMatrix<Complex64>>,
    pub yt: Option<CscMatrix<Complex64>>,
}

/// Resource that stores the bus ordering and index mappings.
#[derive(Resource)]
pub struct PFOrder {
    pub pq_nodes: Vec<usize>,
    pub pv_nodes: Vec<usize>,
    pub slack_nodes: Vec<usize>,
    /// original_bus_idx -> internal_ordered_idx
    pub map: Vec<usize>,
}

/// System to initialize the topology: builds the permuted A matrix.
pub fn initialize_pf_topology_system(
    mut commands: Commands,
    query_buses: Query<(Entity, &BusID, &BusType)>,
    query_branches: Query<(&Port2, &Line)>, // TODO: include transformers
    mut ops: ResMut<NetworkOperators>,
) {
    let nb = query_buses.iter().count();
    let mut pq = Vec::new();
    let mut pv = Vec::new();
    let mut slack = Vec::new();
    
    // 1. Determine Order [PQ | PV | slack]
    for (_, id, btype) in query_buses.iter() {
        let idx = id.0 as usize;
        match btype.0.as_str() {
            "PQ" => pq.push(idx),
            "PV" => pv.push(idx),
            "ref" | "slack" => slack.push(idx),
            _ => pq.push(idx),
        }
    }
    
    let mut map = vec![0usize; nb];
    let mut internal_idx = 0;
    for &orig in &pq { map[orig] = internal_idx; internal_idx += 1; }
    for &orig in &pv { map[orig] = internal_idx; internal_idx += 1; }
    for &orig in &slack { map[orig] = internal_idx; internal_idx += 1; }

    // 2. Build Permuted A Matrix (2nl x nb)
    let nl = query_branches.iter().count();
    let mut a_coo = CooMatrix::<Complex64>::new(2 * nl, nb);
    let one = Complex64::new(1.0, 0.0);
    
    for (l, (port, _)) in query_branches.iter().enumerate() {
        let f_orig = port.0[0] as usize;
        let t_orig = port.0[1] as usize;
        // Use the map to "bake in" the permutation
        a_coo.push(2 * l,     map[f_orig], one);
        a_coo.push(2 * l + 1, map[t_orig], one);
    }
    
    commands.insert_resource(BinaryIncidence { a_mat: CscMatrix::from(&a_coo) });
    commands.insert_resource(PFOrder { pq_nodes: pq, pv_nodes: pv, slack_nodes: slack, map });
}

/// Resource that stores the binary incidence matrix A (2b x n).
#[derive(Resource)]
pub struct BinaryIncidence {
    pub a_mat: CscMatrix<Complex64>,
}

use crate::basic::ecs::elements::trans::Port4MatPatch;

/// System to bridge existing physical calculations (Port4MatPatch) to new_pf path.
/// This ensures ZERO redundant calculations by reusing your existing transformer/line logic.
pub fn calculate_primitive_y_system(
    mut commands: Commands,
    query_patches: Query<(Entity, &Port4MatPatch), Without<PrimitiveY2x2>>,
) {
    for (entity, patch) in query_patches.iter() {
        let g = patch.0;
        // Matrix2 layout: [ (0,0), (1,0), (0,1), (1,1) ]
        // Our PrimitiveY2x2: [yff, yft, ytf, ytt]
        commands.entity(entity).insert(PrimitiveY2x2([
            g[(0, 0)], g[(0, 1)],
            g[(1, 0)], g[(1, 1)]
        ]));
    }
}

pub fn assemble_ybus_system(
    mut ops: ResMut<NetworkOperators>,
    incidence: Res<BinaryIncidence>,
    query_branches: Query<&PrimitiveY2x2>,
) {
    let nl = query_branches.iter().count();
    let mut y_prim_coo = CooMatrix::<Complex64>::new(2 * nl, 2 * nl);
    for (l, prim) in query_branches.iter().enumerate() {
        let p = prim.0;
        y_prim_coo.push(2 * l,     2 * l,     p[0]);
        y_prim_coo.push(2 * l,     2 * l + 1, p[1]);
        y_prim_coo.push(2 * l + 1, 2 * l,     p[2]);
        y_prim_coo.push(2 * l + 1, 2 * l + 1, p[3]);
    }
    let y_prim = CscMatrix::from(&y_prim_coo);
    let a_mat = &incidence.a_mat;
    
    // Ybus = A^T * (Y_prim * A)
    ops.ybus = Some(&a_mat.transpose() * &(&y_prim * a_mat));
}

pub fn assemble_yf_yt_system(
    mut ops: ResMut<NetworkOperators>,
    incidence: Res<BinaryIncidence>,
    query_branches: Query<&PrimitiveY2x2>,
) {
    let nl = query_branches.iter().count();
    let mut y_prim_coo = CooMatrix::<Complex64>::new(2 * nl, 2 * nl);
    for (l, prim) in query_branches.iter().enumerate() {
        let p = prim.0;
        y_prim_coo.push(2 * l,     2 * l,     p[0]);
        y_prim_coo.push(2 * l,     2 * l + 1, p[1]);
        y_prim_coo.push(2 * l + 1, 2 * l,     p[2]);
        y_prim_coo.push(2 * l + 1, 2 * l + 1, p[3]);
    }
    let y_prim = CscMatrix::from(&y_prim_coo);
    let a_mat = &incidence.a_mat;
    
    // M = Y_prim * A
    let m_mat = &y_prim * a_mat;
    
    // Slice M to get Yf and Yt
    // ... Extraction logic ...
}
