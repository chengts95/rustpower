use bevy_ecs::prelude::*;
use nalgebra::{Complex, Matrix2};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_complex::Complex64;
use crate::basic::ecs::elements::{Line, LineParams, PFCommonData, BusID, BusType, Port4MatPatch, NodeLookup, VNominal, FromBus, ToBus, TransformerDevice, ShuntDevice, TargetBus, OutOfService};

/// Specialized component for the new data path: stores the 2x2 primitive block of a branch.
#[derive(Component, Debug, Clone)]
pub struct PrimitiveY2x2(pub [Complex64; 4]); // [yff, yft, ytf, ytt]

/// Voltage base [kV] used to convert a branch's `Port4MatPatch` from physical Siemens
/// to system per-unit (PU = patch · vbase² / sbase).
/// For lines: vbase = from-bus VNominal. For transformers: vbase = TransformerDevice.vn_lv_kv.
#[derive(Component, Debug, Clone)]
pub struct BranchVBase(pub f64);

/// Ordered list of branch entities — index `l` is the canonical "branch slot" used by
/// the incidence matrix A (rows `2l`, `2l+1`) and by the block-diagonal Y_prim.
/// Required to keep topology and admittance iteration in lock-step.
#[derive(Resource, Default)]
pub struct BranchOrder(pub Vec<Entity>);

/// Per-shunt-entity per-unit admittance contribution to the Ybus diagonal at its bus.
///
/// Formula (b0907e7-corrected): for `ShuntDevice { p_mw, q_mvar, vn_kv, step }` on
/// a bus with `VNominal = bus_vn_kv`:
///   Ypu = (p_mw − j·q_mvar) · step · (bus_vn_kv / sh.vn_kv)² / sbase
/// The `(bus_vn_kv / sh.vn_kv)²` scale accounts for shunts rated at a voltage
/// that differs from the bus's nominal voltage. Q sign is negated because
/// pandapower defines q_mvar as consumed reactive power (positive = inductive),
/// and the admittance imaginary part is the negative of the consumed reactive.
#[derive(Component, Clone, Debug)]
pub struct ShuntYpu(pub Complex64);

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

/// System to initialize the topology: builds the permuted A matrix from line + trafo
/// **parent** entities (queried via FromBus/ToBus + marker, not Port2 which lives on
/// line children only). Also records the canonical branch order into `BranchOrder`.
pub fn initialize_pf_topology_system(
    mut commands: Commands,
    all_buses: Query<(Entity, &BusID)>,
    typed_buses: Query<(&BusID, &BusType)>,
    lines: Query<(Entity, &FromBus, &ToBus), With<Line>>,
    trafos: Query<(Entity, &FromBus, &ToBus), With<TransformerDevice>>,
) {
    let nb = all_buses.iter().count();
    let mut pq = Vec::new();
    let mut pv = Vec::new();
    let mut slack = Vec::new();

    // 1. Determine Order [PQ | PV | slack]. BusType is optional — buses without it
    // default to PQ. (The PFOrder permutation is only meaningful for the Newton
    // iteration; Ybus assembly itself doesn't depend on the bus split.)
    use std::collections::HashMap;
    let typed: HashMap<i64, &str> = typed_buses
        .iter()
        .map(|(id, bt)| (id.0, bt.0.as_str()))
        .collect();
    for (_, id) in all_buses.iter() {
        let idx = id.0 as usize;
        match typed.get(&id.0).copied().unwrap_or("PQ") {
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

    // 2. Build Permuted A Matrix (2nl × nb) — lines first, then trafos.
    let nl = lines.iter().count() + trafos.iter().count();
    let mut order: Vec<Entity> = Vec::with_capacity(nl);
    let mut a_coo = CooMatrix::<Complex64>::new(2 * nl, nb);
    let one = Complex64::new(1.0, 0.0);

    let mut l = 0usize;
    for (ent, from, to) in &lines {
        order.push(ent);
        a_coo.push(2 * l,     map[from.0 as usize], one);
        a_coo.push(2 * l + 1, map[to.0 as usize],   one);
        l += 1;
    }
    for (ent, from, to) in &trafos {
        order.push(ent);
        a_coo.push(2 * l,     map[from.0 as usize], one);
        a_coo.push(2 * l + 1, map[to.0 as usize],   one);
        l += 1;
    }

    commands.insert_resource(BinaryIncidence { a_mat: CscMatrix::from(&a_coo) });
    commands.insert_resource(BranchOrder(order));
    commands.insert_resource(PFOrder { pq_nodes: pq, pv_nodes: pv, slack_nodes: slack, map });
}

/// Resource that stores the binary incidence matrix A (2b x n).
#[derive(Resource)]
pub struct BinaryIncidence {
    pub a_mat: CscMatrix<Complex64>,
}

/// Convert per-branch `Port4MatPatch` (physical Siemens) → `PrimitiveY2x2` (system PU).
/// Scaling factor = `vbase² / sbase` from `BranchVBase`. Same per-unit convention as
/// `basic::ecs::powerflow::systems::create_y_bus`.
pub fn calculate_primitive_y_system(
    mut commands: Commands,
    query_patches: Query<(Entity, &Port4MatPatch, &BranchVBase), Without<PrimitiveY2x2>>,
    common: Res<PFCommonData>,
) {
    let s_base = common.sbase;
    for (entity, patch, vb) in query_patches.iter() {
        let scale = vb.0 * vb.0 / s_base;
        let g = patch.0;
        // Matrix2 layout (row-major): [(0,0), (0,1), (1,0), (1,1)] → [yff, yft, ytf, ytt]
        commands.entity(entity).insert(PrimitiveY2x2([
            g[(0, 0)] * scale,
            g[(0, 1)] * scale,
            g[(1, 0)] * scale,
            g[(1, 1)] * scale,
        ]));
    }
}

pub fn assemble_ybus_system(
    mut ops: ResMut<NetworkOperators>,
    incidence: Res<BinaryIncidence>,
    order: Res<BranchOrder>,
    pf_order: Res<PFOrder>,
    query_branches: Query<&PrimitiveY2x2>,
    shunts: Query<(&ShuntYpu, &TargetBus)>,
) {
    let nl = order.0.len();
    let mut y_prim_coo = CooMatrix::<Complex64>::new(2 * nl, 2 * nl);
    for (l, &ent) in order.0.iter().enumerate() {
        let prim = query_branches.get(ent).expect("branch entity missing PrimitiveY2x2");
        let p = prim.0;
        y_prim_coo.push(2 * l,     2 * l,     p[0]);
        y_prim_coo.push(2 * l,     2 * l + 1, p[1]);
        y_prim_coo.push(2 * l + 1, 2 * l,     p[2]);
        y_prim_coo.push(2 * l + 1, 2 * l + 1, p[3]);
    }
    let y_prim = CscMatrix::from(&y_prim_coo);
    let a_mat = &incidence.a_mat;

    // Ybus = A^T * (Y_prim * A) — A has the PFOrder permutation baked in, so Ybus is
    // already in [PQ | PV | slack] order.
    let mut ybus = &a_mat.transpose() * &(&y_prim * a_mat);

    // Add shunts to diagonal at the **permuted** bus index. Reads from shunt entities
    // only (ShuntYpu component); bus entities are untouched.
    for (sh_ypu, target) in shunts.iter() {
        let permuted = pf_order.map[target.0 as usize];
        add_to_csc_diagonal(&mut ybus, permuted, sh_ypu.0);
    }

    ops.ybus = Some(ybus);
}

/// Add a complex value to a CSC matrix's (idx, idx) diagonal entry in place.
/// Panics if the diagonal slot isn't already structurally present.
fn add_to_csc_diagonal(mat: &mut CscMatrix<Complex64>, idx: usize, val: Complex64) {
    let start = mat.col_offsets()[idx];
    let end = mat.col_offsets()[idx + 1];
    for i in start..end {
        if mat.row_indices()[i] == idx {
            mat.values_mut()[i] += val;
            return;
        }
    }
    panic!("Ybus missing diagonal at index {idx}; cannot add shunt");
}

pub fn assemble_yf_yt_system(
    _ops: ResMut<NetworkOperators>,
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
    let _m_mat = &y_prim * a_mat;

    // Slice M to get Yf and Yt
    // ... Extraction logic ...
}

/// Compute the consolidated 2×2 admittance for each line and attach as `Port4MatPatch`.
///
/// Units: **physical Siemens**, matching `setup_transformer` (`trans_systems`).
/// Downstream consumers (`create_y_bus`, `calculate_primitive_y_system`) scale to
/// system per-unit via `vbase² / sbase`.
///
/// Per-line equation (parallel = N parallel lines; line.rs params are per single line):
///   y_series = N / ((r + jx)·length)                     [series admittance]
///   y_shunt  = ½·(g + jb)·length·N at each end           [shunt half on each side]
///   where b = wbase·c_nf_per_km·1e-9, g = g_us_per_km·1e-6
///
/// The full 2×2 (physical Siemens):
///   [[y_series + y_shunt,        -y_series              ],
///    [-y_series,                  y_series + y_shunt    ]]
pub fn setup_line_primitive_y(
    mut commands: Commands,
    lines: Query<(Entity, &LineParams), (With<Line>, Without<Port4MatPatch>)>,
    common: Res<PFCommonData>,
) {
    let wbase = common.wbase;
    for (entity, params) in &lines {
        let length = params.length_km;
        let parallel = params.parallel as f64;

        // Series admittance (physical Siemens) — N parallel paths → multiply by N.
        let rl = params.r_ohm_per_km * length;
        let xl = params.x_ohm_per_km * length;
        let y_series = Complex::new(parallel, 0.0) / Complex::new(rl, xl);

        // Shunt admittance per end (physical Siemens) — total shunt × ½ per end.
        let b = wbase * 1e-9 * params.c_nf_per_km * length * parallel;
        let g = 1e-6 * params.g_us_per_km * length * parallel;
        let y_shunt = 0.5 * Complex::new(g, b);

        let yff = y_series + y_shunt;
        let yft = -y_series;
        let mat = Matrix2::new(yff, yft, yft, yff);
        commands.entity(entity).insert(Port4MatPatch(mat));
    }
}

/// Attach `BranchVBase` to every line and transformer parent entity.
///   - Line: vbase = VNominal of the from-bus.
///   - Transformer: vbase = TransformerDevice.vn_lv_kv (the LV side rated voltage,
///     matching the per-unit convention of `setup_transformer`).
pub fn setup_branch_vbase(
    mut commands: Commands,
    lines: Query<(Entity, &FromBus), (With<Line>, Without<BranchVBase>)>,
    trafos: Query<(Entity, &TransformerDevice), Without<BranchVBase>>,
    buses: Query<&VNominal>,
    lookup: Res<NodeLookup>,
) {
    for (ent, from) in &lines {
        let bus_ent = lookup.get_entity(from.0)
            .unwrap_or_else(|| panic!("Line from_bus {} has no entity", from.0));
        let vb = buses.get(bus_ent)
            .map(|vn| vn.0.0)
            .unwrap_or_else(|_| panic!("Line from_bus entity missing VNominal"));
        commands.entity(ent).insert(BranchVBase(vb));
    }
    for (ent, trans) in &trafos {
        commands.entity(ent).insert(BranchVBase(trans.vn_lv_kv));
    }
}

/// Compute per-unit shunt admittance for each in-service `ShuntDevice` and attach
/// `ShuntYpu` to the **shunt entity itself** (not to the bus). The bus is read for
/// its `VNominal` via `NodeLookup`; no component is added to the bus entity.
pub fn setup_shunt_admittance(
    mut commands: Commands,
    shunts: Query<(Entity, &ShuntDevice, &TargetBus), (Without<OutOfService>, Without<ShuntYpu>)>,
    buses: Query<&VNominal>,
    lookup: Res<NodeLookup>,
    common: Res<PFCommonData>,
) {
    let sbase = common.sbase;
    for (ent, dev, target) in &shunts {
        let bus_ent = lookup
            .get_entity(target.0)
            .unwrap_or_else(|| panic!("Shunt target bus {} has no entity", target.0));
        let bus_vn = buses
            .get(bus_ent)
            .map(|vn| vn.0.0)
            .unwrap_or_else(|_| panic!("Shunt target bus entity missing VNominal"));
        let scale = (bus_vn / dev.vn_kv).powi(2);
        let step = dev.step as f64;
        let g_pu = dev.p_mw * step * scale / sbase;
        let b_pu = -dev.q_mvar * step * scale / sbase;
        commands.entity(ent).insert(ShuntYpu(Complex64::new(g_pu, b_pu)));
    }
}

#[cfg(test)]
mod tests_line_patch {
    use super::*;
    use crate::basic::ecs::network::{DataOps, PowerGrid};
    use crate::io::pandapower::{load_csv_zip, ecs_net_conv::LoadPandapowerNet};
    use crate::basic::ecs::elements::PandapowerEntityMap;
    use bevy_ecs::system::RunSystemOnce;

    #[test]
    fn test_setup_line_primitive_y_ieee118() {
        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        let _ = pf_net.world_mut().run_system_once(setup_line_primitive_y);

        let line_ents = pf_net.world().resource::<PandapowerEntityMap>().line_entities.clone();
        let lines_pp = net.line.as_deref().unwrap_or(&[]);
        assert_eq!(line_ents.len(), lines_pp.len());

        let wbase = 2.0 * std::f64::consts::PI * net.f_hz;
        let mut n_check = 0usize;

        for (i, l) in lines_pp.iter().enumerate() {
            let comp = pf_net.world().entity(line_ents[i]).get::<Port4MatPatch>();
            // Note: setup_line_primitive_y currently doesn't check in_service.
            // We compute expected against the formula for every line and verify match.
            let length = l.length_km;
            let parallel = l.parallel as f64;
            let rl = l.r_ohm_per_km * length;
            let xl = l.x_ohm_per_km * length;
            let y_series = Complex::new(parallel, 0.0) / Complex::new(rl, xl);
            let b = wbase * 1e-9 * l.c_nf_per_km * length * parallel;
            let g = 1e-6 * l.g_us_per_km * length * parallel;
            let y_shunt = 0.5 * Complex::new(g, b);
            let exp_yff = y_series + y_shunt;
            let exp_yft = -y_series;

            let m = comp.expect("line missing Port4MatPatch").0;
            assert!((m[(0,0)] - exp_yff).norm() < 1e-12,
                "line {i} yff: got {:?}, expected {:?}", m[(0,0)], exp_yff);
            assert!((m[(0,1)] - exp_yft).norm() < 1e-12);
            assert!((m[(1,0)] - exp_yft).norm() < 1e-12);
            assert!((m[(1,1)] - exp_yff).norm() < 1e-12);
            n_check += 1;
        }
        println!("IEEE118: {} lines have correct Port4MatPatch", n_check);
        assert_eq!(n_check, lines_pp.len());
    }

    /// Cross-check against opf::builder::line_admittances — the trusted physics path
    /// already used by current OPF.  Both should produce identical (yff, yft, ytf, ytt)
    /// after scaling our physical-Siemens patch to system per-unit.
    #[test]
    fn test_line_patch_matches_opf_builder_ieee118() {
        use crate::opf::builder::line_admittances;

        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        let _ = pf_net.world_mut().run_system_once(setup_line_primitive_y);

        let line_ents = pf_net.world().resource::<PandapowerEntityMap>().line_entities.clone();
        let lines_pp = net.line.as_deref().unwrap_or(&[]);

        let wbase = 2.0 * std::f64::consts::PI * net.f_hz;
        let base_mva = net.sn_mva;

        for (i, l) in lines_pp.iter().enumerate() {
            // from-bus vbase
            let from_bus = net.bus.iter().find(|b| b.index == l.from_bus).unwrap();
            let vbase_kv = from_bus.vn_kv;

            let (e_yff, e_yft, e_ytf, e_ytt, _) =
                line_admittances(l, vbase_kv, base_mva, wbase);

            let scale = vbase_kv * vbase_kv / base_mva;
            let m = pf_net.world().entity(line_ents[i]).get::<Port4MatPatch>().unwrap().0;

            let got_yff = m[(0,0)] * scale;
            let got_yft = m[(0,1)] * scale;
            let got_ytf = m[(1,0)] * scale;
            let got_ytt = m[(1,1)] * scale;

            assert!((got_yff - e_yff).norm() < 1e-12, "line {i} yff");
            assert!((got_yft - e_yft).norm() < 1e-12, "line {i} yft");
            assert!((got_ytf - e_ytf).norm() < 1e-12, "line {i} ytf");
            assert!((got_ytt - e_ytt).norm() < 1e-12, "line {i} ytt");
        }
        println!("IEEE118: new_pf Port4MatPatch == opf::builder line_admittances on all {} lines", lines_pp.len());
    }

    /// End-to-end equivalence: new_pf's full Ybus assembly path produces the same Ybus
    /// (after un-permuting via PFOrder.map) as the old PF's `create_y_bus`.
    #[test]
    fn test_new_pf_ybus_vs_old_pf_ieee118() {
        use crate::basic::ecs::elements::bus_systems::init_node_lookup;
        use crate::basic::ecs::elements::line_systems::setup_line_systems;
        use crate::basic::ecs::elements::trans::trans_systems::setup_transformer;
        use crate::basic::ecs::powerflow::systems::create_y_bus;
        use std::collections::HashMap;

        let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let net = load_csv_zip(&path).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.load_pandapower_net(&net);
        pf_net.world_mut().insert_resource(NetworkOperators::default());

        // Shared setup
        let _ = pf_net.world_mut().run_system_once(init_node_lookup);
        let _ = pf_net.world_mut().run_system_once(setup_transformer);

        // === Old PF path: setup_line_systems (children + Admittance) + create_y_bus ===
        let _ = pf_net.world_mut().run_system_once(setup_line_systems);
        let (_old_incidence, old_ybus) = pf_net.world_mut()
            .run_system_once(create_y_bus).unwrap();

        // === New PF path: setup_line_primitive_y + setup_branch_vbase + topology + Y_prim + assemble ===
        let _ = pf_net.world_mut().run_system_once(setup_line_primitive_y);
        let _ = pf_net.world_mut().run_system_once(setup_branch_vbase);
        let _ = pf_net.world_mut().run_system_once(initialize_pf_topology_system);
        let _ = pf_net.world_mut().run_system_once(calculate_primitive_y_system);
        let _ = pf_net.world_mut().run_system_once(assemble_ybus_system);

        let new_ybus = pf_net.world().resource::<NetworkOperators>().ybus
            .as_ref().expect("new_pf Ybus not populated").clone();
        let pf_order = pf_net.world().resource::<PFOrder>();
        let map = pf_order.map.clone();  // orig_bus_id → permuted_idx

        // Index new_pf Ybus (in permuted order) by its permuted (row, col)
        let mut new_lookup: HashMap<(usize, usize), Complex64> = HashMap::new();
        for j in 0..new_ybus.ncols() {
            for idx in new_ybus.col_offsets()[j]..new_ybus.col_offsets()[j+1] {
                let r = new_ybus.row_indices()[idx];
                new_lookup.insert((r, j), new_ybus.values()[idx]);
            }
        }

        // Compare: for each (i,j) in old_ybus, look up new_ybus at (map[i], map[j])
        let mut max_diff = 0.0f64;
        let mut n_compared = 0usize;
        let mut n_missing = 0usize;
        for j in 0..old_ybus.ncols() {
            for idx in old_ybus.col_offsets()[j]..old_ybus.col_offsets()[j+1] {
                let i = old_ybus.row_indices()[idx];
                let v_old = old_ybus.values()[idx];
                let v_new = new_lookup.get(&(map[i], map[j]))
                    .cloned().unwrap_or_else(|| { n_missing += 1; Complex64::new(0.0, 0.0) });
                let d = (v_old - v_new).norm();
                if d > max_diff { max_diff = d; }
                n_compared += 1;
            }
        }

        println!(
            "IEEE118: new_pf vs old PF Ybus — compared {} entries, missing in new {}, max_diff = {:.4e}",
            n_compared, n_missing, max_diff
        );
        assert_eq!(n_missing, 0, "new_pf Ybus is missing structural entries that old PF has");
        assert!(max_diff < 1e-10, "Ybus values disagree (max_diff = {:.4e})", max_diff);
    }

}
