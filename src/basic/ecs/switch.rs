use crate::{
    basic::sparse::{self, cast::Cast},
    io::pandapower::SwitchType,
};
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
use nalgebra::{vector, Complex, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use std::collections::{HashMap, HashSet};

use self::sparse::conj::RealImage;
use super::{elements::*, network::PowerFlowMat, systems::create_permutation_matrix};

/// Represents a network switch in the power flow network.
///
/// A switch connects two buses or a bus and an element, and can have a given impedance (z_ohm).
/// The switch state is defined by its type (`SwitchType`), the connected buses, and its impedance.
#[derive(Default, Debug, Clone, Component)]
pub struct Switch {
    pub bus: i64,       // Identifier for the bus connected by the switch.
    pub element: i64,   // Identifier for the element connected by the switch.
    pub et: SwitchType, // Switch type that defines its behavior.
    pub z_ohm: f64,     // Impedance in ohms for the switch connection.
}

/// Represents the result of node aggregation as a resource in matrix form.
///
/// This structure holds the merged node matrices (`merge_mat` and `merge_mat_v`) after performing the node aggregation process.
#[derive(Default, Debug, Clone, Resource)]
pub struct NodeAggRes {
    pub merge_mat: CscMatrix<f64>, // Aggregation matrix for the merged nodes.
    pub merge_mat_v: CscMatrix<f64>, // Aggregation matrix for voltage values.
}

/// Represents the state of a switch (either open or closed).
///
/// The state (`true` for closed and `false` for open) is wrapped in the `SwitchState` component.
#[derive(Default, Debug, Clone, Component, Deref, DerefMut)]
pub struct SwitchState(pub bool);

/// Represents the merging of two nodes in the power network.
///
/// Each `MergeNode` instance represents a pair of nodes to be merged.
#[derive(Default, Debug, Clone, Component)]
pub struct MergeNode(pub usize, pub usize);

/// Implements a Union-Find structure for efficiently merging nodes in the network.
///
/// This structure is used to manage merging of nodes and to keep track of their relationships.
#[derive(Default, Debug, Clone)]
pub struct NodeMerge {
    pub parent: HashMap<u64, u64>, // Maps each node to its parent in the union-find structure.
    pub rank: HashMap<u64, u64>,   // Rank used for efficient union operations.
}

/// Represents the mapping of original nodes to their merged nodes after aggregation.
#[derive(Default, Debug, Clone, Deref, DerefMut, Resource)]
pub struct NodeMapping(HashMap<u64, u64>);

impl NodeMerge {
    /// Initializes the Union-Find structure for the given nodes.
    ///
    /// Each node starts as its own parent and has a rank of 0.
    ///
    /// # Arguments
    /// * `nodes` - A list of nodes to initialize in the union-find structure.
    pub fn new(nodes: &[u64]) -> Self {
        let mut parent = HashMap::new();
        let mut rank = HashMap::new();
        for &node in nodes {
            parent.insert(node, node);
            rank.insert(node, 0);
        }
        NodeMerge { parent, rank }
    }

    /// Finds the root of a node using path compression for efficiency.
    ///
    /// # Arguments
    /// * `node` - The node whose root is to be found.
    ///
    /// # Returns
    /// * The root of the specified node.
    fn find(&mut self, node: u64) -> u64 {
        let mut root = node;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }

        let mut current = node;
        while self.parent[&current] != root {
            let parent = self.parent[&current];
            self.parent.insert(current, root);
            current = parent;
        }
        root
    }

    /// Merges two nodes by their roots based on rank.
    ///
    /// # Arguments
    /// * `node1` - The first node to merge.
    /// * `node2` - The second node to merge.
    pub fn union(&mut self, node1: u64, node2: u64) {
        let root1 = self.find(node1);
        let root2 = self.find(node2);
        if root1 != root2 {
            let rank1 = self.rank[&root1];
            let rank2 = self.rank[&root2];
            if rank1 < rank2 {
                self.parent.insert(root1, root2);
            } else {
                self.parent.insert(root2, root1);
                if rank1 == rank2 {
                    *self.rank.get_mut(&root1).unwrap() += 1;
                }
            }
        }
    }

    /// Generates a mapping of original nodes to their merged nodes.
    ///
    /// # Arguments
    /// * `starting_idx` - The starting index for assigning new merged node IDs.
    ///
    /// # Returns
    /// * A hashmap mapping original nodes to their merged counterparts.
    pub fn get_node_mapping(&self, starting_idx: u64) -> HashMap<u64, u64> {
        let mut root_to_new_id = HashMap::new();
        let mut node_mapping = HashMap::new();
        let mut new_node_id = starting_idx;
        let mut nodes: Vec<_> = self.parent.keys().collect();
        nodes.sort();
        for &node in &nodes {
            let root = self.parent[&node];
            if !root_to_new_id.contains_key(&root) {
                root_to_new_id.insert(root, new_node_id);
                new_node_id += 1;
            }
            node_mapping.insert(*node, root_to_new_id[&root]);
        }
        node_mapping
    }
}

/// Processes the state of switches and updates network components accordingly.
///
/// This function performs node merging or adds admittance branches based on the state of switches.
#[allow(dead_code)]
pub fn process_switch_state(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    net: Res<PPNetwork>,
    q: Query<(Entity, &Switch, &SwitchState)>,
) {
    let node_idx: Vec<u64> = nodes.0.keys().map(|&x| x as u64).collect();
    let mut union_find: Option<NodeMerge> = if q.iter().count() > 0 {
        Some(NodeMerge::new(&node_idx))
    } else {
        None
    };

    q.iter().for_each(|(entity, switch, closed)| {
        let _z_ohm = switch.z_ohm;
        match switch.et {
            SwitchType::SwitchTwoBuses if **closed && _z_ohm == 0.0 => {
                union_find
                    .as_mut()
                    .unwrap()
                    .union(switch.bus as u64, switch.element as u64);
            }
            SwitchType::SwitchTwoBuses if **closed => {
                let v_base = net.bus[switch.bus as usize].vn_kv;
                cmd.entity(entity).insert(AdmittanceBranch {
                    y: Admittance(Complex::new(_z_ohm, 0.0)),
                    port: Port2(vector![switch.bus, switch.element]),
                    v_base: VBase(v_base),
                });
            }
            _ => {}
        }
    });

    if let Some(union_find) = union_find {
        cmd.insert_resource(NodeMapping(union_find.get_node_mapping(0)));
    }
}

/// Processes the state of switches and updates network components accordingly.
///
/// This function adds admittance branches based on the state of switches, no ideal switch.
#[allow(dead_code)]
pub fn process_switch_state_admit(
    mut cmd: Commands,
    net: Res<PPNetwork>,
    q: Query<(Entity, &Switch, &SwitchState)>,
) {
    q.iter().for_each(|(entity, switch, closed)| {
        let _z_ohm = switch.z_ohm;
        match switch.et {
            SwitchType::SwitchTwoBuses if **closed && _z_ohm == 0.0 => {
                let (node1, node2) = (switch.bus, switch.element);
                let v_base = net.bus[switch.bus as usize].vn_kv;
                cmd.entity(entity).insert(AdmittanceBranch {
                    y: Admittance(Complex::new(1e6, 0.0)),
                    port: Port2(vector![node1, node2]),
                    v_base: VBase(v_base),
                });
            }
            SwitchType::SwitchTwoBuses if **closed => {
                let v_base = net.bus[switch.bus as usize].vn_kv;
                cmd.entity(entity).insert(AdmittanceBranch {
                    y: Admittance(Complex::new(_z_ohm, 0.0)),
                    port: Port2(vector![switch.bus, switch.element]),
                    v_base: VBase(v_base),
                });
            }
            _ => {}
        }
    });
}

/// Builds an aggregation matrix based on the provided node mapping.
///
/// # Arguments
/// * `node_mapping` - A mapping from original nodes to their merged counterparts.
///
/// # Returns
/// * A COO matrix representing the node aggregation.
fn build_aggregation_matrix(node_mapping: &HashMap<u64, u64>) -> CooMatrix<u64> {
    let mut nodes: Vec<_> = node_mapping.keys().collect();
    nodes.sort();
    let original_node_count = nodes.len();
    let new_node_count = node_mapping.values().collect::<HashSet<_>>().len();

    let mut mat = CooMatrix::new(original_node_count, new_node_count);
    for (i, &node) in nodes.iter().enumerate() {
        let new_node = node_mapping.get(&node).unwrap_or(&node);
        mat.push(i, *new_node as usize, 1);
    }
    mat
}

/// Creates a reverse mapping from merged nodes to their original nodes.
///
/// # Arguments
/// * `node_mapping` - A mapping from original nodes to their merged counterparts.
///
/// # Returns
/// * A hashmap that maps merged nodes back to their original nodes.
fn build_reverse_mapping(node_mapping: &HashMap<u64, u64>) -> HashMap<u64, Vec<u64>> {
    let mut reverse_mapping: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&original_node, &merged_node) in node_mapping {
        reverse_mapping
            .entry(merged_node)
            .or_default()
            .push(original_node);
    }
    reverse_mapping
}

/// Sets a mask for merged nodes based on node types (PV, PQ, EXT).
///
/// # Arguments
/// * `node_mapping` - A mapping from original nodes to their merged counterparts.
/// * `current_node_order` - A slice representing the current order of nodes.
/// * `mats_npv` - Number of PV nodes.
/// * `mats_npq` - Number of PQ nodes.
///
/// # Returns
/// * A vector representing the mask for merged nodes.
fn set_mask_for_merged_nodes(
    node_mapping: &HashMap<u64, u64>,
    current_node_order: &[u64],
    mats_npv: usize,
    mats_npq: usize,
) -> DVector<bool> {
    let ext_idx = mats_npv + mats_npq;
    let pv_nodes: HashSet<_> = current_node_order[0..mats_npv].iter().copied().collect();
    let ext_nodes: HashSet<_> = current_node_order[ext_idx..].iter().copied().collect();
    let reverse_mapping = build_reverse_mapping(node_mapping);
    let mut mask = DVector::from_element(current_node_order.len(), false);

    for original_nodes in reverse_mapping.values() {
        let prioritized_node = original_nodes
            .iter()
            .find(|&&node| ext_nodes.contains(&node))
            .or_else(|| {
                original_nodes
                    .iter()
                    .find(|&&node| pv_nodes.contains(&node))
            })
            .or_else(|| original_nodes.iter().min());
        if let Some(&node) = prioritized_node {
            mask[node as usize] = true;
        }
    }
    mask
}

/// Executes the node aggregation process and returns the aggregation matrices.
///
/// # Arguments
/// * `node_mapping` - Resource containing the mapping of nodes.
/// * `mats` - Power flow matrix resource.
///
/// # Returns
/// * A tuple containing two CSC matrices, one for aggregation and one for voltage values.
pub fn node_aggregation_system(
    node_mapping: Res<NodeMapping>,
    mats: Res<PowerFlowMat>,
) -> (CscMatrix<f64>, CscMatrix<f64>) {
    let coo = build_aggregation_matrix(&node_mapping.0);
    let mut nodes: Vec<_> = node_mapping.keys().copied().collect();
    nodes.sort_unstable();
    let current_node_order =
        (&mats.reorder * DVector::from_vec(nodes).cast::<Complex<f64>>()).map(|x| x.re as u64);
    let mask = set_mask_for_merged_nodes(
        &node_mapping,
        current_node_order.as_slice(),
        mats.npv,
        mats.npq,
    );

    let (pattern, values) = CscMatrix::from(&coo).into_pattern_and_values();

    let pre_select_mat = unsafe {
        CscMatrix::try_from_pattern_and_values(
            pattern,
            values.iter().copied().map(|x| x as f64).collect(),
        )
        .unwrap_unchecked()
    };

    let pre_select_mat_for_voltages = pre_select_mat.filter(|r, _, _| mask[r]);
    (pre_select_mat, pre_select_mat_for_voltages)
}

/// Updates the power flow matrix with the new merged node mappings.
///
/// # Arguments
/// * `agg_mats` - Input aggregation matrices.
/// * `node_mapping` - Node mapping resource.
/// * `pf_mats` - Power flow matrix resource to be updated.
/// * `cmd` - Commands to interact with the ECS.
pub fn handle_node_merge(
    In(agg_mats): In<(CscMatrix<f64>, CscMatrix<f64>)>,
    node_mapping: Res<NodeMapping>,
    pf_mats: ResMut<PowerFlowMat>,
    mut cmd: Commands,
) {
    let (mat, mat_v) = agg_mats;

    let nodes = get_sorted_nodes(&node_mapping);
    let input_vector = DVector::from_iterator(nodes.len(), nodes.iter().map(|&x| x as f64));
    let merged_v_vector = calculate_merged_vector(&mat_v, &input_vector);

    let mats = &pf_mats;
    let (pv_nodes, pq_nodes, ext_nodes) = extract_pv_pq_ext_nodes(mats, &input_vector);

    let (pv, pq, ext, _old_to_new) = filter_and_remap_nodes(
        pv_nodes,
        pq_nodes,
        ext_nodes,
        merged_v_vector.as_slice(),
        mats.v_bus_init.len(),
    );

    let new_total_nodes = merged_v_vector.len();
    let mut mats = pf_mats;
    update_power_flow_matrix(&mut mats, pv, pq, ext, &mat, &mat_v, new_total_nodes);
    cmd.insert_resource(NodeAggRes {
        merge_mat: mat,
        merge_mat_v: mat_v,
    });
}

/// Sorts the nodes based on their keys from the `NodeMapping`.
///
/// # Arguments
/// * `node_mapping` - The node mapping to be sorted.
///
/// # Returns
/// * A sorted vector of node keys.
fn get_sorted_nodes(node_mapping: &NodeMapping) -> Vec<u64> {
    let mut nodes: Vec<_> = node_mapping.keys().cloned().collect();
    nodes.sort_unstable();
    nodes
}

/// Calculates the merged vector based on the given aggregation matrix and input vector.
///
/// # Arguments
/// * `mat_v` - The aggregation matrix for voltages.
/// * `input_vector` - The input vector representing nodes.
///
/// # Returns
/// * A vector containing the merged node indices.
fn calculate_merged_vector(mat_v: &CscMatrix<f64>, input_vector: &DVector<f64>) -> DVector<i64> {
    (&mat_v.clone().transpose() * input_vector).map(|x| x as i64)
}

/// Extracts PV, PQ, and EXT nodes from the reordered structure.
///
/// # Arguments
/// * `mats` - Power flow matrix resource.
/// * `input_vector` - The vector representing the current node structure.
///
/// # Returns
/// * A tuple containing PV nodes, PQ nodes, and EXT nodes as vectors.
fn extract_pv_pq_ext_nodes(
    mats: &PowerFlowMat,
    input_vector: &DVector<f64>,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let reordered_v_before = &mats.reorder.real() * input_vector;
    let reordered_v_before = reordered_v_before.map(|x| x as i64);

    let (npv, npq, _total_nodes) = (mats.npv, mats.npq, mats.v_bus_init.len());
    let ext_idx = npv + npq;

    let pv_nodes = reordered_v_before.as_slice()[0..npv].to_vec();
    let pq_nodes = reordered_v_before.as_slice()[npv..npq].to_vec();
    let ext_nodes = reordered_v_before.as_slice()[ext_idx..].to_vec();

    (pv_nodes, pq_nodes, ext_nodes)
}

/// Filters and remaps the nodes based on the given merged vector and total node count.
///
/// # Arguments
/// * `pv_nodes` - PV nodes vector.
/// * `pq_nodes` - PQ nodes vector.
/// * `ext_nodes` - EXT nodes vector.
/// * `merged_v_vector` - The vector containing merged node indices.
/// * `total_nodes` - The total number of nodes before merging.
///
/// # Returns
/// * A tuple containing the filtered PV, PQ, EXT nodes, and the mapping from old to new indices.
fn filter_and_remap_nodes(
    pv_nodes: Vec<i64>,
    pq_nodes: Vec<i64>,
    ext_nodes: Vec<i64>,
    merged_v_vector: &[i64],
    total_nodes: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let merged_v_set: HashSet<_> = merged_v_vector.iter().cloned().collect();
    let pv_nodes_set: HashSet<_> = pv_nodes.iter().cloned().collect();
    let pq_nodes_set: HashSet<_> = pq_nodes.iter().cloned().collect();
    let ext_nodes_set: HashSet<_> = ext_nodes.iter().cloned().collect();

    let pv = pv_nodes_set
        .intersection(&merged_v_set)
        .cloned()
        .collect::<Vec<_>>();
    let pq = pq_nodes_set
        .intersection(&merged_v_set)
        .cloned()
        .collect::<Vec<_>>();
    let ext = ext_nodes_set
        .intersection(&merged_v_set)
        .cloned()
        .collect::<Vec<_>>();

    if ext.is_empty() {
        panic!("cannot find ext grid after merge!");
    }

    let mut pv = pv.iter().cloned().collect::<Vec<_>>();
    let mut pq = pq.iter().cloned().collect::<Vec<_>>();
    let mut ext = ext.iter().cloned().collect::<Vec<_>>();
    pv.sort_unstable();
    pq.sort_unstable();
    ext.sort_unstable();

    let mut old_to_new = vec![-1; total_nodes];
    for (new_idx, &old_idx) in merged_v_vector.iter().enumerate() {
        old_to_new[old_idx as usize] = new_idx as i64;
    }

    pv.iter_mut()
        .chain(pq.iter_mut())
        .chain(ext.iter_mut())
        .for_each(|x| *x = old_to_new[*x as usize]);

    (pv, pq, ext, old_to_new)
}

/// Updates the power flow matrix based on the new node structure after aggregation.
///
/// # Arguments
/// * `mats` - Power flow matrix resource to be updated.
/// * `pv` - PV nodes vector.
/// * `pq` - PQ nodes vector.
/// * `ext` - EXT nodes vector.
/// * `mat` - Aggregation matrix.
/// * `mat_v` - Aggregation matrix for voltages.
/// * `new_total_nodes` - The new total number of nodes after merging.
fn update_power_flow_matrix(
    mats: &mut PowerFlowMat,
    pv: Vec<i64>,
    pq: Vec<i64>,
    ext: Vec<i64>,
    mat: &CscMatrix<f64>,
    mat_v: &CscMatrix<f64>,
    new_total_nodes: usize,
) {
    let permutation_matrix = create_permutation_matrix(&pv, &pq, &ext, new_total_nodes);
    mats.reorder = sparse::cast::Cast::<_>::cast(&CsrMatrix::from(&permutation_matrix));
    mats.npq = pq.len();
    mats.npv = pv.len();
    mats.y_bus = mat.transpose().cast() * &mats.y_bus * &mat.cast();
    mats.s_bus = mat.transpose().cast() * &mats.s_bus;
    mats.v_bus_init = mat_v.transpose().cast() * &mats.v_bus_init;
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use std::{env, fs};

    use bevy_ecs::system::RunSystemOnce;
    use nalgebra_sparse::CsrMatrix;
    use serde_json::{Map, Value};

    use crate::{
        basic::{
            ecs::{
                network::*, post_processing::PostProcessing, systems::create_permutation_matrix,
            },
            sparse::{self, conj::RealImage},
        },
        io::pandapower::{load_pandapower_json, load_pandapower_json_obj},
    };

    use super::*;

    /// Loads a JSON object from a string.
    fn load_json_from_str(file_content: &str) -> Result<Map<String, Value>, std::io::Error> {
        let parsed: Value = serde_json::from_str(&file_content)?;
        let obj: Map<String, Value> = parsed.as_object().unwrap().clone();
        Ok(obj)
    }

    /// Loads a JSON object from a file.
    fn load_json(file_path: &str) -> Result<Map<String, Value>, std::io::Error> {
        let file_content = fs::read_to_string(file_path).expect("Error reading network file");
        let obj = load_json_from_str(&file_content);
        obj
    }

    #[test]
    /// Tests the node merging logic using union-find (disjoint set).
    fn test_node_merge() {
        let nodes = vec![1, 2, 3, 4, 5, 6, 7];
        let switches = vec![
            Switch {
                bus: 2,
                element: 3,
                et: SwitchType::SwitchTwoBuses,
                z_ohm: 0.0,
            },
            Switch {
                bus: 3,
                element: 4,
                et: SwitchType::SwitchTwoBuses,
                z_ohm: 0.0,
            },
            Switch {
                bus: 5,
                element: 6,
                et: SwitchType::SwitchTwoBuses,
                z_ohm: 0.0,
            },
            Switch {
                bus: 6,
                element: 7,
                et: SwitchType::SwitchTwoBuses,
                z_ohm: 0.0,
            },
        ];

        let switch_states = vec![
            SwitchState(true),
            SwitchState(true),
            SwitchState(false),
            SwitchState(true),
        ];

        let mut uf = NodeMerge::new(&nodes);

        for (switch, state) in switches.iter().zip(switch_states.iter()) {
            if **state {
                if switch.et == SwitchType::SwitchTwoBuses {
                    uf.union(switch.bus as u64, switch.element as u64);
                }
            }
        }

        assert_eq!(uf.find(2), uf.find(3));
        assert_eq!(uf.find(3), uf.find(4));
        assert_ne!(uf.find(5), uf.find(6));
        assert_eq!(uf.find(6), uf.find(7));
    }

    #[test]
    /// Tests the entire power flow ECS system, including switch processing.
    fn test_node_agg_mat() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/test/", dir);
        let name = folder.to_owned() + "/new_input_PFLV_modified.json";
        let json = load_json(&name).unwrap();
        let json: Map<String, Value> = json
            .get("pp_network")
            .and_then(|v| v.as_object())
            .unwrap()
            .clone();
        let net = load_pandapower_json_obj(&json);
        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.init_pf_net();

        // 3. 运行系统并获取结果矩阵 `mat` 和 `mat_v`
        let (mat, mat_v) = pf_net.world_mut().run_system_once(node_aggregation_system);

        // 4. 获取节点映射
        let node_mapping = pf_net.world().get_resource::<NodeMapping>().unwrap();

        let mut nodes: Vec<_> = node_mapping.keys().cloned().collect();
        nodes.sort();

        // 5. 设置测试节点和目标节点
        let merged_nodes = [12, 28, 30];
        let target_node = 0; // 合并的目标节点

        // 6. 构造节点向量并与 `mat_v` 相乘，验证 `mat_v` 的合并效果
        let input_vector = DVector::from_iterator(nodes.len(), nodes.iter().map(|&x| x as f64));
        let result_vector_v = &mat_v.clone().transpose_as_csr() * &input_vector;

        // 确保向量维度符合预期
        assert_eq!(
            result_vector_v.len(),
            nodes.len() - merged_nodes.len(),
            "Result vector dimension mismatch"
        );

        // 检查 `mat_v` 乘完向量后的合并效果
        for node in &merged_nodes {
            assert_eq!(
                result_vector_v.map(|x| x as i32).as_slice().contains(node),
                false,
                "Node {} should be zero after merging in mat_v",
                node
            );
        }

        // 7. 检查 `mat` 的合并效果，确保节点 `0` 是 `12 + 28 + 30` 的累加
        let result_vector = &mat.transpose_as_csr() * &input_vector;
        let expected_sum: f64 = merged_nodes.iter().map(|&n| n as f64).sum();

        assert_eq!(
            result_vector[target_node], expected_sum,
            "Node 0 should equal the sum of nodes 12, 28, 30 in mat"
        );
        println!("All node merges and calculations are correct!");
    }

    #[test]
    /// Tests the entire power flow ECS system, including switch processing.
    fn test_node_agg_pf_mats() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/test/", dir);
        let name = folder.to_owned() + "/new_input_PFLV_modified.json";
        let json = load_json(&name).unwrap();
        let json: Map<String, Value> = json
            .get("pp_network")
            .and_then(|v| v.as_object())
            .unwrap()
            .clone();
        let net = load_pandapower_json_obj(&json);
        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.init_pf_net();

        // Step 3: Run system and retrieve result matrices
        let (mat, mat_v) = pf_net.world_mut().run_system_once(node_aggregation_system);

        // Step 4: Retrieve node mapping and generate input vector
        let node_mapping = pf_net.world().get_resource::<NodeMapping>().unwrap();
        let nodes = get_sorted_nodes(node_mapping);
        let input_vector = DVector::from_iterator(nodes.len(), nodes.iter().map(|&x| x as f64));
        let merged_v_vector = calculate_merged_vector(&mat_v, &input_vector);

        // Step 5: Extract PV, PQ, EXT nodes from the reordered structure
        let mats = pf_net.world().get_resource::<PowerFlowMat>().unwrap();
        let (pv_nodes, pq_nodes, ext_nodes) = extract_pv_pq_ext_nodes(mats, &input_vector);

        // Step 6: Filter and remap nodes, verify that only nodes 12, 28, 30 are merged
        let (pv, pq, ext, old_to_new) = filter_and_remap_nodes(
            pv_nodes,
            pq_nodes,
            ext_nodes,
            merged_v_vector.as_slice(),
            mats.v_bus_init.len(),
        );

        // Check that nodes 12, 28, 30 have been merged (old_to_new contains -1 for these)
        assert_eq!(old_to_new[12], -1, "Node 12 was not merged correctly.");
        assert_eq!(old_to_new[28], -1, "Node 28 was not merged correctly.");
        assert_eq!(old_to_new[30], -1, "Node 30 was not merged correctly.");

        // Step 7: Verify that the total number of nodes is now 28
        let new_total_nodes = merged_v_vector.len();
        assert_eq!(
            new_total_nodes, 28,
            "Total nodes after merging should be 28."
        );

        // Verify that nodes 29 and 30 are swapped in the reordered structure
        let reordered_v_before = &mats.reorder.real() * &input_vector;
        let reordered_v_before = reordered_v_before.map(|x| x as i64);
        assert_eq!(
            reordered_v_before[29], 30,
            "Node 29 should map to position 30 after reordering."
        );
        assert_eq!(
            reordered_v_before[30], 29,
            "Node 30 should map to position 29 after reordering."
        );

        // Step 8: Update PowerFlowMat and verify permutation matrix dimensions
        let mut mats = pf_net
            .world_mut()
            .get_resource_mut::<PowerFlowMat>()
            .unwrap();
        update_power_flow_matrix(&mut mats, pv, pq, ext, &mat, &mat_v, new_total_nodes);
        assert_eq!(
            mats.reorder.nrows(),
            new_total_nodes,
            "Reorder matrix row count should match new total nodes."
        );
        assert_eq!(
            mats.reorder.ncols(),
            new_total_nodes,
            "Reorder matrix column count should match new total nodes."
        );

        // Step 9: Check resulting matrices (optional, for further verification)
        assert_eq!(
            mats.y_bus.nrows(),
            new_total_nodes,
            "Y bus matrix should have dimensions matching new total nodes."
        );
        assert_eq!(
            mats.v_bus_init.nrows(),
            new_total_nodes,
            "V bus init matrix should have dimensions matching new total nodes."
        );
    }

    #[test]
    /// Tests the power flow calculation and generation of aggregation matrix.
    fn test_ecs_pf_switch() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/test/", dir);
        let name = folder.to_owned() + "/new_input_PFLV_modified.json";
        let json = load_json(&name).unwrap();
        let json: Map<String, Value> = json
            .get("pp_network")
            .and_then(|v| v.as_object())
            .unwrap()
            .clone();
        let net = load_pandapower_json_obj(&json);
        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.init_pf_net();
        let mut node_process_schedule = Schedule::default();

        node_process_schedule.add_systems(node_aggregation_system.pipe(handle_node_merge));
        node_process_schedule.run(pf_net.world_mut());
        let mat = pf_net.world().resource::<PowerFlowMat>();
        println!("{:?}", mat.v_bus_init);
        pf_net.run_pf();
        pf_net.post_process();
        pf_net.print_res_bus();
    }
}
