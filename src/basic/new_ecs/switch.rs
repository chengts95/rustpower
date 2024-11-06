use crate::{basic::sparse::cast::Cast, io::pandapower::SwitchType};
use bevy_ecs::prelude::*;
use derive_more::{Deref, DerefMut};
use nalgebra::{vector, Complex, DMatrix, DMatrixView, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use num_traits::Zero;
use std::collections::{HashMap, HashSet};

use super::{elements::*, network::PowerFlowMat};

/// Represents a switch in the network.
#[derive(Default, Debug, Clone, Component)]
pub struct Switch {
    pub bus: i64,
    pub element: i64,
    pub et: SwitchType,
    pub z_ohm: f64,
}

/// Represents a switch state in the network.
#[derive(Default, Debug, Clone, Component, Deref, DerefMut)]
pub struct SwitchState(pub bool);

/// Represents merging two nodes in the network.
#[derive(Default, Debug, Clone, Component)]
pub struct MergeNode(pub usize, pub usize);

/// A union-find (disjoint set) structure for merging nodes.
#[derive(Default, Debug, Clone)]
pub struct NodeMerge {
    pub parent: HashMap<u64, u64>,
    pub rank: HashMap<u64, u64>,
}

/// A mapping from old nodes to new nodes after merging, stored as a resource.
#[derive(Default, Debug, Clone, Deref, DerefMut, Resource)]
pub struct NodeMapping(HashMap<u64, u64>);

impl NodeMerge {
    /// Creates a new union-find (disjoint set) structure for the given nodes.
    pub fn new(nodes: &[u64]) -> Self {
        let mut parent = HashMap::new();
        let mut rank = HashMap::new();
        for &node in nodes {
            parent.insert(node, node);
            rank.insert(node, 0);
        }
        NodeMerge { parent, rank }
    }

    /// Finds the root of the node, with path compression.
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

    /// Unites two nodes by their roots.
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

    /// Generates a node mapping based on union-find results, starting with a given index.
    pub fn get_node_mapping(&self, starting_idx: u64) -> HashMap<u64, u64> {
        let mut root_to_new_id = HashMap::new();
        let mut node_mapping = HashMap::new();
        let mut new_node_id = starting_idx;
        let mut nodes: Vec<_> = self.parent.keys().collect();
        nodes.sort();
        for &node in &nodes {
            let root = self.parent.get(&(*node as u64)).unwrap();
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
#[allow(dead_code)]
pub fn process_switch_state(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    net: Res<PPNetwork>,
    q: Query<(Entity, &Switch, &SwitchState)>,
) {
    let node_idx: Vec<u64> = nodes.0.keys().map(|x| *x as u64).collect();
    let mut union_find: Option<NodeMerge> = if q.iter().len() > 0 {
        Some(NodeMerge::new(&node_idx))
    } else {
        None
    };

    q.iter().for_each(|(entity, switch, closed)| {
        let _z_ohm = switch.z_ohm;

        match switch.et {
            SwitchType::SwitchBusLine => todo!(),
            SwitchType::SwitchBusTransformer => todo!(),
            SwitchType::SwitchTwoBuses => {
                let (node1, node2) = (switch.bus, switch.element);
                if **closed {
                    if _z_ohm == 0.0 {
                        union_find
                            .as_mut()
                            .unwrap()
                            .union(node1 as u64, node2 as u64);
                        // let v_base = net.bus[switch.bus as usize].vn_kv;
                        // cmd.entity(entity).insert(AdmittanceBranch {
                        //     y: Admittance(Complex::new(1e6, 0.0)),
                        //     port: Port2(vector![node1, node2]),
                        //     v_base: VBase(v_base),
                        // });
                    } else {
                        let v_base = net.bus[switch.bus as usize].vn_kv;
                        cmd.entity(entity).insert(AdmittanceBranch {
                            y: Admittance(Complex::new(_z_ohm, 0.0)),
                            port: Port2(vector![node1, node2]),
                            v_base: VBase(v_base),
                        });
                    }
                }
            }
            SwitchType::SwitchBusTransformer3w | SwitchType::Unknown => {}
        }
    });

    if union_find.is_some() {
        cmd.insert_resource(NodeMapping(union_find.unwrap().get_node_mapping(0)));
    }
}

/// Placeholder function for future node merge or split logic.
#[allow(dead_code)]
pub fn node_merge_split(_cmd: Commands, _nodes: Res<NodeMapping>) {}
#[allow(dead_code)]

/// Builds an aggregation matrix based on the provided nodes and node mapping.
fn build_aggregation_matrix(node_mapping: &HashMap<u64, u64>) -> CooMatrix<u64> {
    let mut nodes: Vec<_> = node_mapping.keys().collect();
    nodes.sort();
    let original_node_count = nodes.len();
    let new_node_count = node_mapping.values().collect::<HashSet<_>>().len();

    // Initialize the COO matrix
    let mut mat = CooMatrix::new(original_node_count, new_node_count);

    // Iterate over the nodes and apply the mapping
    for (i, &node) in nodes.iter().enumerate() {
        // Get the mapped new node, default to the original node if not in mapping
        let new_node = node_mapping.get(&node).unwrap_or(&node);
        // Push the value 1 to the corresponding location
        mat.push(i, *new_node as usize, 1);
    }

    mat
}
/// Builds an aggregation matrix based on the provided nodes and node mapping.
// fn build_aggregation_matrix_masked(
//     node_mapping: &HashMap<u64, u64>,
//     mask: &[bool],
// ) -> CooMatrix<u64> {
//     let mut nodes: Vec<_> = node_mapping.keys().collect();
//     nodes.sort();
//     let original_node_count = nodes.len();
//     let new_node_count = node_mapping.values().collect::<HashSet<_>>().len();

//     // Initialize the COO matrix
//     let mut mat = CooMatrix::new(original_node_count, new_node_count);

//     // Iterate over the nodes and apply the mapping
//     for (i, &node) in nodes.iter().enumerate() {
//         // Get the mapped new node, default to the original node if not in mapping
//         let new_node = node_mapping.get(&node).unwrap_or(&node);

//         // Push the value 1 to the corresponding location
//         mat.push(i, *new_node as usize,  mask[i] as u64);

//     }

//     mat
// }

fn build_reverse_mapping(node_mapping: &HashMap<u64, u64>) -> HashMap<u64, Vec<u64>> {
    let mut reverse_mapping: HashMap<u64, Vec<u64>> = HashMap::with_capacity(node_mapping.len());

    for (&original_node, &merged_node) in node_mapping {
        reverse_mapping
            .entry(merged_node)
            .or_insert_with(Vec::new)
            .push(original_node);
    }

    reverse_mapping
}

// 假设 `node_mapping` 是 HashMap<u64, u64> 类型
fn set_mask_for_merged_nodes(
    node_mapping: &HashMap<u64, u64>,
    current_node_order: &[u64],
    mats_npv: usize,
    mats_npq: usize,
) -> DVector<bool> {
    // 定义节点类型区域索引
    let ext_idx = mats_npv + mats_npq;
    let pv_nodes = &current_node_order[0..mats_npv];
    let ext_nodes = &current_node_order[ext_idx..];

    // 创建反向映射，键为合并节点，值为合并前的节点集合
    let reverse_mapping = build_reverse_mapping(node_mapping);

    // 初始化一个 mask 向量，初始值全为 0
    let mut mask = DVector::from_element(current_node_order.len(), false);

    // 查找并设置合并节点的 mask 优先级
    for (_, original_nodes) in &reverse_mapping {
        // 查找最高优先级的节点：ext > pv > pq (按最小编号)
        let prioritized_node = original_nodes
            .iter()
            .find(|&&node| ext_nodes.contains(&node))
            .or_else(|| {
                original_nodes
                    .iter()
                    .find(|&&node| pv_nodes.contains(&node))
            })
            .or_else(|| original_nodes.iter().min_by_key(|&&node| node as u64));

        // 设置 mask，找到的节点按优先级设为 1
        if let Some(&node) = prioritized_node {
            mask[node as usize] = true;
        }
    }

    mask
}

fn node_aggregation_system(
    node_mapping: Res<NodeMapping>,
    mats: Res<PowerFlowMat>,
) -> (CscMatrix<f64>, CscMatrix<f64>) {
    let coo = build_aggregation_matrix(&node_mapping.0);
    let mut nodes: Vec<_> = node_mapping.keys().map(|k| k.clone()).collect();

    nodes.sort();

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
        CscMatrix::try_from_pattern_and_values(pattern, values.iter().map(|x| *x as f64).collect())
            .unwrap_unchecked()
    };

    // let mut binding = csc.transpose_as_csr();
    let pre_select_mat_for_voltages = pre_select_mat.filter(|r, _c, _v| {
        return mask[r];
    });

    (pre_select_mat, pre_select_mat_for_voltages)
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use std::{env, fs};

    use bevy_ecs::system::RunSystemOnce;
    use serde_json::{Map, Value};

    use crate::{
        basic::{
            new_ecs::{network::*, post_processing::PostProcessing},
            sparse::conj::RealImage,
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

        // 3. 运行系统并获取结果矩阵 `mat` 和 `mat_v`
        let (mat, mat_v) = pf_net.world_mut().run_system_once(node_aggregation_system);

        // 4. 获取节点映射
        let node_mapping = pf_net.world().get_resource::<NodeMapping>().unwrap();

        let mats = pf_net.world().get_resource::<PowerFlowMat>().unwrap();
        println!(
            "s1: {:?},{:?}",
            (mat_v.nrows(), mat_v.ncols()),
            (mats.reorder.nrows(), mats.reorder.ncols())
        );

        let mut nodes: Vec<_> = node_mapping.keys().cloned().collect();
        nodes.sort();
        let input_vector = DVector::from_iterator(nodes.len(), nodes.iter().map(|&x| x as f64));
        let m2 = &mat_v.clone().transpose() * &input_vector;

        let (npv,npq,nodes) = (mats.npv,mats.npq,mats.v_bus_init.len());
        let reordered_v_before = &mats.reorder.real() * &input_vector;
        let reordered_v_before =reordered_v_before.map(|x| x as u64);
        let ext_idx = npv + npq;
        let pv_nodes = &reordered_v_before.as_slice()[0..npv];
        let pv_nodes = &reordered_v_before.as_slice()[npv..npq];
        let ext_nodes = &reordered_v_before.as_slice()[ext_idx..];
        println!("result: {} {}", m2, reordered_v_before);


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
        let node_mapping = pf_net.world().get_resource::<NodeMapping>().unwrap();
        let mut nodes: Vec<u64> = node_mapping.keys().map(|x| *x).collect();
        nodes.sort();

        // let p_matrix = build_aggregation_matrix(nodes.as_slice(), &node_mapping.0);
        // println!("\nAggregation Matrix P:\n{:?}", p_matrix);
    }
}
