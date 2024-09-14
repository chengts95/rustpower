use crate::io::pandapower::SwitchType;
use bevy_ecs::prelude::*;
use bevy_hierarchy::BuildChildren;
use derive_more::{Deref, DerefMut};
use nalgebra::{vector, Complex};
use nalgebra_sparse::ops::Op;
use std::collections::{HashMap, HashSet};

use super::elements::*;
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

/// Merge 2 nodes
#[derive(Default, Debug, Clone, Component)]
pub struct MergeNode(pub usize, pub usize);

#[derive(Default, Debug, Clone)]
/// 并查集结构的实现
pub struct NodeMerge {
    parent: HashMap<u64, u64>,
    rank: HashMap<u64, u64>,
}

#[derive(Default, Debug, Clone, Deref,DerefMut, Resource)]
/// 并查集结构的实现
pub struct NodeMapping(HashMap<u64,u64>);

impl NodeMerge {
    pub fn new(nodes: &[u64]) -> Self {
        let mut parent = HashMap::new();
        let mut rank = HashMap::new();
        for &node in nodes {
            parent.insert(node, node);
            rank.insert(node, 0);
        }
        NodeMerge { parent, rank }
    }

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
    pub fn get_node_mapping(&self, starting_idx: u64) -> HashMap<u64, u64> {
        // 建立节点映射
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

#[allow(dead_code)]
pub fn process_switch_state(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    net: Res<PPNetwork>,
    q: Query<(Entity, &Switch, &SwitchState)>,
) {
    let mut node_idx: Vec<u64> = nodes.0.keys().map(|x| *x as u64).collect();
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
#[allow(dead_code)]
pub fn node_merge_split(mut cmd: Commands, nodes: Res<NodeMapping>) {}

fn build_aggregation_matrix(
    nodes: &[u64],
    node_mapping: &HashMap<u64, usize>,
) -> nalgebra::DMatrix<u32> {
    let original_node_count = nodes.len();
    let new_node_count = node_mapping.values().collect::<HashSet<_>>().len();

    let mut data = vec![0u32; original_node_count * new_node_count];

    for (i, &node) in nodes.iter().enumerate() {
        let new_node = node_mapping[&node];
        data[i * new_node_count + new_node - 1] = 1; // 索引从0开始
    }

    nalgebra::DMatrix::from_row_slice(original_node_count, new_node_count, &data)
}
#[cfg(test)]
#[allow(unused_imports)]

mod tests {
    use std::{env, fs};

    use serde_json::{Map, Value};

    use crate::{
        basic::new_ecs::network::*,
        io::pandapower::{load_pandapower_json, load_pandapower_json_obj},
    };

    use super::*;
    fn load_json_from_str(file_content: &str) -> Result<Map<String, Value>, std::io::Error> {
        let parsed: Value = serde_json::from_str(&file_content)?;
        let obj: Map<String, Value> = parsed.as_object().unwrap().clone();
        Ok(obj)
    }

    fn load_json(file_path: &str) -> Result<Map<String, Value>, std::io::Error> {
        let file_content = fs::read_to_string(file_path)
            .expect(format!("Error reading file network file").as_str());
        let obj = load_json_from_str(&file_content);
        obj
    }
    #[test]
    fn test_node_merge() {
        let nodes = vec![1, 2, 3, 4, 5, 6, 7];
        // 定义开关列表
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

        // 定义开关状态列表，与开关列表对应
        let switch_states = vec![
            SwitchState(true),
            SwitchState(true),
            SwitchState(false),
            SwitchState(true),
        ];

        // 初始化并查集
        let mut uf = NodeMerge::new(&nodes);

        // 处理开关数据
        for (switch, state) in switches.iter().zip(switch_states.iter()) {
            if **state {
                if switch.et == SwitchType::SwitchTwoBuses {
                    uf.union(switch.bus as u64, switch.element as u64);
                }
            }
        }

        // 检查节点的代表元
        assert_eq!(uf.find(2), uf.find(3));
        assert_eq!(uf.find(3), uf.find(4));
        assert_ne!(uf.find(5), uf.find(6));
        assert_eq!(uf.find(6), uf.find(7));

        // 输出结果
        println!("节点到代表元的映射（父节点）：");
        for &node in &nodes {
            println!(
                "节点 {} 的代表元（根节点）是 {}",
                node,
                uf.parent.get(&node).unwrap()
            );
        }

        // // 建立节点映射
        // let mut root_to_new_id = HashMap::new();
        // let mut node_mapping = HashMap::new();
        // let mut new_node_id = 1;

        // for &node in &nodes {
        //     let root = uf.find(node);
        //     if !root_to_new_id.contains_key(&root) {
        //         root_to_new_id.insert(root, new_node_id);
        //         new_node_id += 1;
        //     }
        //     node_mapping.insert(node, root_to_new_id[&root]);
        // }
        // println!("\n节点到新节点编号的映射：");
        // for &node in &nodes {
        //     println!("原始节点 {} 映射到新节点 {}", node, node_mapping[&node]);
        // }
        // // 检查节点映射
        // assert_eq!(node_mapping[&1], 1);
        // assert_eq!(node_mapping[&2], 2);
        // assert_eq!(node_mapping[&3], 2);
        // assert_eq!(node_mapping[&4], 2);
        // assert_eq!(node_mapping[&5], 3);
        // assert_eq!(node_mapping[&6], 4);
        // assert_eq!(node_mapping[&7], 4);
        // // 构建聚合矩阵（可选）
        // let p_matrix = build_aggregation_matrix(&nodes, &node_mapping);
        // println!("\n聚合矩阵 P：\n{}", p_matrix);
    }
    #[test]
    fn test_ecs_pf() {
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
        let mut nodes : Vec<_> = node_mapping.keys().collect();
        nodes.sort();
        println!("\n节点到新节点编号的映射：");
        for &node in &nodes {
            println!("原始节点 {} 映射到新节点 {}", node, node_mapping[&node]);
        }
        pf_net.run_pf();
        assert_eq!(
            pf_net
                .world()
                .get_resource::<PowerFlowResult>()
                .unwrap()
                .converged,
            true
        );
    }
}
