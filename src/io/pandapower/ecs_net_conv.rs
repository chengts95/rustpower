use bevy_ecs::system::RunSystemOnce;
use derive_more::{Deref, DerefMut};
use network::GND;

use crate::basic::new_ecs::*;
use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use bevy_hierarchy::prelude::*;
use elements::*;
use nalgebra::vector;
use nalgebra::Complex;
use std::collections::HashMap;
use std::default;
use std::f64::consts::PI;

use crate::prelude::ExtGridNode;
use crate::prelude::PQNode;
use crate::prelude::PVNode;

#[derive(Debug, Component, Deref, DerefMut)]
pub struct ElemIdx(pub usize);
#[derive(Debug, Component, Deref, DerefMut)]
pub struct PFNode(pub usize);

#[derive(Default, Debug, Resource)]
pub struct NodeLookup(pub HashMap<i64, Entity>);
#[derive(Debug, Component)]
pub struct AuxNode {
    pub bus: i64,
}
#[derive(Debug, Component)]
pub struct Line;
#[derive(Debug, Component)]
pub struct Transformer;

#[derive(Debug, Resource)]
pub struct PFCommonData {
    pub wbase: f64,
    pub sbase: f64,
}

#[derive(Debug, Clone, Copy, Default, Component)]
pub struct PQLoad {
    /// The complex power injected at the node.
    pub s: Complex<f64>,
    /// The bus identifier of the node.
    pub bus: i64,
}

#[derive(Debug, Component)]
pub enum NodeType {
    PQ(PQNode),
    PV(PVNode),
    EXT(ExtGridNode),
    AUX(AuxNode),
}
impl Default for NodeType {
    fn default() -> Self {
        NodeType::PQ(PQNode::default())
    }
}

impl From<PQNode> for NodeType {
    fn from(node: PQNode) -> Self {
        NodeType::PQ(node)
    }
}

impl From<PVNode> for NodeType {
    fn from(node: PVNode) -> Self {
        NodeType::PV(node)
    }
}

impl From<ExtGridNode> for NodeType {
    fn from(node: ExtGridNode) -> Self {
        NodeType::EXT(node)
    }
}

impl From<AuxNode> for NodeType {
    fn from(node: AuxNode) -> Self {
        NodeType::AUX(node)
    }
}
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

fn line_to_admit(cmd: &mut Commands, net: &Network) {
    let wbase = 2.0 * PI * net.f_hz;
    let bus = &net.bus;
    net.line
        .as_ref()
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
        .for_each(|(idx, line)| {
            let b = wbase * 1e-9 * line.c_nf_per_km * line.length_km * (line.parallel as f64);
            let g = line.g_us_per_km * line.length_km * 1e-6 * (line.parallel as f64);
            let v_base = bus[line.from_bus as usize].vn_kv;
            let a = Admittance(0.5 * Complex { re: g, im: b });
            let mut entity = cmd.spawn((
                Line,
                ElemIdx(idx),
                Port2(vector![line.from_bus, line.to_bus]),
            ));

            if line.g_us_per_km != 0.0 || line.c_nf_per_km != 0.0 {
                let (mut shunt_f, mut shunt_t) =
                    (AdmittanceBranch::default(), AdmittanceBranch::default());
                shunt_f.y = a.clone();
                *shunt_f.v_base = v_base;
                shunt_t.y = a;
                *shunt_t.v_base = v_base;
                shunt_f.port = Port2(vector![line.from_bus as i64, GND.into()]);
                shunt_t.port = Port2(vector![line.to_bus as i64, GND.into()]);
                entity.with_children(|p| {
                    p.spawn(shunt_f);
                    p.spawn(shunt_t);
                });
            }

            let rl = line.r_ohm_per_km * line.length_km * (line.parallel as f64);
            let xl = line.x_ohm_per_km * line.length_km * (line.parallel as f64);
            let l = AdmittanceBranch {
                y: Admittance(1.0 / Complex { re: rl, im: xl }),
                port: Port2(vector![line.from_bus as i64, line.to_bus as i64]),
                v_base: VBase(v_base),
            };
            entity.with_children(|p| {
                p.spawn(l);
            });
        });
}
/// Converts a transformer to its equivalent admittance branches.
fn trafo_to_admit(cmd: &mut Commands, net: &Network)  {
    net.trafo.as_ref()
    .unwrap_or(&Vec::new())
    .iter()
    .enumerate()
    .for_each(|(idx, item)|{
        let v_base = item.vn_lv_kv;
        let vkr = item.vkr_percent * 0.01;
        let vk = item.vk_percent * 0.01;
    
        let tap_m = 1.0
            + (item.tap_pos.unwrap_or(0.0) - item.tap_neutral.unwrap_or(0.0))
                * 0.01
                * item.tap_step_percent.unwrap_or(0.0);
        let zbase = v_base * v_base / item.sn_mva;
        let z = zbase * vk;
        let parallel = item.parallel;
    
        let re = zbase * vkr;
        let im = (z.powi(2) - re.powi(2)).sqrt();
        let port = Port2(vector![item.hv_bus.into(), item.lv_bus.into()]);
        let y = 1.0 / (Complex { re, im } * parallel as f64);
        let mut entity = cmd.spawn((
            Transformer,
            ElemIdx(idx),
            Port2(vector![item.hv_bus as i64,item.lv_bus as i64]),
        ));
    });

    // let sc = AdmittanceBranch {
    //     y: Admittance(y / tap_m), 
    //     port,
    //     v_base,
    // };
    // let mut v = Vec::new();
    // v.push(sc);
    // v.push(AdmittanceBranch {
    //     y: Admittance((1.0 - tap_m) * y / tap_m.powi(2)),
    //     port: Port2(vector![item.hv_bus, GND]),
    //     v_base,
    // });
    // v.push(AdmittanceBranch {
    //     y: Admittance((1.0 - 1.0 / tap_m) * y),
    //     port: Port2(vector![item.lv_bus, GND]),
    //     v_base,
    // });
    // let re = zbase * (0.001 * item.pfe_kw) / item.sn_mva;
    // let im = zbase / (0.01 * item.i0_percent);
    // let c = parallel as f64 / Complex { re, im };

    // if c.is_nan() {
    //     return v;
    // }
    // let port = Port2(vector![item.hv_bus, GND]);
    // let y = Admittance(0.5 * c / tap_m.powi(2));
    // let shunt = AdmittanceBranch { y, port, v_base };
    // v.push(shunt);
    // let port = Port2(vector![item.lv_bus, GND]);
    // let y = Admittance(0.5 * c);
    // let shunt = AdmittanceBranch { y, port, v_base };
    // v.push(shunt);
    // v
}

fn processing_pq_elems(cmd: &mut Commands, net: &Network) {
    fn process_and_spawn_elements<T>(
        cmd: &mut Commands,
        items: &Option<Vec<T>>,
        to_pqnode_fn: impl Fn(&T) -> PQNode,
    ) {
        if let Some(elements) = items {
            let pq_nodes: Vec<_> = elements
                .iter()
                .enumerate()
                .map(|(idx, x)| {
                    let a = to_pqnode_fn(x);
                    (NodeType::from(a), ElemIdx(idx))
                })
                .collect();
            cmd.spawn_batch(pq_nodes.into_iter());
        }
    }
    process_and_spawn_elements(cmd, &net.load, |item| load_to_pqnode(item));
    process_and_spawn_elements(cmd, &net.shunt, |item| shunt_to_pqnode(item));
    process_and_spawn_elements(cmd, &net.sgen, |item| sgen_to_pqnode(item));
}

fn processing_pv_nodes(cmd: &mut Commands, net: &Network) {
    if let Some(elems) = &net.gen {
        let m: Vec<_> = elems
            .iter()
            .enumerate()
            .map(|(idx, x)| (NodeType::from(gen_to_pvnode(x)), ElemIdx(idx)))
            .collect();
        cmd.spawn_batch(m);
    }
}
/// Converts a load to its equivalent PQ nodes.
fn load_to_pqnode(item: &Load) -> PQNode {
    let s = Complex::new(item.p_mw, item.q_mvar);
    let bus = item.bus;
    PQNode { s, bus }
}
/// Converts a shunt to its equivalent PQ nodes.
fn shunt_to_pqnode(item: &Shunt) -> PQNode {
    let s = Complex::new(item.p_mw, item.q_mvar);
    let bus = item.bus;
    PQNode { s, bus }
}

/// Converts a shunt to its equivalent PQ nodes.
fn sgen_to_pqnode(item: &SGen) -> PQNode {
    let s = Complex::new(-item.p_mw, -item.q_mvar);
    let bus = item.bus;
    PQNode { s, bus }
}

/// Converts a generator to its equivalent PV nodes.
fn gen_to_pvnode(item: &Gen) -> PVNode {
    let p = item.p_mw;
    let v = item.vm_pu;
    let bus = item.bus;
    PVNode { p, v, bus }
}
/// Converts an external grid to its equivalent external grid node.
fn extgrid_to_extnode(cmd: &mut Commands, net: &Network) {
    let item = &net.ext_grid.as_ref().unwrap()[0];
    let bus = item.bus;
    let v = item.vm_pu;
    let phase = item.va_degree.to_radians();

    cmd.spawn((NodeType::from(ExtGridNode { v, phase, bus }), ElemIdx(0)));
}

// Converts an external grid to its equivalent external grid node.
fn process_switch(mut cmd: Commands, net: Res<PPNetwork>) {
    let switch = net.switch.as_ref();
    if let Some(switch) = switch {
        switch.iter().enumerate().for_each(|(idx, x)| {
            cmd.spawn((
                Switch {
                    bus: x.bus,
                    element: x.element,
                    et: x.et.clone(),
                    z_ohm: x.z_ohm,
                },
                ElemIdx(idx),
            ));
        });
    }
}

fn process_switch_state(mut cmd: Commands, q: Query<(Entity, &Switch, &SwitchState)>) {
    q.iter().for_each(|(entity, switch, closed)| {
        let z_ohm = switch.z_ohm;
        let from = switch.bus;
        match switch.et {
            SwitchType::SwitchBusLine => {
                if **closed {
                    //do nothing
                } else {
                    cmd.entity(entity).with_children(|p| {
                        //we add an aux node to break this edge
                        p.spawn(NodeType::from(AuxNode { bus: from }));
                    });
                }
            }
            SwitchType::SwitchBusTransformer => {}
            SwitchType::SwitchTwoBuses => {
                if **closed {
                    cmd.entity(entity).with_children(|p| {
                        //we will merge 2 nodes
                        //  p.spawn();
                    });
                } else {
                    //do nothing
                }
            }
            SwitchType::SwitchBusTransformer3w | SwitchType::Unknown => {}
        }
    });
}



pub fn init_sw(mut world: World) {
    world.run_system_once(process_switch);
}
pub fn init_pf(mut cmd: Commands, value: Res<PPNetwork>) {
    let net = value.as_ref();
    init_node_lookup(&value, &mut cmd);
    line_to_admit(&mut cmd, net);
    processing_pq_elems(&mut cmd, net);
    processing_pv_nodes(&mut cmd, net);
    extgrid_to_extnode(&mut cmd, net);
}

fn init_node_lookup(value: &Res<PPNetwork>, cmd: &mut Commands) {
    let mut d = NodeLookup::default();
    for i in &value.bus {
        let idx = cmd.spawn(PFNode(i.index as usize));
        d.0.insert(i.index, idx.id());
    }
    cmd.insert_resource(d);
}

mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use network::{DataOps, PowerGrid};

    use super::*;
    use std::env;

    #[test]
    fn test_load_csv_zip(){
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        let mut pf_net = PowerGrid::default();
        let world = pf_net.world_mut();
        println!("{}", net.bus.len());
        world.insert_resource(PPNetwork(net));
        world.run_system_once(init_pf);
        let a = world.get_resource::<NodeLookup>();

        println!("{:?}", a.unwrap().0.len());
    }
}
