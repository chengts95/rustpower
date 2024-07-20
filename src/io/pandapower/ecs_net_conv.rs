use bevy_ecs::system::RunSystemOnce;

use network::PowerGrid;
use network::GND;

use crate::basic;
use crate::basic::new_ecs::*;
use crate::basic::system::PFNetwork;
use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use bevy_hierarchy::prelude::*;
use elements::*;
use nalgebra::vector;
use nalgebra::Complex;

use std::f64::consts::PI;
use std::fmt;

use elements::PQNode;
use elements::PVNode;
use elements::Switch;
use elements::Transformer as Transformer;

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
fn trafo_to_admit(cmd: &mut Commands, net: &Network) {
    net.trafo
        .as_ref()
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
        .for_each(|(idx, item)| {
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
                Port2(vector![item.hv_bus as i64, item.lv_bus as i64]),
            ));

            let sc = AdmittanceBranch {
                y: Admittance(y / tap_m),
                port,
                v_base: VBase(v_base),
            };

            entity.with_children(|p| {
                p.spawn(sc);
                p.spawn(AdmittanceBranch {
                    y: Admittance((1.0 - tap_m) * y / tap_m.powi(2)),
                    port: Port2(vector![item.hv_bus.into(), GND.into()]),
                    v_base: VBase(v_base),
                });
                p.spawn(AdmittanceBranch {
                    y: Admittance((1.0 - 1.0 / tap_m) * y),
                    port: Port2(vector![item.lv_bus.into(), GND.into()]),
                    v_base: VBase(v_base),
                });
            });
            let re = zbase * (0.001 * item.pfe_kw) / item.sn_mva;
            let im = zbase / (0.01 * item.i0_percent);
            let c = parallel as f64 / Complex { re, im };

            if c.is_nan() {
                return;
            }
            let port = Port2(vector![item.hv_bus.into(), GND.into()]);
            let y = Admittance(c / tap_m.powi(2));
            let shunt1 = AdmittanceBranch {
                y,
                port,
                v_base: VBase(v_base),
            };

            // let port = Port2(vector![item.lv_bus.into(), GND.into()]);
            // let y = Admittance(0.5 * c);
            // let shunt2 = AdmittanceBranch { y, port, v_base:VBase(v_base)};
            entity.with_children(|p| {
                p.spawn(shunt1);
                //  p.spawn(shunt2);
            });
        });
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
#[allow(dead_code)]
fn process_switch_state(mut cmd: Commands, q: Query<(Entity, &Switch, &SwitchState)>) {
    q.iter().for_each(|(entity, switch, closed)| {
        let _z_ohm = switch.z_ohm;
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
                        let to = switch.element;
                        p.spawn(MergeNode(from as usize, to as usize));
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
pub fn init_pf(world: &mut World) {
    world.resource_scope::<PPNetwork, _>(|world, net: Mut<PPNetwork>| {
        let net = &net;
        world.insert_resource(PFCommonData {
            wbase: 2.0 * PI * net.f_hz,
            sbase: net.sn_mva,
        });

        init_node_lookup(net, world);
        let mut cmd = world.commands();
        line_to_admit(&mut cmd, net);
        trafo_to_admit(&mut cmd, net);
        processing_pq_elems(&mut cmd, net);
        processing_pv_nodes(&mut cmd, net);
        extgrid_to_extnode(&mut cmd, net);
    });
}

fn init_node_lookup(value: &PPNetwork, world: &mut World) {
    let mut d = NodeLookup::default();
    for i in &value.bus {
        let idx = world.spawn(PFNode(i.index as usize));
        d.0.insert(i.index, idx.id());
    }
    world.insert_resource(d);
}

#[derive(Debug)]
pub enum ParseError {
    InvalidData,
    ConversionError(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidData => write!(f, "Invalid input data"),
            ParseError::ConversionError(msg) => write!(f, "Conversion failed: {}", msg),
        }
    }
}
impl std::error::Error for ParseError {}

impl TryFrom<&mut PowerGrid> for PFNetwork {
    type Error = ParseError;

    fn try_from(value: &mut PowerGrid) -> Result<Self, Self::Error> {
        use crate::basic::new_ecs::network::DataOps;
        let world = value.world_mut();
        if world.get_resource::<PPNetwork>().is_none() {
            return Err(ParseError::ConversionError(
                "Net resource not found".to_string(),
            ));
        }
        world.run_system_once(init_pf);
        let net = &world.get_resource::<PPNetwork>().unwrap();
        let buses = net.bus.clone();
        let v_base = net.bus[0].vn_kv;
        let s_base = net.sn_mva;
        let pq_loads = extract_node(world, |x| {
            if let NodeType::PQ(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let pv_nodes = extract_node(world, |x| {
            if let NodeType::PV(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let binding = extract_node(world, |x| {
            if let NodeType::EXT(v) = x {
                Some(v.clone())
            } else {
                None
            }
        });
        let ext = binding
            .get(0)
            .ok_or_else(|| ParseError::ConversionError("No external node found".to_string()))?;
        let ext = ext.clone();
        let y_br: Vec<_> = world
            .query::<(&Admittance, &Port2, &VBase)>()
            .iter(world)
            .map(|(a, p, vb)| basic::system::AdmittanceBranch {
                y: basic::system::Admittance(a.0),
                port: basic::system::Port2(p.0.cast()),
                v_base: vb.0,
            })
            .collect();

        let net = PFNetwork {
            v_base,
            s_base,
            buses,
            pq_loads,
            pv_nodes,
            ext,
            y_br,
        };
        Ok(net)
    }
}
fn extract_node<T, F>(world: &mut World, extractor: F) -> Vec<T>
where
    F: Fn(&NodeType) -> Option<T>,
{
    world
        .query::<&NodeType>()
        .iter(world)
        .filter_map(extractor)
        .collect()
}
#[allow(unused_imports)]
mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;
    use network::{DataOps, PowerGrid};

    use crate::basic::{self, system::RunPF};

    use super::*;
    use std::env;

    #[test]
    fn test_load_csv_zip() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        let mut pf_net = PowerGrid::default();
        let world = pf_net.world_mut();
        println!("{}", net.bus.len());
        world.insert_resource(PPNetwork(net));
        world.run_system_once(init_pf);
        let mut a = world.query::<(&Transformer, &Port2)>();

        println!("{:?}", a.iter(world).collect::<Vec<_>>().len());
    }

    #[test]
    fn test_to_pf_net() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        let net = PFNetwork::try_from(&mut pf_net).unwrap();
        let v_init = net.create_v_init();
        let tol = Some(1e-8);
        let max_it = Some(10);
        let (v, iter) = net.run_pf(v_init.clone(), max_it, tol);
        println!("Vm,\t angle");
        for (x, i) in v.iter().enumerate() {
            println!("{} {:.5}, {:.5}", x, i.modulus(), i.argument().to_degrees());
        }
        println!("converged within {} iterations", iter);
    }
}
