use bevy_app::App;
use bevy_app::Plugin;
use bevy_app::Startup;
use bevy_ecs::schedule;

use network::GND;
use plugin::PFInitStage;

use crate::basic::ecs::*;

use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use elements::*;
use nalgebra::Complex;
use nalgebra::vector;

use std::f64::consts::PI;

use elements::PQNode;
use elements::PVNode;
use elements::Switch;
use elements::Transformer;

/// Adds the admittance elements of lines to the ECS.
///
/// This function processes each line in the network, calculates admittance
/// values based on the line's parameters, and spawns them into the ECS.
/// It also adds shunt elements (if applicable) and handles the line's resistive
/// and reactive components.
fn line_to_admit(mut cmd: Commands, net: Res<PPNetwork>) {
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

            // Adds shunt elements for the line if there is non-zero conductance or capacitance
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

            // Adds the series admittance component based on the line's resistance and reactance
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

/// Converts a transformer to its equivalent admittance branches and spawns it into the ECS.
///
/// For each transformer in the network, this function calculates its series
/// and shunt admittances, considering tap settings and transformer parameters.
/// It then spawns the transformer and its corresponding branches into the ECS.
fn trafo_to_admit(mut cmd: Commands, net: Res<PPNetwork>) {
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

            // Spawns the short-circuit branch and shunt branches due to tap changers
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

            // Handle core losses and no-load current
            let re = zbase * (0.001 * item.pfe_kw) / item.sn_mva;
            let im = zbase / (0.01 * item.i0_percent);
            let c = parallel as f64 / Complex { re, im };

            if c.is_nan() {
                return;
            }
            let port = Port2(vector![item.hv_bus.into(), GND.into()]);
            let y = Admittance(0.5 * c / tap_m.powi(2));
            let shunt1 = AdmittanceBranch {
                y,
                port,
                v_base: VBase(v_base),
            };

            let port = Port2(vector![item.lv_bus.into(), GND.into()]);
            let y = Admittance(0.5 * c);
            let shunt2 = AdmittanceBranch {
                y,
                port,
                v_base: VBase(v_base),
            };

            entity.with_children(|p| {
                p.spawn(shunt1);
                p.spawn(shunt2);
            });
        });
}

/// Processes PQ elements (loads and generators) in the network and spawns them into the ECS.
fn processing_pq_elems(mut cmd: Commands, net: Res<PPNetwork>) {
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
    process_and_spawn_elements(&mut cmd, &net.load, |item| load_to_pqnode(item));
    process_and_spawn_elements(&mut cmd, &net.sgen, |item| sgen_to_pqnode(item));
}

/// Processes PV nodes (generators) in the network and spawns them into the ECS.
fn processing_pv_nodes(mut cmd: Commands, net: Res<PPNetwork>) {
    if let Some(elems) = &net.r#gen {
        let m: Vec<_> = elems
            .iter()
            .enumerate()
            .map(|(idx, x)| (NodeType::from(gen_to_pvnode(x)), ElemIdx(idx)))
            .collect();
        cmd.spawn_batch(m);
    }
}

/// Converts a load to its equivalent PQ node representation.
fn load_to_pqnode(item: &Load) -> PQNode {
    let s = Complex::new(item.p_mw, item.q_mvar);
    let bus = item.bus;
    PQNode { s, bus }
}

/// Converts a shunt to its equivalent admittance branch.
fn shunt_to_admit(mut cmd: Commands, net: Res<PPNetwork>) {
    fn shunt_internal(item: &Shunt) -> AdmittanceBranch {
        let s = Complex::new(-item.p_mw, -item.q_mvar) * Complex::new(item.step as f64, 0.0);
        let y = s / (item.vn_kv * item.vn_kv);
        AdmittanceBranch {
            y: Admittance(y),
            port: Port2(vector![item.bus as i64, GND.into()]),
            v_base: VBase(item.vn_kv),
        }
    }
    net.shunt
        .as_ref()
        .unwrap_or(&Vec::new())
        .iter()
        .enumerate()
        .for_each(|(idx, item)| {
            let a = shunt_internal(item);
            cmd.spawn((EShunt, a, ElemIdx(idx)));
        });
}

/// Converts a static generator to its equivalent PQ node.
fn sgen_to_pqnode(item: &SGen) -> PQNode {
    let s = Complex::new(-item.p_mw, -item.q_mvar);
    let bus = item.bus;
    PQNode { s, bus }
}

/// Converts a generator to its equivalent PV node.
fn gen_to_pvnode(item: &Gen) -> PVNode {
    let p = item.p_mw;
    let v = item.vm_pu;
    let bus = item.bus;
    PVNode { p, v, bus }
}

/// Converts an external grid to its equivalent external grid node and spawns it into the ECS.
fn extgrid_to_extnode(mut cmd: Commands, net: Res<PPNetwork>) {
    let item = &net.ext_grid.as_ref().unwrap()[0];
    let bus = item.bus;
    let v = item.vm_pu;
    let phase = item.va_degree.to_radians();

    cmd.spawn((NodeType::from(ExtGridNode { v, phase, bus }), ElemIdx(0)));
}

/// Processes the switches in the network and spawns them into the ECS.
pub fn process_switch(mut cmd: Commands, net: Res<PPNetwork>) {
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
                SwitchState(x.closed),
            ));
        });
    }
}

fn inital_setup(mut cmd: Commands, net: Res<PPNetwork>) {
    cmd.insert_resource(PFCommonData {
        wbase: 2.0 * PI * net.f_hz,
        f_hz: net.f_hz,
        sbase: net.sn_mva,
    });
}
/// Initializes the power flow analysis by spawning network elements into the ECS.
pub fn init_pf(world: &mut World) {
    let mut schedule = Schedule::default();
    schedule.set_executor_kind(schedule::ExecutorKind::MultiThreaded);
    schedule.add_systems(
        (
            (inital_setup, init_node_lookup),
            shunt_to_admit,
            line_to_admit,
            trafo_to_admit,
            processing_pq_elems,
            processing_pv_nodes,
            extgrid_to_extnode,
            process_switch,
        )
            .chain(),
    );

    schedule.run(world);
}

pub struct PandaPowerStartupPlugin;

#[derive(Debug, SystemSet, Hash, PartialEq, Eq, Clone, Copy)]
pub struct PandaPowerInit;

impl Plugin for PandaPowerStartupPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (
                (inital_setup, init_node_lookup),
                shunt_to_admit,
                line_to_admit,
                trafo_to_admit,
                processing_pq_elems,
                processing_pv_nodes,
                extgrid_to_extnode,
                (process_switch, process_switch_state).chain(),
            )
                .chain()
                .in_set(PandaPowerInit),
        );
        app.configure_sets(
            Startup,
            PandaPowerInit
                .run_if(resource_exists::<PPNetwork>)
                .before(PFInitStage),
        );
    }
}
/// Initializes the node lookup by mapping bus indices to ECS entities.
fn init_node_lookup(mut cmd: Commands, value: Res<PPNetwork>) {
    let mut d = NodeLookup::default();
    for i in &value.bus {
        let idx = cmd.spawn(PFNode(i.index as usize));
        d.insert(i.index, idx.id());
    }
    cmd.insert_resource(d);
}

#[allow(unused_imports)]
mod tests {
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;
    use network::{DataOps, PowerGrid};

    use crate::basic;

    use super::*;
    use std::env;

    #[test]
    /// Test function for loading and running the power flow system using a CSV zip file.
    fn test_load_csv_zip() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        let mut pf_net = PowerGrid::default();
        let world = pf_net.world_mut();
        println!("{}", net.bus.len());
        world.insert_resource(PPNetwork(net));
        world.run_system_once(init_pf).unwrap();
        let mut a = world.query::<(&Transformer, &Port2)>();

        println!("{:?}", a.iter(world).collect::<Vec<_>>().len());
    }
}
