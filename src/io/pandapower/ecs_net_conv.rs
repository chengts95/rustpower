use crate::basic::ecs::network::DataOps;
use crate::basic::ecs::network::PowerGrid;
use crate::basic::ecs::*;
use crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer;
use std::collections::HashMap;

use crate::prelude::pandapower::*;
use bevy_ecs::prelude::*;
use elements::*;

/// Validate bus labels and all supported bus references before creating entities.
/// The row order defines the dense internal IDs. Element IDs are a separate namespace.
pub fn pandapower_bus_mapping(net: &Network) -> Result<HashMap<i64, i64>, String> {
    let mut ids = HashMap::with_capacity(net.bus.len());
    for (row, bus) in net.bus.iter().enumerate() {
        if ids.insert(bus.index, row as i64).is_some() {
            return Err(format!("duplicate pandapower bus index {}", bus.index));
        }
    }
    let check = |id: i64, field: &str| {
        ids.get(&id)
            .map(|_| ())
            .ok_or_else(|| format!("{field} references missing pandapower bus {id}"))
    };
    for line in net.line.iter().flatten().filter(|e| e.in_service) {
        check(line.from_bus, "line.from_bus")?;
        check(line.to_bus, "line.to_bus")?;
    }
    for trafo in net.trafo.iter().flatten().filter(|e| e.in_service) {
        check(trafo.hv_bus, "trafo.hv_bus")?;
        check(trafo.lv_bus, "trafo.lv_bus")?;
    }
    macro_rules! check_targets {
        ($($field:ident),*) => { $(for element in net.$field.iter().flatten().filter(|e| e.in_service) {
            check(element.bus, concat!(stringify!($field), ".bus"))?;
        })* };
    }
    check_targets!(load, r#gen, ext_grid, shunt, sgen);
    for switch in net.switch.iter().flatten() {
        check(switch.bus, "switch.bus")?;
        if matches!(switch.et, SwitchType::SwitchTwoBuses) {
            check(switch.element, "switch.element (bus-bus)")?;
        }
    }
    let mut references = Vec::new();
    for line in net.line.iter().flatten() {
        references.extend([line.from_bus, line.to_bus]);
    }
    for trafo in net.trafo.iter().flatten() {
        references.extend([trafo.hv_bus, trafo.lv_bus]);
    }
    macro_rules! collect_targets {
        ($($field:ident),*) => { $(for element in net.$field.iter().flatten() { references.push(element.bus); })* };
    }
    collect_targets!(load, r#gen, ext_grid, shunt, sgen);
    let mut missing = -2; // -1 remains reserved for ground.
    for label in references {
        ids.entry(label).or_insert_with(|| {
            let id = missing;
            missing -= 1;
            id
        });
    }
    Ok(ids)
}

pub trait LoadPandapowerNet {
    fn try_load_pandapower_net(&mut self, net: &Network) -> Result<(), String>;

    fn load_pandapower_net(&mut self, net: &Network) {
        self.try_load_pandapower_net(net)
            .expect("invalid pandapower bus references");
    }
}

impl LoadPandapowerNet for PowerGrid {
    fn try_load_pandapower_net(&mut self, net: &Network) -> Result<(), String> {
        self.world_mut().try_load_pandapower_net(net)
    }
}
impl LoadPandapowerNet for World {
    fn try_load_pandapower_net(&mut self, net: &Network) -> Result<(), String> {
        let ids = pandapower_bus_mapping(net)?;
        let world = self;
        let mut buffer = HarvardCommandBuffer::new();

        for bus in &net.bus {
            let mut b: BusBundle = bus.into();
            b.bus_id = BusID(ids[&bus.index]);
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, b);
            buffer.insert(world, e, PPBusID(bus.index));
        }

        for (row, trafo) in net.trafo.iter().flatten().enumerate() {
            let mut t: TransformerBundle = trafo.into();
            t.from_bus = FromBus(ids[&trafo.hv_bus]);
            t.to_bus = ToBus(ids[&trafo.lv_bus]);
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, t);
            buffer.insert(world, e, TransformerID(trafo.index.unwrap_or(row as i64)));
        }

        for (row, line) in net.line.iter().flatten().enumerate() {
            let mut l: LineBundle = line.into();
            l.from = FromBus(ids[&line.from_bus]);
            l.to = ToBus(ids[&line.to_bus]);
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, l);
            buffer.insert(world, e, LineID(line.index.unwrap_or(row as i64)));
        }

        macro_rules! insert_targets {
            ($($field:ident => $bundle:ty),*) => { $(
                for element in net.$field.iter().flatten() {
                    let mut b: $bundle = element.into();
                    b.target_bus = TargetBus(ids[&element.bus]);
                    let e = world.spawn_empty().id();
                    buffer.insert_bundle(world, e, b);
                }
            )* };
        }
        insert_targets!(r#gen => GeneratorBundle, load => LoadBundle,
            ext_grid => ExtGridBundle, shunt => ShuntBundle, sgen => SGenBundle);

        for switch in net.switch.iter().flatten() {
            let mut s: SwitchBundle = switch.into();
            s.switch.bus = ids[&switch.bus];
            if matches!(switch.et, SwitchType::SwitchTwoBuses) {
                s.switch.element = ids[&switch.element];
            }
            // Line/transformer switch.element identifies the device, not a bus.
            let e = world.spawn_empty().id();
            buffer.insert_bundle(world, e, s);
        }
        buffer.apply(world);
        world.insert_resource(PFCommonData {
            wbase: net.f_hz * 2.0 * std::f64::consts::PI,
            f_hz: net.f_hz,
            sbase: net.sn_mva,
        });
        Ok(())
    }
}

pub fn pandapower_init_system(world: &mut World) {
    let net = world.remove_resource::<PPNetwork>();
    if let Some(net) = net {
        world.load_pandapower_net(&net.0);
    }
}
pub fn init_powergrid_from_net(net: &Network, world: &mut World) {
    world.load_pandapower_net(net);
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
    fn import_remaps_all_bus_references_and_preserves_labels() {
        use crate::io::pandapower::{
            Bus, ExtGrid, Gen, Line, Load, SGen, Shunt, Switch, Transformer,
        };
        let mut net = Network::default();
        net.bus = vec![
            Bus {
                index: 42,
                vn_kv: 110.0,
                ..Default::default()
            },
            Bus {
                index: -7,
                vn_kv: 110.0,
                ..Default::default()
            },
        ];
        net.line = Some(vec![Line {
            index: Some(900),
            from_bus: -7,
            to_bus: 42,
            ..Default::default()
        }]);
        net.trafo = Some(vec![Transformer {
            hv_bus: 42,
            lv_bus: -7,
            ..Default::default()
        }]);
        net.load = Some(vec![Load {
            bus: -7,
            ..Default::default()
        }]);
        net.r#gen = Some(vec![Gen {
            bus: -7,
            ..Default::default()
        }]);
        net.ext_grid = Some(vec![ExtGrid {
            bus: -7,
            ..Default::default()
        }]);
        net.shunt = Some(vec![Shunt {
            bus: -7,
            ..Default::default()
        }]);
        net.sgen = Some(vec![SGen {
            bus: -7,
            ..Default::default()
        }]);
        net.switch = Some(vec![
            Switch {
                bus: -7,
                element: 900,
                et: SwitchType::SwitchBusLine,
                ..Default::default()
            },
            Switch {
                bus: -7,
                element: 42,
                et: SwitchType::SwitchTwoBuses,
                ..Default::default()
            },
            Switch {
                bus: 42,
                element: 777,
                et: SwitchType::SwitchBusTransformer,
                ..Default::default()
            },
        ]);
        let mut world = World::new();
        world.try_load_pandapower_net(&net).unwrap();
        let mut buses: Vec<_> = world
            .query::<(&BusID, &PPBusID)>()
            .iter(&world)
            .map(|(id, pp)| (id.0, pp.0))
            .collect();
        buses.sort();
        assert_eq!(buses, vec![(0, 42), (1, -7)]);
        assert_eq!(world.query::<&TargetBus>().iter(&world).count(), 5);
        assert!(
            world
                .query::<&TargetBus>()
                .iter(&world)
                .all(|bus| bus.0 == 1)
        );
        let mut ends: Vec<_> = world
            .query::<(&FromBus, &ToBus)>()
            .iter(&world)
            .map(|(f, t)| (f.0, t.0))
            .collect();
        ends.sort();
        assert_eq!(ends, vec![(0, 1), (1, 0)]);
        let switches: Vec<_> = world
            .query::<&crate::basic::ecs::elements::Switch>()
            .iter(&world)
            .map(|s| (s.bus, s.element))
            .collect();
        assert!(switches.contains(&(1, 900)));
        assert!(switches.contains(&(1, 0)));
        assert!(switches.contains(&(0, 777)));
    }

    #[test]
    fn invalid_bus_references_do_not_partially_import() {
        use crate::io::pandapower::{Bus, Load};
        let mut net = Network::default();
        net.bus = vec![Bus {
            index: 42,
            ..Default::default()
        }];
        net.load = Some(vec![Load {
            bus: 99,
            in_service: true,
            ..Default::default()
        }]);
        let mut world = World::new();
        let initial_entities = world.entities().len();
        let error = world.try_load_pandapower_net(&net).unwrap_err();
        assert!(error.contains("load.bus"));
        assert!(error.contains("99"));
        assert_eq!(world.entities().len(), initial_entities);
        net.load = None;
        net.bus.push(net.bus[0].clone());
        assert!(
            world
                .try_load_pandapower_net(&net)
                .unwrap_err()
                .contains("duplicate")
        );
        assert_eq!(world.entities().len(), initial_entities);
    }

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
    }
}
