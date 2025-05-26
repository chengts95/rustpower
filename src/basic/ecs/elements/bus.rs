use std::marker::PhantomData;

use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use const_format::concatcp;
use derive_more::derive::{Deref, DerefMut, From, Into};
use nalgebra::Complex;

use crate::{define_snapshot, io::pandapower::Bus};

use super::units::*;

use bevy_ecs::name::Name;
#[derive(Component, Clone, serde::Serialize, serde::Deserialize)]
pub struct VBusPu(pub Complex<f64>);
#[derive(Component, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SBusPu(pub Complex<f64>);
impl Default for VBusPu {
    fn default() -> Self {
        VBusPu(Complex::new(1.0, 0.0))
    }
}

#[derive(Component, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutOfService;
#[derive(Component)]
#[derive(Eq, Ord, PartialEq, PartialOrd)]
#[require(VNominal)]
#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct BusID(pub i64);
#[derive(Component, serde::Serialize, serde::Deserialize)]
pub struct BusType(pub String);

#[derive(Component, Clone, From, Into, Deref, DerefMut, serde::Serialize, serde::Deserialize)]
pub struct VmLimit<T: UnitTrait>(pub Pair<Limit<f64>, T>);
impl Default for VmLimit<PerUnit> {
    fn default() -> Self {
        Self(Pair(Limit { min: 0.9, max: 1.1 }, PhantomData))
    }
}

impl<T: UnitTrait> VmLimit<T> {
    pub fn new(min: f64, max: f64) -> Self {
        VmLimit(Pair(Limit { min, max }, PhantomData::default()))
    }
    pub fn max(&self) -> f64 {
        self.max
    }
    pub fn min(&self) -> f64 {
        self.min
    }
}
#[derive(Component, serde::Serialize, serde::Deserialize)]
pub struct VNominal(pub Pair<f64, KV>);
impl Default for VNominal {
    fn default() -> Self {
        VNominal(Pair(110.0, PhantomData::default()))
    }
}

#[derive(Component, Default, serde::Serialize, serde::Deserialize)]
pub struct Zone(pub i64);
#[derive(Bundle, Default)]
pub struct BusBundle {
    pub name: Name,
    pub bus_id: BusID,
    pub vm_pu: VmLimit<PerUnit>,
    pub vn_kv: VNominal,
    pub zone: Zone,
}

impl From<&Bus> for BusBundle {
    fn from(bus: &Bus) -> Self {
        Self {
            name: Name::new(
                bus.name
                    .as_ref()
                    .map(|x| x.clone())
                    .unwrap_or_else(|| format!("bus_{}", bus.index)),
            ),
            bus_id: BusID(bus.index),
            vm_pu: VmLimit::new(bus.min_vm_pu.unwrap_or(0.9), bus.max_vm_pu.unwrap_or(1.1)),
            vn_kv: VNominal(Pair(bus.vn_kv, PhantomData)),
            zone: Zone(bus.zone.unwrap_or(0)),
        }
    }
}
// Type alias for VmLimit<PerUnit> to use in macro
type VmLimitPerUnit = VmLimit<PerUnit>;

define_snapshot!(VmLimitPerUnit, "Vm", PerUnit);
define_snapshot!(VNominal, "Vn", KV);
pub trait SnaptShotRegGroup {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {}
}

pub struct BusSnapShotReg;
impl SnaptShotRegGroup for BusSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<BusID>();
        reg.register::<Zone>();
        reg.register_with::<Name, NameWrapper>();
        VmLimitPerUnit::register_snap_shot(reg);
        VNominal::register_snap_shot(reg);
    }
}
#[derive(Component, Default, serde::Serialize, serde::Deserialize)]
pub struct NameWrapper(pub String);
impl From<&Name> for NameWrapper {
    fn from(value: &Name) -> Self {
        NameWrapper(value.into())
    }
}
impl Into<Name> for NameWrapper {
    fn into(self) -> Name {
        Name::new(self.0)
    }
}

pub mod systems {

    use crate::basic::ecs::elements::NodeLookup;

    use super::*;
    pub fn init_node_lookup(mut cmd: Commands, bus_ids: Query<(Entity, &BusID)>) {
        let mut node_lookup = NodeLookup::default();
        bus_ids.iter().for_each(|(entity, bus_id)| {
            node_lookup.insert(bus_id.0, entity);
            cmd.entity(entity)
                .insert((SBusPu::default(), VBusPu::default()));
        });
        cmd.insert_resource(node_lookup);
    }
    pub fn update_node_lookup(
        mut lookup: ResMut<NodeLookup>,
        changed: Query<(Entity, &BusID), Changed<BusID>>,
        mut removed: RemovedComponents<BusID>,
    ) {
        // 1️⃣ 清理已移除的
        for entity in removed.read() {
            lookup.remove_entity(entity);
        }

        // 2️⃣ 更新变更/新增的
        for (entity, bus_id) in changed.iter() {
            if lookup.contains_entity(entity) {
                lookup.remove_entity(entity);
            }
            lookup.insert(bus_id.0, entity);
        }
    }
}
#[cfg(test)]
mod tests {
    use std::env;

    use bevy_archive::prelude::{SnapshotRegistry, save_world_manifest};

    use crate::{
        basic::ecs::network::{DataOps, PowerGrid},
        io::pandapower::{Network, load_csv_zip},
    };

    use super::*;
    fn load_net() -> Network {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();
        net
    }
    #[test]
    fn test_snapshot() {
        let net = load_net();
        let buses: Vec<BusBundle> = net.bus.iter().map(|x| x.into()).collect();
        let mut pf_net = PowerGrid::default();
        let mut cmd = pf_net.world_mut().commands();
        cmd.spawn_batch(buses);

        pf_net.world_mut().flush();
        let mut registry = SnapshotRegistry::default();
        BusSnapShotReg::register_snap_shot(&mut registry);
        let world = pf_net.world();
        let a = save_world_manifest(world, &registry).unwrap();
        a.to_file("test_bus.toml", None).unwrap();
    }
}
