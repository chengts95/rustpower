use std::marker::PhantomData;

use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::component::Component;
use derive_more::derive::{Deref, DerefMut, From};

use const_format::concatcp;
use derive_more::derive::Into;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
macro_rules! define_unit {
    ($unit:ident, $suffix:literal) => {
        #[derive(Component, Debug, Default, Serialize, Deserialize, Clone)]
        pub struct $unit;

        impl UnitTrait for $unit {
            const SUFFIX: &'static str = $suffix;

            fn suffix() -> &'static str {
                Self::SUFFIX
            }
        }
    };
}
#[macro_export]
macro_rules! define_snapshot {
    ($ty:ident,$short:literal, $suffix:ident) => {
        impl SnapshotInfo for $ty {
            const REGISTERED_NAME: &'static str = concatcp!($short, "_", $suffix::SUFFIX);
        }
        impl SnapShotReg for $ty {}
    };
}
#[derive(Component, Debug,Default, Serialize, Deserialize, Clone, From, Into, Deref, DerefMut)]
#[serde(transparent)]
pub struct Pair<T, Unit>(
    pub T,
    #[deref(ignore)]
    #[deref_mut(ignore)]
    pub PhantomData<Unit>,
);

pub trait SnapshotInfo {
    const REGISTERED_NAME: &'static str;

    fn registered_name(&self) -> &'static str {
        Self::REGISTERED_NAME
    }
}
pub trait SnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry)
    where
        Self: SnapshotInfo + Component + Serialize + DeserializeOwned,
    {
        let tname = Self::REGISTERED_NAME;

        reg.register_named::<Self>(tname);
    }
}

pub trait UnitTrait {
    const SUFFIX: &'static str;

    fn suffix() -> &'static str {
        Self::SUFFIX
    }
}

define_unit!(PerUnit, "pu");
define_unit!(KV, "kv");
define_unit!(MW, "mw");
define_unit!(MVar, "mvar");
define_unit!(KW, "kw");


impl<T, Unit: UnitTrait> UnitTrait for Pair<T, Unit> {
    fn suffix() -> &'static str {
        Unit::suffix()
    }

    const SUFFIX: &'static str = Unit::SUFFIX;
}
#[derive(Debug,Component, Serialize, Deserialize, Clone)]
pub struct Limit<T> {
    pub min: T,
    pub max: T,
}

#[cfg(test)]
mod test {
    use bevy_archive::prelude::{
        load_world_manifest, read_manifest_from_file, save_world_manifest,
    };

    use crate::basic::ecs::network::{DataOps, PowerGrid};

    use super::*;
    #[derive(Component, Serialize, Deserialize, Clone, From, Into, Deref, DerefMut)]
    pub struct VmLimit<T: UnitTrait>(pub Pair<Limit<f64>, T>);
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

    impl SnapshotInfo for VmLimit<PerUnit> {
        const REGISTERED_NAME: &'static str = concatcp!("vm", "_", PerUnit::SUFFIX);
    }
    impl SnapShotReg for VmLimit<PerUnit> {}
    #[test]
    fn test_units() {
        let mut pf_net = PowerGrid::default();

        let e = pf_net
            .world_mut()
            .spawn(VmLimit::from(Pair(
                Limit { min: 0.9, max: 1.1 },
                PhantomData::<PerUnit>,
            )))
            .id();
        let mut reg = SnapshotRegistry::default();

        VmLimit::<PerUnit>::register_snap_shot(&mut reg);
        let reg = reg.clone();
        let a = save_world_manifest(pf_net.world(), &reg);
        a.unwrap().to_file("tt.toml", None).unwrap();
        let manifest = read_manifest_from_file("tt.toml", None).unwrap();
        let _ = load_world_manifest(pf_net.world_mut(), &manifest, &reg).unwrap();
        let vm: &VmLimit<PerUnit> = pf_net.world().entity(e).get().unwrap();
        assert_eq!(vm.min(), 0.9);
        assert_eq!(vm.max(), 1.1);
    }
}
