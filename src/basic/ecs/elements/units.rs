use std::marker::PhantomData;

use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::component::Component;
#[allow(unused_imports)]
use const_format::concatcp;
use derive_more::derive::Into;
use derive_more::derive::{Deref, DerefMut, From};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
/// Macro for defining a new unit type as a marker component.
///
/// This defines a unit struct implementing [`UnitTrait`] with a specific suffix string.
///
/// # Example
/// ```rust
/// define_unit!(MW, "mw");
/// ```
/// This creates a marker component `MW` representing the "megawatt" unit.
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

/// Macro to define snapshot registry metadata for a given type and unit.
///
/// This adds [`SnapshotInfo`] and [`SnapShotReg`] implementations
/// to allow registering named snapshot fields using both type and unit suffix.
///
/// # Example
/// ```rust
/// define_snapshot!(Voltage, "v", KV);
/// ```
#[macro_export]
macro_rules! define_snapshot {
    ($ty:ident,$short:literal, $suffix:ident) => {
        impl SnapshotInfo for $ty {
            const REGISTERED_NAME: &'static str = concatcp!($short, "_", $suffix::SUFFIX);
        }
        impl SnapShotReg for $ty {}
    };
}

/// A strongly-typed quantity paired with a unit marker.
///
/// Wraps a value of type `T` and carries phantom unit information for
/// compile-time tagging of units like `MW`, `KV`, `PerUnit`, etc.
///
/// This enables stronger semantics and safer handling of different units
/// in ECS systems and snapshot registration.
#[derive(Component, Debug, Default, Serialize, Deserialize, Clone, From, Into, Deref, DerefMut)]
#[serde(transparent)]
pub struct Pair<T, Unit>(
    /// The underlying numeric value.
    pub T,
    /// Phantom marker for the unit type.
    #[deref(ignore)]
    #[deref_mut(ignore)]
    pub PhantomData<Unit>,
);

/// Trait for ECS types that support unit metadata registration into snapshots.
pub trait SnapshotInfo {
    /// The global string name registered in snapshot formats.
    const REGISTERED_NAME: &'static str;

    /// Returns the registered name string.
    fn registered_name(&self) -> &'static str {
        Self::REGISTERED_NAME
    }
}

/// Snapshot registration trait for ECS components.
///
/// Automatically calls `register_named` for types with [`SnapshotInfo`] metadata.
pub trait SnapShotReg {
    /// Registers the snapshot entry into the [`SnapshotRegistry`] with the correct name.
    fn register_snap_shot(reg: &mut SnapshotRegistry)
    where
        Self: SnapshotInfo + Component + Serialize + DeserializeOwned,
    {
        let tname = Self::REGISTERED_NAME;
        reg.register_named::<Self>(tname);
    }
}

/// Trait for unit marker types, providing string suffix information.
///
/// Used to label `Pair<T, Unit>` with meaningful string suffixes like "mw", "pu", etc.
pub trait UnitTrait {
    /// The unit suffix used in snapshot and display, e.g. "kv", "mw", "pu".
    const SUFFIX: &'static str;

    /// Returns the string suffix of this unit.
    fn suffix() -> &'static str {
        Self::SUFFIX
    }
}

// Declare common units used in the simulation.
define_unit!(PerUnit, "pu");
define_unit!(KV, "kv");
define_unit!(MW, "mw");
define_unit!(MVar, "mvar");
define_unit!(KW, "kw");

/// Inherit unit trait for composite `Pair<T, Unit>` types.
impl<T, Unit: UnitTrait> UnitTrait for Pair<T, Unit> {
    fn suffix() -> &'static str {
        Unit::suffix()
    }

    const SUFFIX: &'static str = Unit::SUFFIX;
}

/// A simple structure representing min/max bounds on a value.
///
/// Commonly used for constraining power or reactive output ranges.
#[derive(Debug, Component, Serialize, Deserialize, Clone)]
pub struct Limit<T> {
    /// Minimum value.
    pub min: T,
    /// Maximum value.
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
