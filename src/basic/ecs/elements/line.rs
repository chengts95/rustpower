use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use derive_more::From;
use rustpower_proc_marco::DeferBundle;

use crate::io::pandapower::Line;

use super::bus::{OutOfService, SnaptShotRegGroup};
use crate::prelude::ecs::defer_builder::*;
use bevy_ecs::name::Name;

/// Source bus ID (i64) for a  connection.
///
/// Identifies the originating bus of the line.
/// Must correspond to a valid `BusID` entity in the system.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FromBus(pub i64);

/// Destination bus ID (i64) for a  connection.
///
/// Identifies the target or receiving bus of the line.
/// Must correspond to a valid `BusID` entity in the system.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToBus(pub i64);

/// Physical and electrical parameters of a transmission line.
///
/// All parameters are per-unit-length (per km) unless noted otherwise.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LineParams {
    /// Resistance (Ohm/km)
    pub r_ohm_per_km: f64,
    /// Reactance (Ohm/km)
    pub x_ohm_per_km: f64,
    /// Shunt conductance (μS/km)
    pub g_us_per_km: f64,
    /// Capacitance (nF/km)
    pub c_nf_per_km: f64,
    /// Physical length of the line (km)
    pub length_km: f64,
    /// Dielectric factor (unitless)
    ///
    /// Usually used for correction of charging effect or insulation model.
    pub df: f64,
    /// Number of parallel lines (integer)
    ///
    /// Indicates how many identical lines are in parallel between the buses.
    pub parallel: i32,
}

/// Bundle for initializing a transmission line entity in the ECS world.
///
/// Combines connection endpoints, physical parameters, optional naming,
/// standard specification and operational status.
#[derive(Clone, DeferBundle)]
pub struct LineBundle {
    /// Source bus ID
    pub from: FromBus,
    /// Target bus ID
    pub to: ToBus,
    /// Line electrical parameters
    pub params: LineParams,
    /// Optional human-readable name (e.g. "Line_1")
    pub name: Option<Name>,
    /// Optional standard type name (e.g. "NAYY150SE")
    ///
    /// For referencing predefined line specifications.
    pub std_spec: Option<StandardModelType>,
    /// Optional marker if this line is out of service
    pub out: Option<OutOfService>,
}

/// Standard line model name (e.g. from library or external spec).
///
/// This allows referencing a known cable type or vendor model
/// for reuse of parameter templates. **Currently no use.**
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StandardModelType(pub String);
/// Registers components relevant to line modeling in the snapshot system.
///
/// Ensures that line connections and parameters can be persisted
/// and restored across simulation snapshots or saved ECS states.
pub struct LineSnapShotReg;
impl From<&Line> for LineBundle {
    fn from(line: &Line) -> Self {
        Self {
            from: FromBus(line.from_bus),
            to: ToBus(line.to_bus),
            params: LineParams {
                r_ohm_per_km: line.r_ohm_per_km,
                x_ohm_per_km: line.x_ohm_per_km,
                g_us_per_km: line.g_us_per_km,
                c_nf_per_km: line.c_nf_per_km,
                df: line.df,
                length_km: line.length_km,
                parallel: line.parallel,
            },
            name: line.name.clone().map(Name::new),
            std_spec: line.std_type.clone().map(StandardModelType),
            out: (!line.in_service).then_some(OutOfService),
        }
    }
}

pub struct LineSnapshotReg;

impl SnaptShotRegGroup for LineSnapshotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register::<FromBus>();
        reg.register::<ToBus>();
        reg.register::<LineParams>();
        reg.register::<StandardModelType>();
    }
}

pub mod systems {
    use nalgebra::{Complex, vector};

    use crate::basic::ecs::{elements::*, network::GND};

    use super::*;
    pub fn setup_line_systems(
        mut commands: Commands,
        q: Query<(Entity, &LineParams, &FromBus, &ToBus)>,
        buses: Query<&VNominal>,
        lut: Res<NodeLookup>,
        common: Res<PFCommonData>,
    ) {
        for (entity, params, from, to) in &q {
            let length = params.length_km;
            let parallel = params.parallel as f64;
            let wbase = common.wbase;

            let b = wbase * 1e-9 * params.c_nf_per_km * length * parallel;
            let g = 1e-6 * params.g_us_per_km * length * parallel;
            let y_shunt = 0.5 * Complex::new(g, b);

            let rl = params.r_ohm_per_km * length * parallel;
            let xl = params.x_ohm_per_km * length * parallel;
            let y_series = 1.0 / Complex::new(rl, xl);
            let vbase = lut.get_entity(from.0).unwrap();
            let vbase = buses.get(vbase).unwrap().0.0;
            // Shunt: from and to → GND
            commands.entity(entity).with_children(|p| {
                if g != 0.0 || b != 0.0 {
                    p.spawn(AdmittanceBranch {
                        y: Admittance(y_shunt),
                        port: Port2(vector![from.0, GND.into()]),
                        v_base: VBase(vbase), // 1.0 per unit unless otherwise specified
                    });
                    p.spawn(AdmittanceBranch {
                        y: Admittance(y_shunt),
                        port: Port2(vector![to.0, GND.into()]),
                        v_base: VBase(vbase),
                    });
                }

                // Series element between from and to
                p.spawn(AdmittanceBranch {
                    y: Admittance(y_series),
                    port: Port2(vector![from.0, to.0]),
                    v_base: VBase(vbase),
                });
            });
        }
    }
}
