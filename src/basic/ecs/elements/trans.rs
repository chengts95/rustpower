use crate::io::pandapower::Transformer;
use crate::prelude::ecs::defer_builder::DeferBundle;
use crate::prelude::ecs::defer_builder::DeferredBundleBuilder;
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;
use nalgebra::Complex;
use nalgebra::Matrix2;
use rustpower_proc_marco::DeferBundle;

use super::{
    bus::SnaptShotRegGroup,
    line::{FromBus, StandardModelType, ToBus},
};
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Port4MatPatch(pub Matrix2<Complex<f64>>);
// #[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
// pub struct Port4 {
//     pub from_port: Vector2<i64>,
//     pub to_port: Vector2<i64>,
// }
/// Represents the electrical and modeling parameters of a transformer.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerDevice {
    /// Dielectric factor (unitless), used to scale impedance. Common default is 1.0.
    pub df: f64,
    /// No-load current as a percentage of rated current (%). Used to model magnetizing branch.
    pub i0_percent: f64,
    /// Iron losses (core losses) in kilowatts (kW).
    pub pfe_kw: f64,
    /// Short-circuit voltage (%), representing the magnitude of leakage impedance.
    pub vk_percent: f64,
    /// Resistive portion of the short-circuit voltage (%), used to separate R/X ratio.
    pub vkr_percent: f64,
    /// Phase shift angle in degrees (°), used for phase-shifting transformers.
    pub shift_degree: f64,
    /// Rated apparent power of the transformer in megavolt-amperes (MVA).
    pub sn_mva: f64,
    /// Rated voltage of the high-voltage side (kV).
    pub vn_hv_kv: f64,
    /// Rated voltage of the low-voltage side (kV).
    pub vn_lv_kv: f64,
    /// Optional upper limit on transformer loading in percentage (%).
    pub max_loading_percent: Option<f64>,
    /// Number of parallel transformers. Used to scale impedance or capacity.
    pub parallel: i32,
    /// Optional tap changer configuration.
    #[serde(flatten)]
    pub tap: Option<TapChanger>,
}

/// Configuration of a tap changer for voltage or phase regulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TapChanger {
    /// Side on which the tap changer is installed, e.g., "hv" or "lv".
    pub side: Option<String>,
    /// Neutral tap position (typically 0.0).
    pub neutral: Option<f64>,
    /// Maximum tap position.
    pub max: Option<f64>,
    /// Minimum tap position.
    pub min: Option<f64>,
    /// Current tap position.
    pub pos: Option<f64>,
    /// Phase shift per tap in degrees (°), for phase shifter modeling.
    pub step_degree: Option<f64>,
    /// Voltage change per tap in percentage (%), for tap ratio modeling.
    pub step_percent: Option<f64>,
    /// Indicates whether this tap changer acts as a phase shifter.
    pub is_phase_shifter: bool,
}

/// ECS bundle representing a transformer entity.
#[derive(DeferBundle, Debug, Clone)]
pub struct TransformerBundle {
    /// Transformer device parameters.
    pub device: TransformerDevice,
    /// The high-voltage side connection (from bus).
    pub from_bus: FromBus,
    /// The low-voltage side connection (to bus).
    pub to_bus: ToBus,
    /// Optional transformer name.
    pub name: Option<Name>,
    /// Optional standard type string (e.g., "25MVA_110/10kV_OFAF").
    pub std_type: Option<StandardModelType>,
}

impl From<&Transformer> for TransformerBundle {
    fn from(t: &Transformer) -> Self {
        Self {
            device: TransformerDevice {
                df: t.df,
                i0_percent: t.i0_percent,
                pfe_kw: t.pfe_kw,
                vk_percent: t.vk_percent,
                vkr_percent: t.vkr_percent,
                shift_degree: t.shift_degree,
                sn_mva: t.sn_mva,
                vn_hv_kv: t.vn_hv_kv,
                vn_lv_kv: t.vn_lv_kv,
                max_loading_percent: t.max_loading_percent,
                parallel: t.parallel,
                tap: Some(TapChanger {
                    side: t.tap_side.clone(),
                    neutral: t.tap_neutral,
                    max: t.tap_max,
                    min: t.tap_min,
                    pos: t.tap_pos,
                    step_degree: t.tap_step_degree,
                    step_percent: t.tap_step_percent,
                    is_phase_shifter: t.tap_phase_shifter,
                }),
            },
            from_bus: FromBus(t.hv_bus as i64),
            to_bus: ToBus(t.lv_bus as i64),
            name: t.name.as_ref().map(|x| Name::new(x.clone())),
            std_type: t.std_type.as_ref().map(|x| StandardModelType(x.clone())),
        }
    }
}
pub struct TransSnapShotReg;
impl SnaptShotRegGroup for TransSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register_named::<TransformerDevice>("trafo");
    }
}
pub mod systems {
    use nalgebra::{Complex, ComplexField};

    use super::*;
    pub fn setup_transformer(mut commands: Commands, q: Query<(Entity, &TransformerDevice)>) {
        q.iter().for_each(|(entity, transformer)| {
            setup_transformer_admittance(&mut commands, entity, transformer);
        });
    }
    fn setup_transformer_admittance(
        commands: &mut Commands,
        parent: Entity,
        dev: &TransformerDevice,
    ) {
        commands.entity(parent).despawn_related::<Children>();

        let tap_m = dev.tap.as_ref().map_or(1.0, |tap| {
            let pos = tap.pos.unwrap_or(0.0);
            let neutral = tap.neutral.unwrap_or(0.0);
            let step = tap.step_percent.unwrap_or(0.0);
            1.0 + (pos - neutral) * 0.01 * step
        });

        let v_base = dev.vn_lv_kv;
        let z_base = v_base * v_base / dev.sn_mva;
        let vk = dev.vk_percent * 0.01;
        let vkr = dev.vkr_percent * 0.01;
        let z = z_base * vk;
        let re = z_base * vkr;
        let im = (z.powi(2) - re.powi(2)).sqrt();
        let y = dev.parallel as f64 / Complex::new(re, im);
        let re_core = z_base * 0.001 * dev.pfe_kw / dev.sn_mva;
        let im_core = z_base / (0.01 * dev.i0_percent);
        let z_m = Complex::new(re_core, im_core);
        let a = tap_m * Complex::from_polar(1.0, dev.shift_degree.to_radians());
        let a = a.recip();
        let t = Matrix2::new(
            a,
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        let mut g = Matrix2::new(y, -y, -y, y);
        let y_m = dev.parallel as f64 / z_m;
        if y_m.is_finite() {
            g[(0, 0)] += 0.5 * y_m;
            g[(1, 1)] += 0.5 * y_m;
        }

        let g = t.conjugate() * g * t; 
        commands.entity(parent).insert(Port4MatPatch(g));
    }
}
