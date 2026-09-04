use crate::io::pandapower::Transformer;
use bevy_archive::prelude::SnapshotRegistry;

use bevy_ecs::prelude::*;
use nalgebra::Complex;
use nalgebra::Matrix2;
use rustpower_proc_marco::DeferBundle;

use super::{
    bus::{OutOfService, SnaptShotRegGroup},
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

impl TransformerDevice {
    /// Returns true if the tap changer is installed on the low-voltage side.
    #[inline]
    pub fn is_lv_tap(&self) -> bool {
        self.tap
            .as_ref()
            .and_then(|t| t.side.as_deref())
            .map_or(false, |s| s.eq_ignore_ascii_case("lv") || s == "2")
    }

    /// Computes the effective electrical tap parameters `(ratio, shift_degree, z_scale, tap_factor)`.
    pub fn effective_tap_params(&self) -> (f64, f64, f64, f64) {
        let is_lv = self.is_lv_tap();

        let (pos, neutral, step_p, step_d) = self.tap.as_ref().map_or(
            (0.0, 0.0, 0.0, 0.0),
            |tap| (
                tap.pos.unwrap_or(0.0),
                tap.neutral.unwrap_or(0.0),
                tap.step_percent.unwrap_or(0.0),
                tap.step_degree.unwrap_or(0.0),
            ),
        );
        let n_steps = pos - neutral;
        let tap_factor = 1.0 + n_steps * 0.01 * step_p;

        let (ratio, shift_deg, z_scale) = if is_lv {
            (
                1.0 / tap_factor,
                self.shift_degree - n_steps * step_d,
                tap_factor * tap_factor,
            )
        } else {
            (
                tap_factor,
                self.shift_degree + n_steps * step_d,
                1.0,
            )
        };
        (ratio, shift_deg, z_scale, tap_factor)
    }

    /// Computes the effective phase shift angle in degrees including tap changer adjustments.
    #[inline]
    pub fn effective_shift_degree(&self) -> f64 {
        self.effective_tap_params().1
    }
}
#[cfg(feature = "arrow")]
/// Represents the electrical and modeling parameters of a transformer.
#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerDeviceArrow {
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
    pub tap: Option<TapChanger>,
}
#[cfg(feature = "arrow")]
impl From<TransformerDeviceArrow> for TransformerDevice {
    fn from(value: TransformerDeviceArrow) -> Self {
        TransformerDevice {
            df: value.df,
            i0_percent: value.i0_percent,
            pfe_kw: value.pfe_kw,
            vk_percent: value.vk_percent,
            vkr_percent: value.vkr_percent,
            shift_degree: value.shift_degree,
            sn_mva: value.sn_mva,
            vn_hv_kv: value.vn_hv_kv,
            vn_lv_kv: value.vn_lv_kv,
            max_loading_percent: value.max_loading_percent,
            parallel: value.parallel,
            tap: value.tap,
        }
    }
}
#[cfg(feature = "arrow")]
impl From<&TransformerDevice> for TransformerDeviceArrow {
    fn from(value: &TransformerDevice) -> Self {
        TransformerDeviceArrow {
            df: value.df,
            i0_percent: value.i0_percent,
            pfe_kw: value.pfe_kw,
            vk_percent: value.vk_percent,
            vkr_percent: value.vkr_percent,
            shift_degree: value.shift_degree,
            sn_mva: value.sn_mva,
            vn_hv_kv: value.vn_hv_kv,
            vn_lv_kv: value.vn_lv_kv,
            max_loading_percent: value.max_loading_percent,
            parallel: value.parallel,
            tap: value.tap.clone(),
        }
    }
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
#[derive(Debug, Clone, DeferBundle)]
pub struct TransformerBundle {
    /// tag
    pub tag: crate::basic::ecs::elements::Transformer,
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
    /// Optional marker if this transformer is out of service
    pub out: Option<OutOfService>,
    /// Pre-allocated result data component for zero-allocation power flow post-processing
    pub res: crate::basic::ecs::post_processing::TrafoResultData,
}

impl From<&Transformer> for TransformerBundle {
    fn from(t: &Transformer) -> Self {
        Self {
            tag: crate::basic::ecs::elements::Transformer,
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
            out: (!t.in_service).then_some(OutOfService),
            res: crate::basic::ecs::post_processing::TrafoResultData::default(),
        }
    }
}
pub struct TransSnapShotReg;
impl SnaptShotRegGroup for TransSnapShotReg {
    fn register_snap_shot(reg: &mut SnapshotRegistry) {
        reg.register_named::<TransformerDevice>("trafo");
        #[cfg(feature = "arrow")]
        {
            use bevy_archive::prelude::vec_snapshot_factory::ArrowSnapshotFactory;
            reg.get_factory_mut("trafo").unwrap().arrow = Some(ArrowSnapshotFactory::new_with::<
                TransformerDevice,
                TransformerDeviceArrow,
            >());
        }
    }
}
pub mod trans_systems {
    use nalgebra::{Complex, ComplexField};

    use super::*;
    use crate::basic::ecs::elements::OutOfService;
    pub fn setup_transformer(
        mut commands: Commands,
        q: Query<(Entity, &TransformerDevice), Without<OutOfService>>,
    ) {
        q.iter().for_each(|(entity, transformer)| {
            setup_transformer_admittance(&mut commands, entity, transformer);
        });
    }
    fn setup_transformer_admittance(
        commands: &mut Commands,
        parent: Entity,
        dev: &TransformerDevice,
    ) {
        let (ratio, shift_deg, z_scale, tap_factor) = dev.effective_tap_params();
        let is_lv = dev.is_lv_tap();

        // 1. All branch parameters directly in rated per-unit (O(1) range, no z_base needed)
        let vk = dev.vk_percent * 0.01 * z_scale;
        let vkr = dev.vkr_percent * 0.01 * z_scale;
        let r_pu = vkr;
        let x_pu = (vk * vk - vkr * vkr).max(0.0).sqrt();
        let z_series_pu = Complex::new(r_pu, x_pu) / (dev.parallel as f64);

        let g_m_pu = (dev.pfe_kw * 0.001) / dev.sn_mva;
        let y0_pu = dev.i0_percent * 0.01;
        let b_m_pu = (y0_pu * y0_pu - g_m_pu * g_m_pu).max(0.0).sqrt();
        let y_m_single_pu = Complex::new(g_m_pu, -b_m_pu) / (if is_lv { tap_factor * tap_factor } else { 1.0 });
        let y_m_pu = dev.parallel as f64 * y_m_single_pu;

        // 2. T-model: Kron reduction of the internal star node entirely in per-unit
        // d1 = r1 + j*x1, d2 = r2 + j*x2 in p.u.; y_m in p.u.
        // g = 1 / (y_m + d1^-1 + d2^-1) in p.u.
        // Y_c = [d1^-1 - g*d1^-2, -g*(d1*d2)^-1; -g*(d1*d2)^-1, d2^-1 - g*d2^-2] in p.u.
        let r_ratio = 0.5;
        let x_ratio = 0.5;
        let d1 = Complex::new(z_series_pu.re * r_ratio, z_series_pu.im * x_ratio);
        let d2 = Complex::new(z_series_pu.re * (1.0 - r_ratio), z_series_pu.im * (1.0 - x_ratio));
        let y1 = Complex::new(1.0, 0.0) / d1;
        let y2 = Complex::new(1.0, 0.0) / d2;
        let g = Complex::new(1.0, 0.0) / (y_m_pu + y1 + y2);

        let y11 = y1 - g * y1 * y1;
        let y12 = -g * y1 * y2;
        let y22 = y2 - g * y2 * y2;
        let y_mat_pu = Matrix2::new(y11, y12, y12, y22);

        // 3. Ideal tap transformer scaling
        let a = ratio * Complex::from_polar(1.0, shift_deg.to_radians());
        let a_inv = a.recip();
        let t = Matrix2::new(
            a_inv,
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        let g_mat_pu = t.conjugate() * y_mat_pu * t;

        // 4. Apply z_base^-1 at the very end to store nominal physical value (Siemens S)
        // z_base = vn_lv_kv^2 / sn_mva  =>  y_base = sn_mva / vn_lv_kv^2
        let y_base = dev.sn_mva / (dev.vn_lv_kv * dev.vn_lv_kv);
        let g_physical = g_mat_pu.scale(y_base);

        commands.entity(parent).insert(Port4MatPatch(g_physical));
    }
}
