
use bevy_archive::prelude::SnapshotRegistry;
use bevy_ecs::prelude::*;

use crate::io::pandapower::{Gen, Transformer};

use super::line::{FromBus, ToBus};

#[derive(Component, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransformerDevice {
    pub df: f64,
    pub i0_percent: f64,
    pub pfe_kw: f64,
    pub vk_percent: f64,
    pub vkr_percent: f64,
    pub shift_degree: f64,
    pub sn_mva: f64,
    pub vn_hv_kv: f64,
    pub vn_lv_kv: f64,
    pub max_loading_percent: Option<f64>,
    pub parallel: i32,
    #[serde(flatten)]
    pub tap: Option<TapChanger>,
}
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TapChanger {
    pub side: Option<String>,
    pub neutral: Option<f64>,
    pub max: Option<f64>,
    pub min: Option<f64>,
    pub pos: Option<f64>,
    pub step_degree: Option<f64>,
    pub step_percent: Option<f64>,
    pub is_phase_shifter: bool,
}

#[derive(Bundle, Debug, Clone)]
pub struct TransformerBundle {
    pub device: TransformerDevice,
    pub from_bus: FromBus,  // 包装 hv_bus
    pub to_bus: ToBus,      // 包装 lv_bus
}


impl From<&Transformer> for TransformerBundle {
    fn from(t: &Transformer) -> Self {
        TransformerBundle {
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
        }
    }
}