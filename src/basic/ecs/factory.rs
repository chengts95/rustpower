use std::collections::HashMap;
use bevy_ecs::prelude::*;
use crate::basic::ecs::elements::*;
use crate::basic::ecs::network::{PowerGrid, DataOps};
use std::marker::PhantomData;
use bevy_ecs::name::Name;
use crate::io::pandapower::SwitchType;
use crate::bevy_cmdbuffer::buffer::HarvardCommandBuffer;
use crate::basic::ecs::elements::units::PerUnit;

#[derive(Resource, Default)]
pub struct StdTypeLibrary {
    pub lines: HashMap<String, LineTemplate>,
    pub trafos: HashMap<String, TrafoTemplate>,
}

#[derive(Clone, Debug)]
pub struct LineTemplate {
    pub r_ohm_per_km: f64,
    pub x_ohm_per_km: f64,
    pub c_nf_per_km: f64,
    pub g_us_per_km: f64,
    pub max_i_ka: f64,
}

#[derive(Clone, Debug)]
pub struct TrafoTemplate {
    pub sn_mva: f64,
    pub vn_hv_kv: f64,
    pub vn_lv_kv: f64,
    pub vk_percent: f64,
    pub vkr_percent: f64,
    pub pfe_kw: f64,
    pub i0_percent: f64,
}

impl StdTypeLibrary {
    pub fn add_line_type(&mut self, name: String, r: f64, x: f64, c: f64, g: f64, max_i: f64) {
        self.lines.insert(name, LineTemplate {
            r_ohm_per_km: r,
            x_ohm_per_km: x,
            c_nf_per_km: c,
            g_us_per_km: g,
            max_i_ka: max_i,
        });
    }

    pub fn add_trafo_type(&mut self, name: String, sn_mva: f64, vn_hv: f64, vn_lv: f64, vk: f64, vkr: f64, pfe: f64, i0: f64) {
        self.trafos.insert(name, TrafoTemplate {
            sn_mva,
            vn_hv_kv: vn_hv,
            vn_lv_kv: vn_lv,
            vk_percent: vk,
            vkr_percent: vkr,
            pfe_kw: pfe,
            i0_percent: i0,
        });
    }
}

pub trait GridFactory {
    fn set_base(&mut self, f_hz: f64, sn_mva: f64);
    fn add_std_line_type(&mut self, name: String, r: f64, x: f64, c: f64, g: f64, max_i: f64);
    fn add_std_trafo_type(&mut self, name: String, sn_mva: f64, vn_hv: f64, vn_lv: f64, vk: f64, vkr: f64, pfe: f64, i0: f64);
    
    fn add_bus(&mut self, buffer: &mut HarvardCommandBuffer, id: i64, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> Entity;
    fn add_line(&mut self, buffer: &mut HarvardCommandBuffer, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, params: Option<LineParams>, name: Option<String>) -> Entity;
    fn add_load(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> Entity;
    fn add_gen(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> Entity;
    fn add_ext_grid(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> Entity;
    fn add_trafo(&mut self, buffer: &mut HarvardCommandBuffer, hv_bus: i64, lv_bus: i64, std_type: Option<String>, params: Option<TransformerDevice>, name: Option<String>) -> Entity;
    fn add_shunt(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, vn_kv: f64, step: i32, name: Option<String>) -> Entity;
    fn add_sgen(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> Entity;
    fn add_switch(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, element: i64, et: String, closed: bool, name: Option<String>, z_ohm: f64) -> Entity;
}

impl GridFactory for PowerGrid {
    fn set_base(&mut self, f_hz: f64, sn_mva: f64) {
        self.world_mut().insert_resource(PFCommonData {
            wbase: f_hz * 2.0 * std::f64::consts::PI,
            f_hz,
            sbase: sn_mva,
        });
    }

    fn add_std_line_type(&mut self, name: String, r: f64, x: f64, c: f64, g: f64, max_i: f64) {
        if !self.world().contains_resource::<StdTypeLibrary>() {
            self.world_mut().init_resource::<StdTypeLibrary>();
        }
        self.world_mut().resource_mut::<StdTypeLibrary>().add_line_type(name, r, x, c, g, max_i);
    }

    fn add_std_trafo_type(&mut self, name: String, sn_mva: f64, vn_hv: f64, vn_lv: f64, vk: f64, vkr: f64, pfe: f64, i0: f64) {
        if !self.world().contains_resource::<StdTypeLibrary>() {
            self.world_mut().init_resource::<StdTypeLibrary>();
        }
        self.world_mut().resource_mut::<StdTypeLibrary>().add_trafo_type(name, sn_mva, vn_hv, vn_lv, vk, vkr, pfe, i0);
    }

    fn add_bus(&mut self, buffer: &mut HarvardCommandBuffer, id: i64, vn_kv: f64, name: Option<String>, vm_min: f64, vm_max: f64, zone: i64) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            BusID(id),
            VmLimit::<PerUnit>::new(vm_min, vm_max),
            VNominal(Pair(vn_kv, PhantomData)),
            Zone(zone),
            Name::new(name.unwrap_or_else(|| format!("bus_{}", id))),
        ));
        entity
    }

    fn add_line(&mut self, buffer: &mut HarvardCommandBuffer, from_bus: i64, to_bus: i64, length_km: f64, std_type: Option<String>, params: Option<LineParams>, name: Option<String>) -> Entity {
        let final_params = if let Some(st) = std_type.clone() {
            let lib = self.world().get_resource::<StdTypeLibrary>().expect("StdTypeLibrary not initialized");
            let template = lib.lines.get(&st).expect("Line type not found");
            LineParams {
                r_ohm_per_km: template.r_ohm_per_km,
                x_ohm_per_km: template.x_ohm_per_km,
                c_nf_per_km: template.c_nf_per_km,
                g_us_per_km: template.g_us_per_km,
                length_km,
                df: 1.0,
                parallel: 1,
                max_i_ka: template.max_i_ka,
            }
        } else {
            params.expect("Either std_type or params must be provided")
        };

        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            Line,
            FromBus(from_bus),
            ToBus(to_bus),
            final_params,
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        if let Some(st) = std_type { buffer.insert(world, entity, StandardModelType(st)); }
        entity
    }

    fn add_load(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            TargetBus(bus),
            TargetPMW(-p_mw),
            TargetQMVar(-q_mvar),
            LoadCfg::default(),
            LoadModelType {
                const_i_percent: 0.0,
                const_z_percent: 0.0,
            },
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }

    fn add_gen(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, vm_pu: f64, p_min: f64, p_max: f64, q_min: f64, q_max: f64, name: Option<String>) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            TargetBus(bus),
            TargetPMW(p_mw),
            TargetVmPu(vm_pu),
            PQLim::new(p_min, p_max, q_min, q_max),
            GeneratorCfg::default(),
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }

    fn add_ext_grid(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, vm_pu: f64, va_degree: f64, name: Option<String>) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            TargetBus(bus),
            TargetVmPu(vm_pu),
            TargetVaDeg(va_degree),
            GeneratorCfg::default(),
            PQLim::new(-f64::MAX, f64::MAX, -f64::MAX, f64::MAX),
            Slack,
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }

    fn add_trafo(&mut self, buffer: &mut HarvardCommandBuffer, hv_bus: i64, lv_bus: i64, std_type: Option<String>, params: Option<TransformerDevice>, name: Option<String>) -> Entity {
        let final_dev = if let Some(st) = std_type.clone() {
            let lib = self.world().get_resource::<StdTypeLibrary>().expect("StdTypeLibrary not initialized");
            let template = lib.trafos.get(&st).expect("Trafo type not found");
            TransformerDevice {
                df: 1.0,
                i0_percent: template.i0_percent,
                pfe_kw: template.pfe_kw,
                vk_percent: template.vk_percent,
                vkr_percent: template.vkr_percent,
                shift_degree: 0.0,
                sn_mva: template.sn_mva,
                vn_hv_kv: template.vn_hv_kv,
                vn_lv_kv: template.vn_lv_kv,
                max_loading_percent: None,
                parallel: 1,
                tap: Some(TapChanger {
                    side: Some("hv".to_string()),
                    neutral: Some(0.0),
                    max: Some(10.0),
                    min: Some(-10.0),
                    pos: Some(0.0),
                    step_degree: Some(0.0),
                    step_percent: Some(1.25),
                    is_phase_shifter: false,
                }),
            }
        } else {
            params.expect("Either std_type or params must be provided")
        };

        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            final_dev,
            FromBus(hv_bus),
            ToBus(lv_bus),
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        if let Some(st) = std_type { buffer.insert(world, entity, StandardModelType(st)); }
        entity
    }

    fn add_shunt(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, vn_kv: f64, step: i32, name: Option<String>) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            TargetBus(bus),
            ShuntDevice { p_mw, q_mvar, vn_kv, step, max_step: step },
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }

    fn add_sgen(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, p_mw: f64, q_mvar: f64, name: Option<String>) -> Entity {
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            TargetBus(bus),
            SGenDevice { p_mw, q_mvar, scaling: 1.0, sn_mva: None, gen_type: None, is_current_source: false },
            TargetPMW(p_mw),
            TargetQMVar(q_mvar),
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }

    fn add_switch(&mut self, buffer: &mut HarvardCommandBuffer, bus: i64, element: i64, et: String, closed: bool, name: Option<String>, z_ohm: f64) -> Entity {
        let et = match et.as_str() {
            "l" => SwitchType::SwitchBusLine,
            "t" => SwitchType::SwitchBusTransformer,
            "t3" => SwitchType::SwitchBusTransformer3w,
            "b" => SwitchType::SwitchTwoBuses,
            _ => SwitchType::Unknown,
        };
        let mut world = self.world_mut();
        let entity = world.spawn_empty().id();
        buffer.insert_bundle(world, entity, (
            Switch { bus, element, et, z_ohm },
            SwitchState(closed),
        ));
        if let Some(n) = name { buffer.insert(world, entity, Name::new(n)); }
        entity
    }
}
