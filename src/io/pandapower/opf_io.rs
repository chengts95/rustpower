use std::collections::HashMap;
use std::io::Read;

use bevy_ecs::prelude::World;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};

/// One row of pandapower's `poly_cost` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolyCostRow {
    pub element: i64,
    pub et: String,
    pub cp0_eur: f64,
    pub cp1_eur_per_mw: f64,
    pub cp2_eur_per_mw2: f64,
    pub cq0_eur: Option<f64>,
    pub cq1_eur_per_mvar: Option<f64>,
    pub cq2_eur_per_mvar2: Option<f64>,
}

/// Indexed OPF cost configuration: lookup by (et, element_index).
///
/// `element` matches the positional index of the generator in the corresponding
/// pandapower Vec (gen[0] → element=0, ext_grid[0] → element=0).
pub struct OPFCfg {
    costs: HashMap<(String, i64), PolyCostRow>,
}

impl OPFCfg {
    pub fn from_rows(rows: Vec<PolyCostRow>) -> Self {
        let costs = rows
            .into_iter()
            .map(|r| ((r.et.clone(), r.element), r))
            .collect();
        Self { costs }
    }

    pub fn get(&self, et: &str, element: i64) -> Option<&PolyCostRow> {
        self.costs.get(&(et.to_string(), element))
    }

    pub fn len(&self) -> usize {
        self.costs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.costs.is_empty()
    }
}

// ── loaders ───────────────────────────────────────────────────────────────────

pub fn opf_cfg_from_csv_str(s: &str) -> Option<OPFCfg> {
    let mut rdr = ReaderBuilder::new().from_reader(s.as_bytes());
    let mut rows: Vec<PolyCostRow> = Vec::new();
    let headers = rdr.headers().ok()?.to_owned();
    for rec in rdr.records() {
        let rec = rec.ok()?;
        let row: PolyCostRow = rec.deserialize(Some(&headers)).ok()?;
        rows.push(row);
    }
    if rows.is_empty() { None } else { Some(OPFCfg::from_rows(rows)) }
}

pub fn load_opf_cfg_csv(path: &str) -> Option<OPFCfg> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut s = String::new();
    file.read_to_string(&mut s).ok()?;
    opf_cfg_from_csv_str(&s)
}

pub fn load_opf_cfg_zip(path: &str) -> Option<OPFCfg> {
    let f = std::fs::File::open(path).ok()?;
    let mut zip = zip::ZipArchive::new(f).ok()?;
    let mut entry = zip.by_name("poly_cost.csv").ok()?;
    let mut s = String::new();
    entry.read_to_string(&mut s).ok()?;
    opf_cfg_from_csv_str(&s)
}

/// Load from the flat JSON format used by embedded test cases (e.g. case_ieee39).
pub fn load_opf_cfg_json_str(json_str: &str) -> Option<OPFCfg> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let arr = v.get("poly_cost")?.as_array()?;
    let rows: Vec<PolyCostRow> = arr
        .iter()
        .filter_map(|x| serde_json::from_value(x.clone()).ok())
        .collect();
    if rows.is_empty() { None } else { Some(OPFCfg::from_rows(rows)) }
}

// ── ECS patch ─────────────────────────────────────────────────────────────────

/// Patches `GenCost` components onto generator and ext_grid entities.
///
/// Requires `PandapowerEntityMap` resource (inserted by `load_pandapower_net`).
/// Entities with no matching `OPFCfg` entry are left unchanged.
pub fn patch_gen_cost(world: &mut World, opf_cfg: &OPFCfg) {
    use crate::basic::ecs::elements::{GenCost, PandapowerEntityMap};

    let (gen_ids, ext_grid_ids) = {
        let map = world
            .get_resource::<PandapowerEntityMap>()
            .expect("PandapowerEntityMap missing; call load_pandapower_net first");
        (map.gen_entities.clone(), map.ext_grid_entities.clone())
    };

    for (idx, entity) in gen_ids.iter().enumerate() {
        if let Some(row) = opf_cfg.get("gen", idx as i64) {
            world.entity_mut(*entity).insert(GenCost::from(row));
        }
    }
    for (idx, entity) in ext_grid_ids.iter().enumerate() {
        if let Some(row) = opf_cfg.get("ext_grid", idx as i64) {
            world.entity_mut(*entity).insert(GenCost::from(row));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_csv_str_roundtrip() {
        let csv = "element,et,cp0_eur,cp1_eur_per_mw,cp2_eur_per_mw2,cq0_eur,cq1_eur_per_mvar,cq2_eur_per_mvar2\n\
                   0,gen,0.2,0.3,0.01,0.0,0.0,0.0\n\
                   0,ext_grid,0.2,0.3,0.01,0.0,0.0,0.0\n\
                   1,gen,0.5,1.0,0.02,0.0,0.0,0.0\n";
        let cfg = opf_cfg_from_csv_str(csv).expect("parse failed");
        assert_eq!(cfg.len(), 3);

        let row = cfg.get("gen", 0).unwrap();
        assert_eq!(row.cp1_eur_per_mw, 0.3);
        assert_eq!(row.cp2_eur_per_mw2, 0.01);
        assert_eq!(row.cp0_eur, 0.2);

        let row2 = cfg.get("ext_grid", 0).unwrap();
        assert_eq!(row2.et, "ext_grid");

        let row3 = cfg.get("gen", 1).unwrap();
        assert_eq!(row3.cp2_eur_per_mw2, 0.02);

        assert!(cfg.get("gen", 99).is_none());
    }

    #[test]
    fn test_load_csv_file_ieee39() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE39/poly_cost.csv", dir);
        let cfg = load_opf_cfg_csv(&path).expect("failed to load IEEE39 poly_cost.csv");
        // 9 gens + 1 ext_grid
        assert_eq!(cfg.len(), 10);
        let row = cfg.get("gen", 0).unwrap();
        assert_eq!(row.cp1_eur_per_mw, 0.3);
        assert_eq!(row.cp2_eur_per_mw2, 0.01);
        assert!(cfg.get("ext_grid", 0).is_some());
        // all gens 0..8 should be present
        for i in 0i64..9 {
            assert!(cfg.get("gen", i).is_some(), "gen[{}] missing", i);
        }
    }

    #[test]
    fn test_load_zip_ieee118() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/cases/IEEE118/data.zip", dir);
        let cfg = load_opf_cfg_zip(&path).expect("poly_cost.csv not found in IEEE118 zip");
        // 53 gens + 1 ext_grid = 54 entries
        assert_eq!(cfg.len(), 54);
        // spot-check gen[0]
        let row = cfg.get("gen", 0).unwrap();
        assert_eq!(row.cp1_eur_per_mw, 40.0);
        // ext_grid[0]
        assert!(cfg.get("ext_grid", 0).is_some());
    }

    #[test]
    fn test_load_json_str_ieee39() {
        let cfg = load_opf_cfg_json_str(crate::testcases::case_ieee39::IEEE_39)
            .expect("poly_cost missing from case39 JSON");
        assert_eq!(cfg.len(), 10);
        let row = cfg.get("gen", 8).unwrap();
        assert_eq!(row.cp2_eur_per_mw2, 0.01);
    }

    /// Verify that Network deserialization silently ignores poly_cost (unknown field).
    #[test]
    fn test_network_ignores_poly_cost() {
        let net: crate::io::pandapower::Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39)
                .expect("Network deserialize failed even with unknown poly_cost field");
        assert_eq!(net.bus.len(), 39);
    }

    /// Full ECS round-trip: load network → patch GenCost → query components.
    #[test]
    fn test_patch_gen_cost_ecs_ieee39() {
        use bevy_ecs::prelude::*;
        use crate::basic::ecs::elements::{GenCost, PandapowerEntityMap};
        use crate::io::pandapower::ecs_net_conv::LoadPandapowerNet;

        let net: crate::io::pandapower::Network =
            serde_json::from_str(crate::testcases::case_ieee39::IEEE_39).unwrap();
        let opf_cfg = load_opf_cfg_json_str(crate::testcases::case_ieee39::IEEE_39).unwrap();

        let mut world = World::new();
        world.load_pandapower_net(&net);

        // Before patch: no GenCost components
        let count_before = world.query::<&GenCost>().iter(&world).count();
        assert_eq!(count_before, 0, "no GenCost before patch");

        patch_gen_cost(&mut world, &opf_cfg);

        // After patch: 9 gens + 1 ext_grid = 10 entities with GenCost
        let costs: Vec<&GenCost> = world.query::<&GenCost>().iter(&world).collect();
        assert_eq!(costs.len(), 10, "10 GenCost components after patch");

        // Spot-check: all entries should have cp1=0.3, cp2=0.01
        for c in &costs {
            assert_eq!(c.cp1_eur_per_mw, 0.3);
            assert_eq!(c.cp2_eur_per_mw2, 0.01);
        }

        // Entity map should have 9 gens and 1 ext_grid
        let map = world.resource::<PandapowerEntityMap>();
        assert_eq!(map.gen_entities.len(), 9);
        assert_eq!(map.ext_grid_entities.len(), 1);
    }
}
