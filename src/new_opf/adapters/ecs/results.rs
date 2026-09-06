use crate::new_opf::model::NewOPFData;
use crate::new_opf::solution::PipsResult;
use bevy_ecs::prelude::{Entity, World};

impl NewOPFData {
    pub fn write_results(&self, world: &mut World, result: &PipsResult) {
        use crate::basic::ecs::elements::*;
        use crate::new_opf::adapters::ecs::components::*;

        let nb = self.nb;
        let ng = self.ng;

        // Clone mappings to avoid borrow conflicts
        let (gen_entities, ext_grid_entities) = {
            let map = world
                .get_resource::<PandapowerEntityMap>()
                .expect("PandapowerEntityMap missing");
            (map.gen_entities.clone(), map.ext_grid_entities.clone())
        };

        let bus_entities: Vec<Option<Entity>> = {
            let node_lookup = world
                .get_resource::<NodeLookup>()
                .expect("NodeLookup missing");
            (0..nb).map(|i| node_lookup.get_entity(i as i64)).collect()
        };

        // 1. Write Bus Results (Vm, Va, Lambda)
        for i in 0..nb {
            if let Some(entity) = bus_entities[i] {
                let va = result.x[i];
                let vm = result.x[nb + i];
                let lp = result.lam_eq[i];
                let lq = result.lam_eq[nb + i];

                world.entity_mut(entity).insert((
                    OpfResultVa(va),
                    OpfResultVm(vm),
                    LambdaBus { p: lp, q: lq },
                ));
            }
        }

        // 2. Write Generator Results (Pg, Qg)
        let pg_off = 2 * nb;
        let qg_off = 2 * nb + ng;
        let num_ext = ext_grid_entities.len();

        for g in 0..ng {
            let entity = if g < num_ext {
                ext_grid_entities[g]
            } else {
                gen_entities[g - num_ext]
            };

            let pg = result.x[pg_off + g];
            let qg = result.x[qg_off + g];

            world
                .entity_mut(entity)
                .insert((OpfResultPg(pg), OpfResultQg(qg)));
        }
    }
}
