use bevy_ecs::{component::ComponentId, prelude::*};
use serde::{Deserialize, Serialize};
use std::{any::TypeId, collections::HashMap, fs, path::Path};
type ExportFn = fn(&World, Entity) -> Option<serde_json::Value>;
type ImportFn = fn(&serde_json::Value, &mut World, Entity) -> Result<(), String>;
type CompIdFn = fn(&World) -> Option<ComponentId>;
#[derive(Resource, Default, Debug)]
pub struct SnapshotRegistry {
    pub exporters: HashMap<&'static str, ExportFn>,
    pub importers: HashMap<&'static str, ImportFn>,
    pub type_registry: HashMap<&'static str, TypeId>,
    pub component_id: HashMap<&'static str, CompIdFn>,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotFile {
    #[serde(rename = "entity")]
    pub entities: Vec<EntitySnapshot>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct ComponentSnapshot {
    pub r#type: String,

    pub value: serde_json::Value,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct EntitySnapshot {
    pub id: u64,

    pub components: Vec<ComponentSnapshot>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub entities: Vec<EntitySnapshot>,
}

impl SnapshotRegistry {
    pub fn register<T>(&mut self)
    where
        T: serde::Serialize + serde::de::DeserializeOwned + Component + 'static,
    {
        let name = short_type_name::<T>();
        self.type_registry.insert(name, TypeId::of::<T>());
        self.component_id
            .insert(name, |world| world.component_id::<T>());
        self.exporters.insert(name, |world, entity| {
            // any.downcast_ref::<T>()
            //     .and_then(|t| serde_json::to_value(t).ok())
            world
                .entity(entity)
                .get::<T>()
                .and_then(|t| serde_json::to_value(t).ok())
        });

        self.importers.insert(name, |val, world, entity| {
            let val = serde_json::from_value::<T>(val.clone()).map_err(|e| {
                format!(
                    "Deserialization error for {}:{} ",
                    short_type_name::<T>(),
                    e
                )
            })?;

            world.entity_mut(entity).insert(val);
            Ok(())
        });
    }
    pub fn comp_id_by_name(&self, name: &str, world: &World) -> Option<ComponentId> {
        self.component_id.get(name).and_then(|f| f(world))
    }
    pub fn comp_id<T>(&self, world: &World) -> Option<ComponentId> {
        let name = short_type_name::<T>();
        self.component_id.get(name).and_then(|f| f(world))
    }
}
fn short_type_name<T>() -> &'static str {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap_or("unknown")
}

use serde_json::Value as JsonValue;

pub fn save_world_snapshot(world: &World, reg: &SnapshotRegistry) -> WorldSnapshot {
    let mut entities_snapshot = Vec::new();
    for e in world.iter_entities() {
        let mut es = EntitySnapshot::default();
        es.id = e.id().index() as u64;
        for key in reg.type_registry.keys() {
            if let Some(func) = reg.exporters.get(key) {
                if let Some(value) = func(world, e.id()) {
                    es.components.push(ComponentSnapshot {
                        r#type: key.to_string(),
                        value,
                    });
                }
            }
        }
        entities_snapshot.push(es);
    }
    WorldSnapshot {
        entities: entities_snapshot,
    }
}

pub fn load_world_snapshot(world: &mut World, snapshot: &WorldSnapshot, reg: &SnapshotRegistry) {
    let mut max_id = 0;
    for e in &snapshot.entities {
        max_id = max_id.max(e.id);
    }
    world.entities().reserve_entities((max_id + 1) as u32);
    world.flush();
    for e in &snapshot.entities {
        let entity = Entity::from_raw(e.id as u32);
        for c in &e.components {
            reg.importers
                .get(&c.r#type.as_str())
                .and_then(|f| Some(f(&c.value, world, entity).unwrap()))
                .unwrap()
        }
    }
}

pub fn save_snapshot_to_file<P: AsRef<Path>>(
    snapshot: &WorldSnapshot,
    path: P,
) -> Result<(), std::io::Error> {
    let content = serde_json::to_string_pretty(snapshot).unwrap();
    fs::write(path, content)
}

pub fn load_snapshot_from_file<P: AsRef<Path>>(path: P) -> Result<WorldSnapshot, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("I/O error: {}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("Deserialization error: {}", e))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::ecs::elements::*;
    use bevy_ecs::world::World;
    use nalgebra::vector;
    use serde_json::json;
    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Component)]
    struct TestComponent {
        pub value: i32,
    }
    #[test]
    fn test_snapshot_registry_world() {
        let mut registry = SnapshotRegistry::default();
        let mut world = World::default();

        registry.register::<Port2>();
        let a: Vec<_> = world.entities().reserve_entities(10).collect();
        world.flush();
        a.into_iter().enumerate().for_each(|(i, x)| {
            world.entity_mut(x).insert(Port2(vector![0, i as i64]));
        });

        let _w = save_world_snapshot(&world, &registry);
    }

    #[test]
    fn test_snapshot_registry() {
        let mut registry = SnapshotRegistry::default();
        registry.register::<TestComponent>();

        let component = TestComponent { value: 42 };
        let mut world = World::default();
        let entity = world.spawn(component.clone()).id();

        // Export
        let exported = registry.exporters.get("TestComponent").unwrap()(&world, entity);
        assert!(exported.is_some());
        let exported_value = exported.unwrap();
        assert_eq!(exported_value, json!({"value": 42}));
        assert_eq!(exported_value.get("value").unwrap().as_i64().unwrap(), 42);
        println!("Exported JSON: {}", exported_value);
    }
}
