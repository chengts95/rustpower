//! Resolve labels at topology preparation; numerical systems consume component caches.
use super::*;
use std::collections::HashSet;

fn build_lookup(world: &mut World, excluded: &HashSet<Entity>) -> Result<NodeLookup, String> {
    let mut buses: Vec<_> = world
        .query::<(Entity, &BusID)>()
        .iter(world)
        .filter(|(e, _)| !excluded.contains(e))
        .map(|(e, id)| (id.0, e))
        .collect();
    buses.sort_unstable_by_key(|&(id, _)| id);
    let mut lookup = NodeLookup::default();
    if let Some(old) = world.get_resource::<NodeLookup>() {
        lookup.world_id = old.world_id;
    }
    lookup.revision = world
        .get_resource::<NodeLookup>()
        .map_or(1, |lut| lut.revision + 1);
    for (label, entity) in buses {
        if label == crate::basic::ecs::network::GND {
            return Err("BusID -1 is reserved for archived ground ports".to_string());
        }
        if lookup.contains_id(label) {
            return Err(format!("duplicate BusID {label}"));
        }
        lookup.insert(label, entity);
    }
    Ok(lookup)
}

fn resolve(
    label: i64,
    previous: Option<BusConnection>,
    lookup: &NodeLookup,
) -> Result<BusConnection, String> {
    // An unchanged connection keeps its Entity identity, including its generation.
    let entity = match previous {
        Some(old) if old.label == label => {
            if old.world_id != lookup.world_id {
                return Err(format!("bus {label} reference belongs to another world"));
            }
            if !lookup.contains_entity(old.entity) {
                return Err(format!(
                    "bus {label} references a deleted bus entity {:?}",
                    old.entity
                ));
            }
            old.entity
        }
        _ => lookup
            .get_entity(label)
            .ok_or_else(|| format!("missing bus {label}"))?,
    };
    Ok(BusConnection {
        entity,
        index: lookup.matrix_index(entity).unwrap(),
        label: lookup.get_id(entity).unwrap(),
        world_id: lookup.world_id,
    })
}

fn collect_connections(
    world: &mut World,
    lookup: &NodeLookup,
    excluded: &HashSet<Entity>,
) -> Result<
    Vec<(
        Entity,
        Option<FromBusRef>,
        Option<ToBusRef>,
        Option<TargetBusRef>,
        Option<Port2Ref>,
    )>,
    String,
> {
    let mut pending = Vec::new();
    let derived: HashSet<_> = world
        .query_filtered::<(Entity, &ChildOf), With<EShunt>>()
        .iter(world)
        .filter(|(_, parent)| excluded.contains(&parent.0))
        .map(|(e, _)| e)
        .collect();
    let mut q = world.query_filtered::<(
        Entity,
        Option<&FromBus>,
        Option<&ToBus>,
        Option<&TargetBus>,
        Option<&Port2>,
        Option<&FromBusRef>,
        Option<&ToBusRef>,
        Option<&TargetBusRef>,
        Option<&Port2Ref>,
    ), Or<(
        With<FromBus>,
        With<ToBus>,
        With<TargetBus>,
        With<Port2>,
        With<FromBusRef>,
        With<ToBusRef>,
        With<TargetBusRef>,
        With<Port2Ref>,
    )>>();
    for (e, from, to, target, port, old_f, old_t, old_b, old_p) in q.iter(world) {
        if excluded.contains(&e) || derived.contains(&e) {
            continue;
        }
        let bind = |label, previous, field| {
            resolve(label, previous, lookup).map_err(|err| format!("{field} on {e:?}: {err}"))
        };
        let f = from
            .map(|f| f.0)
            .or_else(|| old_f.map(|r| r.0.label))
            .map(|label| bind(label, old_f.map(|r| r.0), "FromBus").map(FromBusRef))
            .transpose()?;
        let t = to
            .map(|t| t.0)
            .or_else(|| old_t.map(|r| r.0.label))
            .map(|label| bind(label, old_t.map(|r| r.0), "ToBus").map(ToBusRef))
            .transpose()?;
        let b = target
            .map(|b| b.0)
            .or_else(|| old_b.map(|r| r.0.label))
            .map(|label| bind(label, old_b.map(|r| r.0), "TargetBus").map(TargetBusRef))
            .transpose()?;
        let p = if let Some(port) = port {
            let mut ends = [None; 2];
            for i in 0..2 {
                if port.0[i] == crate::basic::ecs::network::GND {
                    continue;
                }
                ends[i] = Some(bind(port.0[i], old_p.and_then(|r| r.0[i]), "Port2")?);
            }
            Some(Port2Ref(ends))
        } else if let Some(old) = old_p {
            let mut ends = [None; 2];
            for (i, previous) in old.0.iter().enumerate() {
                if let Some(previous) = previous {
                    ends[i] = Some(bind(previous.label, Some(*previous), "Port2")?);
                }
            }
            Some(Port2Ref(ends))
        } else {
            None
        };
        if f.is_some() || t.is_some() || b.is_some() || p.is_some() {
            pending.push((e, f, t, b, p));
        }
    }
    Ok(pending)
}

fn resolve_switches(
    world: &mut World,
    lookup: &NodeLookup,
    excluded: &HashSet<Entity>,
) -> Result<
    Vec<(
        Entity,
        SwitchBusRef,
        Option<SwitchOtherBusRef>,
        Option<SwitchLineRef>,
    )>,
    String,
> {
    let has_line_switch = world
        .query::<&Switch>()
        .iter(world)
        .any(|s| s.et == crate::io::pandapower::SwitchType::SwitchBusLine);
    let mut lines = std::collections::HashMap::new();
    if has_line_switch {
        for (e, id) in world
            .query_filtered::<(Entity, &LineID), With<LineParams>>()
            .iter(world)
        {
            if excluded.contains(&e) {
                continue;
            }
            if lines.insert(id.0, e).is_some() {
                return Err(format!("duplicate LineID {}", id.0));
            }
        }
    }
    let mut pending = Vec::new();
    for (e, switch, bus, other, old_line) in world
        .query::<(
            Entity,
            &Switch,
            Option<&SwitchBusRef>,
            Option<&SwitchOtherBusRef>,
            Option<&SwitchLineRef>,
        )>()
        .iter(world)
    {
        if excluded.contains(&e) {
            continue;
        }
        let bus = resolve(switch.bus, bus.map(|b| b.0), lookup)
            .map_err(|err| format!("switch.bus on {e:?}: {err}"))?;
        let other = if matches!(switch.et, crate::io::pandapower::SwitchType::SwitchTwoBuses) {
            Some(SwitchOtherBusRef(
                resolve(switch.element, other.map(|b| b.0), lookup)
                    .map_err(|err| format!("switch.element on {e:?}: {err}"))?,
            ))
        } else {
            None
        };
        let line = if switch.et == crate::io::pandapower::SwitchType::SwitchBusLine {
            let entity = if let Some(old) = old_line.filter(|r| r.label == switch.element) {
                if old.world_id != lookup.world_id {
                    return Err(format!(
                        "switch line reference on {e:?} belongs to another world"
                    ));
                }
                if excluded.contains(&old.entity) || world.get::<LineParams>(old.entity).is_none() {
                    return Err(format!(
                        "switch.element on {e:?} references a deleted line entity {:?}",
                        old.entity
                    ));
                }
                old.entity
            } else {
                *lines.get(&switch.element).ok_or_else(|| {
                    format!(
                        "switch.element on {e:?} references missing line {}",
                        switch.element
                    )
                })?
            };
            let endpoint = |from: bool| -> Result<Entity, String> {
                let label = if from {
                    world.get::<FromBus>(entity).map(|r| r.0)
                } else {
                    world.get::<ToBus>(entity).map(|r| r.0)
                };
                let old = if from {
                    world.get::<FromBusRef>(entity).map(|r| r.0)
                } else {
                    world.get::<ToBusRef>(entity).map(|r| r.0)
                };
                let label = label
                    .or_else(|| old.map(|r| r.label))
                    .ok_or_else(|| format!("line {entity:?} is missing an endpoint"))?;
                Ok(resolve(label, old, lookup)?.entity)
            };
            let from = endpoint(true)?;
            let to = endpoint(false)?;
            let terminal = if bus.entity == from && bus.entity != to {
                LineTerminal::From
            } else if bus.entity == to && bus.entity != from {
                LineTerminal::To
            } else {
                return Err(format!(
                    "switch.bus on {e:?} is not an unambiguous endpoint of line {}",
                    switch.element
                ));
            };
            Some(SwitchLineRef {
                entity,
                terminal,
                label: world
                    .get::<LineID>(entity)
                    .map_or(switch.element, |id| id.0),
                world_id: lookup.world_id,
            })
        } else {
            None
        };
        pending.push((e, SwitchBusRef(bus), other, line));
    }
    Ok(pending)
}

/// Validate a prospective graph without publishing caches or changing its identity.
pub fn validate_topology(world: &mut World, excluded: &HashSet<Entity>) -> Result<(), String> {
    let lookup = build_lookup(world, excluded)?;
    collect_connections(world, &lookup, excluded)?;
    resolve_switches(world, &lookup, excluded)?;
    Ok(())
}

/// Rebuild dense offsets and validate all persistent Entity references atomically.
pub fn prepare_topology(world: &mut World) -> Result<(), String> {
    let excluded = HashSet::new();
    let lookup = build_lookup(world, &excluded)?;
    let pending = collect_connections(world, &lookup, &excluded)?;
    let switches = resolve_switches(world, &lookup, &excluded)?;
    let former_buses: Vec<_> = world
        .query_filtered::<Entity, (With<BusMatrixIndex>, Without<BusID>)>()
        .iter(world)
        .collect();
    for entity in former_buses {
        world.entity_mut(entity).remove::<BusMatrixIndex>();
    }
    for (index, entity) in lookup.dense_iter() {
        world.entity_mut(entity).insert(BusMatrixIndex(index));
    }
    for (e, f, t, b, p) in pending {
        let mut entity = world.entity_mut(e);
        if let Some(f) = f {
            entity.insert((FromBus(f.0.label), FromBusMatrixIndex(f.0.index), f));
        }
        if let Some(t) = t {
            entity.insert((ToBus(t.0.label), ToBusMatrixIndex(t.0.index), t));
        }
        if let Some(b) = b {
            entity.insert((TargetBus(b.0.label), TargetBusMatrixIndex(b.0.index), b));
        }
        if let Some(p) = p {
            let labels =
                p.0.map(|b| b.map_or(crate::basic::ecs::network::GND, |b| b.label));
            entity.insert((
                Port2::new(labels[0], labels[1]),
                Port2MatrixIndex(p.0.map(|b| b.map(|b| b.index))),
                p,
            ));
        }
    }
    for (e, bus, other, line) in switches {
        let mut switch = world.get_mut::<Switch>(e).unwrap();
        switch.bus = bus.0.label;
        if let Some(other) = other {
            switch.element = other.0.label;
        }
        if let Some(line) = line {
            switch.element = line.label;
        }
        let mut entity = world.entity_mut(e);
        entity.insert(bus);
        if let Some(other) = other {
            entity.insert(other);
        } else {
            entity.remove::<SwitchOtherBusRef>();
        }
        if let Some(line) = line {
            entity.insert(line);
        } else {
            entity.remove::<SwitchLineRef>();
        }
    }
    world.insert_resource(lookup);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_labels_cache_dense_indices_and_keep_entity_identity() {
        let mut world = World::new();
        let a = world.spawn(BusID(10)).id();
        let b = world.spawn(BusID(1_000_000_000_000)).id();
        let branch = world.spawn((FromBus(10), ToBus(1_000_000_000_000))).id();
        let load = world.spawn(TargetBus(1_000_000_000_000)).id();
        prepare_topology(&mut world).unwrap();
        assert_eq!(world.get::<FromBusRef>(branch).unwrap().0.entity, a);
        assert_eq!(world.get::<ToBusRef>(branch).unwrap().0.entity, b);
        assert_eq!(world.get::<ToBusMatrixIndex>(branch).unwrap().0, 1);
        assert_eq!(world.get::<TargetBusRef>(load).unwrap().0.entity, b);
        assert_eq!(world.resource::<NodeLookup>().len(), 2);
        world.get_mut::<BusID>(b).unwrap().0 = 3;
        prepare_topology(&mut world).unwrap();
        assert_eq!(world.get::<ToBus>(branch).unwrap().0, 3);
        assert_eq!(world.get::<ToBusRef>(branch).unwrap().0.entity, b);
        assert_eq!(world.get::<ToBusMatrixIndex>(branch).unwrap().0, 0);
        assert_eq!(world.get::<FromBusMatrixIndex>(branch).unwrap().0, 1);
        assert_eq!(world.get::<TargetBus>(load).unwrap().0, 3);
        assert_eq!(world.get::<TargetBusMatrixIndex>(load).unwrap().0, 0);
    }

    #[test]
    fn reused_label_cannot_reconnect_a_stale_reference() {
        let mut world = World::new();
        let bus = world.spawn(BusID(42)).id();
        let load = world.spawn(TargetBus(42)).id();
        prepare_topology(&mut world).unwrap();
        world.despawn(bus);
        let replacement = world.spawn(BusID(42)).id();
        assert_ne!(bus, replacement);
        assert!(
            prepare_topology(&mut world)
                .unwrap_err()
                .contains("deleted bus entity")
        );
        assert_eq!(world.get::<TargetBusRef>(load).unwrap().0.entity, bus);
    }

    #[test]
    fn removed_middle_bus_rebuilds_offsets_and_ground_is_explicit() {
        let mut world = World::new();
        let a = world.spawn(BusID(0)).id();
        let middle = world.spawn(BusID(1)).id();
        let b = world.spawn(BusID(2)).id();
        let branch = world.spawn((FromBus(0), ToBus(2))).id();
        let ground = world
            .spawn(Port2::new(2, crate::basic::ecs::network::GND))
            .id();
        prepare_topology(&mut world).unwrap();
        world.despawn(middle);
        prepare_topology(&mut world).unwrap();
        assert_eq!(world.get::<BusID>(b).unwrap().0, 2);
        assert_eq!(world.get::<ToBusMatrixIndex>(branch).unwrap().0, 1);
        assert_eq!(
            world.get::<Port2MatrixIndex>(ground).unwrap().0,
            [Some(1), None]
        );
        assert_eq!(world.resource::<NodeLookup>().get_by_index(0), Some(a));
        assert_eq!(world.resource::<NodeLookup>().get_by_index(1), Some(b));
    }

    #[test]
    fn native_entity_connections_do_not_require_label_components() {
        let mut world = World::new();
        let bus = world.spawn(BusID(42)).id();
        prepare_topology(&mut world).unwrap();
        let connection = world.resource::<NodeLookup>().connection(bus).unwrap();
        let load = world.spawn(TargetBusRef(connection)).id();
        prepare_topology(&mut world).unwrap();
        assert_eq!(world.get::<TargetBus>(load).unwrap().0, 42);
        assert_eq!(world.get::<TargetBusMatrixIndex>(load).unwrap().0, 0);
        assert_eq!(world.get::<TargetBusRef>(load).unwrap().0.entity, bus);
    }

    #[test]
    fn references_from_another_world_are_rejected_even_when_entity_bits_match() {
        let mut one = World::new();
        one.spawn(BusID(42));
        let load = one.spawn(TargetBus(42)).id();
        prepare_topology(&mut one).unwrap();
        let foreign = *one.get::<TargetBusRef>(load).unwrap();
        let mut two = World::new();
        two.spawn(BusID(42));
        two.spawn((TargetBus(42), foreign));
        assert!(
            prepare_topology(&mut two)
                .unwrap_err()
                .contains("another world")
        );
    }

    #[test]
    fn duplicate_and_missing_labels_are_rejected_without_publishing_caches() {
        let mut world = World::new();
        let first = world.spawn(BusID(42)).id();
        world.spawn(BusID(42));
        assert!(
            prepare_topology(&mut world)
                .unwrap_err()
                .contains("duplicate")
        );
        assert!(world.get::<BusMatrixIndex>(first).is_none());
        assert_eq!(
            std::mem::size_of::<BusMatrixIndex>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<FromBusMatrixIndex>(),
            std::mem::size_of::<usize>()
        );
    }
}
