//! One-time bridge analysis of the baseline's undirected electrical multigraph.
//! Bus indices follow the permuted Ybus; edge indices follow BranchCatalog.
use super::{BranchCatalog, SharedBaseline};
use std::ops::Range;

const NONE: usize = usize::MAX;

/// Both sides of a bridge outage, expressed as ranges in DFS order.
/// The other side is `component` minus `subtree` (at most two slices).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeCut {
    pub subtree: Range<usize>,
    pub component: Range<usize>,
    pub subtree_slacks: usize,
    pub other_slacks: usize,
}
impl BridgeCut {
    pub fn subtree_bus_count(&self) -> usize {
        self.subtree.len()
    }
    pub fn other_bus_count(&self) -> usize {
        self.component.len() - self.subtree.len()
    }
    /// Test a bus's DFS number, obtained from N1Topology::dfs_number.
    pub fn contains_in_subtree(&self, dfs_number: usize) -> bool {
        self.subtree.contains(&dfs_number)
    }
    /// Number of buses on the newly separated sides that have no fixed Slack.
    pub fn slackless_bus_count(&self) -> usize {
        usize::from(self.subtree_slacks == 0) * self.subtree_bus_count()
            + usize::from(self.other_slacks == 0) * self.other_bus_count()
    }
}

#[derive(Clone, Copy)]
struct DfsNode {
    number: usize,
    subtree_size: usize,
    subtree_slacks: usize,
    component_root: usize,
}

/// Shared O(V+E) metadata. Outage queries allocate nothing and take O(1).
/// Parallel devices remain separate edges, even when Ybus combines their entries.
pub struct N1Topology {
    nodes: Vec<DfsNode>,
    dfs_buses: Vec<usize>,
    bridge_children: Vec<usize>,
}
impl N1Topology {
    /// Build once from the same branch catalog and solver bus order as Ybus.
    /// Inactive devices, merged self-loops and zero mutual-admittance devices
    /// contribute no connectivity. A shunt contributes only to the diagonal.
    pub fn from_baseline(base: &SharedBaseline, catalog: &BranchCatalog) -> Result<Self, String> {
        let n = base.y_bus().ncols();
        let [_, y_tf, y_ft, _] = catalog.admittances();
        Self::from_edges(
            n,
            base.npq() + base.npv()..n,
            (0..catalog.len()).map(|i| {
                (
                    catalog.from_buses()[i],
                    catalog.to_buses()[i],
                    catalog.in_service()[i]
                        && (y_tf[i] != num_complex::Complex64::default()
                            || y_ft[i] != num_complex::Complex64::default()),
                )
            }),
        )
    }

    fn from_edges(
        n: usize,
        slacks: impl IntoIterator<Item = usize>,
        edges: impl IntoIterator<Item = (usize, usize, bool)>,
    ) -> Result<Self, String> {
        let mut nodes = vec![
            DfsNode {
                number: NONE,
                subtree_size: 1,
                subtree_slacks: 0,
                component_root: NONE,
            };
            n
        ];
        for slack in slacks {
            let node = nodes
                .get_mut(slack)
                .ok_or("topology Slack bus is out of bounds")?;
            node.subtree_slacks = 1;
        }
        let edges: Vec<_> = edges.into_iter().collect();
        // Temporary CSR adjacency; it is released after the single DFS forest.
        let mut offsets = vec![0; n + 1];
        for &(from, to, active) in &edges {
            if from >= n || to >= n {
                return Err("topology branch endpoint is out of bounds".into());
            }
            if active && from != to {
                offsets[from + 1] += 1;
                offsets[to + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut cursor = offsets[..n].to_vec();
        let mut adjacency = vec![(0, 0); offsets[n]];
        for (edge, &(from, to, active)) in edges.iter().enumerate() {
            if active && from != to {
                adjacency[cursor[from]] = (to, edge);
                cursor[from] += 1;
                adjacency[cursor[to]] = (from, edge);
                cursor[to] += 1;
            }
        }
        let mut bridge_children = vec![NONE; edges.len()];
        let mut low = vec![0; n];
        let mut dfs_buses = Vec::with_capacity(n);
        // Explicit stack also handles long radial feeders without recursion.
        struct Frame {
            bus: usize,
            parent_edge: usize,
            next: usize,
        }
        let mut stack = Vec::<Frame>::new();
        for root in 0..n {
            if nodes[root].number != NONE {
                continue;
            }
            nodes[root].number = dfs_buses.len();
            nodes[root].component_root = root;
            low[root] = dfs_buses.len();
            dfs_buses.push(root);
            stack.push(Frame {
                bus: root,
                parent_edge: NONE,
                next: offsets[root],
            });
            while let Some(frame) = stack.last_mut() {
                let bus = frame.bus;
                if frame.next < offsets[bus + 1] {
                    let (neighbor, edge) = adjacency[frame.next];
                    frame.next += 1;
                    // Skip only the actual parent edge; a parallel edge is a back edge.
                    if edge == frame.parent_edge {
                        continue;
                    }
                    if nodes[neighbor].number == NONE {
                        nodes[neighbor].number = dfs_buses.len();
                        nodes[neighbor].component_root = root;
                        low[neighbor] = dfs_buses.len();
                        dfs_buses.push(neighbor);
                        stack.push(Frame {
                            bus: neighbor,
                            parent_edge: edge,
                            next: offsets[neighbor],
                        });
                    } else {
                        low[bus] = low[bus].min(nodes[neighbor].number);
                    }
                } else {
                    let finished = stack.pop().unwrap();
                    if let Some(parent) = stack.last() {
                        let p = parent.bus;
                        if low[bus] > nodes[p].number {
                            bridge_children[finished.parent_edge] = bus;
                        }
                        low[p] = low[p].min(low[bus]);
                        nodes[p].subtree_size += nodes[bus].subtree_size;
                        nodes[p].subtree_slacks += nodes[bus].subtree_slacks;
                    }
                }
            }
        }
        Ok(Self {
            nodes,
            dfs_buses,
            bridge_children,
        })
    }

    /// Solver bus indices ordered by their DFS discovery number.
    pub fn dfs_buses(&self) -> &[usize] {
        &self.dfs_buses
    }
    pub fn dfs_number(&self, solver_bus: usize) -> Option<usize> {
        self.nodes.get(solver_bus).map(|node| node.number)
    }
    pub fn subtree_size(&self, solver_bus: usize) -> Option<usize> {
        self.nodes.get(solver_bus).map(|node| node.subtree_size)
    }
    pub fn bridge_count(&self) -> usize {
        self.bridge_children
            .iter()
            .filter(|&&child| child != NONE)
            .count()
    }
    /// None for a non-bridge, inactive device, self-loop or invalid branch index.
    /// Counts use all fixed Slack buses, including multiple references.
    pub fn bridge_cut(&self, branch_index: usize) -> Option<BridgeCut> {
        let child = *self.bridge_children.get(branch_index)?;
        let node = *self.nodes.get(child)?;
        let root = self.nodes[node.component_root];
        Some(BridgeCut {
            subtree: node.number..node.number + node.subtree_size,
            component: root.number..root.number + root.subtree_size,
            subtree_slacks: node.subtree_slacks,
            other_slacks: root.subtree_slacks - node.subtree_slacks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn reachable(
        n: usize,
        edges: &[(usize, usize, bool)],
        start: usize,
        skip: usize,
    ) -> BTreeSet<usize> {
        let mut seen = BTreeSet::from([start]);
        let mut queue = vec![start];
        while let Some(bus) = queue.pop() {
            for (i, &(from, to, active)) in edges.iter().enumerate() {
                if i == skip || !active {
                    continue;
                }
                let next = if from == bus {
                    to
                } else if to == bus {
                    from
                } else {
                    continue;
                };
                if seen.insert(next) {
                    queue.push(next);
                }
            }
        }
        assert!(seen.len() <= n);
        seen
    }

    fn check_against_removal(n: usize, slacks: &[usize], edges: &[(usize, usize, bool)]) {
        let graph =
            N1Topology::from_edges(n, slacks.iter().copied(), edges.iter().copied()).unwrap();
        assert_eq!(
            graph.dfs_buses().iter().copied().collect::<BTreeSet<_>>(),
            (0..n).collect()
        );
        for (number, &bus) in graph.dfs_buses().iter().enumerate() {
            assert_eq!(graph.dfs_number(bus), Some(number));
        }
        for (i, &(from, to, active)) in edges.iter().enumerate() {
            let before = reachable(n, edges, from, NONE);
            let after = reachable(n, edges, from, i);
            let expected_bridge = active && !after.contains(&to);
            let cut = graph.bridge_cut(i);
            assert_eq!(cut.is_some(), expected_bridge, "edge {i} in {edges:?}");
            if let Some(cut) = cut {
                let subtree: BTreeSet<_> = graph.dfs_buses()[cut.subtree.clone()]
                    .iter()
                    .copied()
                    .collect();
                let component: BTreeSet<_> = graph.dfs_buses()[cut.component.clone()]
                    .iter()
                    .copied()
                    .collect();
                assert_eq!(component, before);
                assert!(
                    subtree == after || subtree == before.difference(&after).copied().collect()
                );
                let child = graph.dfs_buses()[cut.subtree.start];
                assert_eq!(graph.subtree_size(child), Some(subtree.len()));
                assert_eq!(
                    cut.subtree_slacks,
                    slacks.iter().filter(|s| subtree.contains(s)).count()
                );
                assert_eq!(
                    cut.other_slacks,
                    slacks
                        .iter()
                        .filter(|s| before.contains(s) && !subtree.contains(s))
                        .count()
                );
                for bus in 0..n {
                    assert_eq!(
                        cut.contains_in_subtree(graph.dfs_number(bus).unwrap()),
                        subtree.contains(&bus)
                    );
                }
                let expected_slackless = if cut.subtree_slacks == 0 {
                    subtree.len()
                } else {
                    0
                } + if cut.other_slacks == 0 {
                    before.len() - subtree.len()
                } else {
                    0
                };
                assert_eq!(cut.slackless_bus_count(), expected_slackless);
            }
        }
        assert!(graph.bridge_cut(edges.len()).is_none());
        assert!(graph.dfs_number(n).is_none());
    }

    #[test]
    fn all_five_bus_simple_graphs_match_explicit_edge_removal() {
        let possible: Vec<_> = (0..5)
            .flat_map(|from| (from + 1..5).map(move |to| (from, to, true)))
            .collect();
        for mask in 0..1usize << possible.len() {
            let edges: Vec<_> = possible
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, &edge)| edge)
                .collect();
            check_against_removal(5, &[4], &edges);
        }
    }

    #[test]
    fn parallel_devices_inactive_edges_self_loops_and_multiple_slacks() {
        let edges = [
            (0, 1, true),
            (0, 1, true), // Neither parallel device is a bridge.
            (1, 2, true),
            (1, 2, false), // Inactive parallel device offers no path.
            (2, 2, true),
            (2, 3, true),
            (4, 5, true), // Separate base component.
        ];
        for slacks in [&[0][..], &[3][..], &[0, 3, 5][..], &[][..]] {
            check_against_removal(7, slacks, &edges);
        }
        let graph = N1Topology::from_edges(7, [0, 3, 5], edges).unwrap();
        assert_eq!(graph.bridge_count(), 3);
        assert_eq!(graph.bridge_cut(2).unwrap().slackless_bus_count(), 0);
    }

    #[test]
    fn long_feeder_uses_an_explicit_stack_and_slack_can_be_inside_the_subtree() {
        let n = 100_000;
        let graph =
            N1Topology::from_edges(n, [n - 1], (1..n).map(|bus| (bus - 1, bus, true))).unwrap();
        assert_eq!(graph.bridge_count(), n - 1);
        let cut = graph.bridge_cut(0).unwrap();
        assert_eq!((cut.subtree_slacks, cut.other_slacks), (1, 0));
        assert_eq!(cut.slackless_bus_count(), 1);
        assert_eq!(graph.subtree_size(0), Some(n));
        assert_eq!(graph.subtree_size(n - 1), Some(1));
    }

    #[test]
    fn invalid_endpoints_and_slacks_return_errors_and_empty_graph_is_valid() {
        assert!(N1Topology::from_edges(2, [2], []).is_err());
        assert!(N1Topology::from_edges(2, [0], [(0, 2, false)]).is_err());
        let empty = N1Topology::from_edges(0, [], []).unwrap();
        assert_eq!(empty.bridge_count(), 0);
        assert!(empty.dfs_buses().is_empty());
    }
}
