//! Structural graph algorithms over the shared native snapshot.

use crate::graph::NativeGraph;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub(crate) fn cyclic_components(graph: &NativeGraph) -> Vec<Vec<u64>> {
    let n = graph.node_ids.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next_edge)) = stack.pop() {
            if next_edge < graph.adjacency[node].len() {
                stack.push((node, next_edge + 1));
                let target = graph.adjacency[node][next_edge].target;
                if !visited[target] {
                    visited[target] = true;
                    stack.push((target, 0));
                }
            } else {
                order.push(node);
            }
        }
    }

    visited.fill(false);
    let mut components = Vec::new();
    for &start in order.iter().rev() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(node) = stack.pop() {
            component.push(node);
            for edge in &graph.reverse_adjacency[node] {
                if !visited[edge.target] {
                    visited[edge.target] = true;
                    stack.push(edge.target);
                }
            }
        }
        let cyclic = component.len() > 1
            || graph.adjacency[component[0]]
                .iter()
                .any(|edge| edge.target == component[0]);
        if cyclic {
            let mut ids: Vec<u64> = component
                .into_iter()
                .map(|index| graph.node_ids[index])
                .collect();
            ids.sort_unstable();
            components.push(ids);
        }
    }
    components.sort_by_key(|component| component[0]);
    components
}

fn undirected(graph: &NativeGraph) -> Vec<Vec<usize>> {
    let mut sets = vec![HashSet::new(); graph.node_ids.len()];
    for (source, edges) in graph.adjacency.iter().enumerate() {
        for edge in edges {
            if source != edge.target {
                sets[source].insert(edge.target);
                sets[edge.target].insert(source);
            }
        }
    }
    sets.into_iter()
        .map(|set| {
            let mut neighbors: Vec<_> = set.into_iter().collect();
            neighbors.sort_unstable();
            neighbors
        })
        .collect()
}

pub(crate) fn core_numbers(graph: &NativeGraph) -> HashMap<u64, usize> {
    let adjacency = undirected(graph);
    let n = adjacency.len();
    let mut degree: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut removed = vec![false; n];
    let mut heap: BinaryHeap<Reverse<(usize, usize)>> = degree
        .iter()
        .enumerate()
        .map(|(node, &degree)| Reverse((degree, node)))
        .collect();
    let mut core = vec![0usize; n];
    let mut level = 0;
    while let Some(Reverse((candidate_degree, node))) = heap.pop() {
        if removed[node] || candidate_degree != degree[node] {
            continue;
        }
        removed[node] = true;
        level = level.max(candidate_degree);
        core[node] = level;
        for &neighbor in &adjacency[node] {
            if !removed[neighbor] {
                degree[neighbor] = degree[neighbor].saturating_sub(1);
                heap.push(Reverse((degree[neighbor], neighbor)));
            }
        }
    }
    graph.node_ids.iter().copied().zip(core).collect()
}

pub(crate) fn articulation_points(graph: &NativeGraph) -> Vec<u64> {
    let adjacency = undirected(graph);
    let n = adjacency.len();
    let mut discovery = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![usize::MAX; n];
    let mut points = vec![false; n];
    let mut time = 0usize;

    fn visit(
        node: usize,
        adjacency: &[Vec<usize>],
        discovery: &mut [usize],
        low: &mut [usize],
        parent: &mut [usize],
        points: &mut [bool],
        time: &mut usize,
    ) {
        discovery[node] = *time;
        low[node] = *time;
        *time += 1;
        let mut children = 0;
        for &neighbor in &adjacency[node] {
            if discovery[neighbor] == usize::MAX {
                children += 1;
                parent[neighbor] = node;
                visit(neighbor, adjacency, discovery, low, parent, points, time);
                low[node] = low[node].min(low[neighbor]);
                if parent[node] == usize::MAX && children > 1 {
                    points[node] = true;
                }
                if parent[node] != usize::MAX && low[neighbor] >= discovery[node] {
                    points[node] = true;
                }
            } else if neighbor != parent[node] {
                low[node] = low[node].min(discovery[neighbor]);
            }
        }
    }

    for node in 0..n {
        if discovery[node] == usize::MAX {
            visit(
                node,
                &adjacency,
                &mut discovery,
                &mut low,
                &mut parent,
                &mut points,
                &mut time,
            );
        }
    }
    points
        .into_iter()
        .enumerate()
        .filter_map(|(index, point)| point.then_some(graph.node_ids[index]))
        .collect()
}
