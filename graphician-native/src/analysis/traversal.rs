//! Shared bounded traversal, path enumeration, and impact walking.

use crate::graph::{NativeEdge, NativeGraph};
use pyo3::exceptions::PyKeyError;
use pyo3::PyResult;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

fn start_index(graph: &NativeGraph, id: u64) -> PyResult<usize> {
    graph
        .node_index
        .get(&id)
        .copied()
        .ok_or_else(|| PyKeyError::new_err(format!("node {id} is not present")))
}

pub(crate) fn traverse(
    graph: &NativeGraph,
    start: u64,
    edge_kind: Option<&str>,
    reverse: bool,
    max_hops: usize,
) -> PyResult<Vec<u64>> {
    let start = start_index(graph, start)?;
    let adjacency = if reverse {
        &graph.reverse_adjacency
    } else {
        &graph.adjacency
    };
    let mut visited = vec![false; graph.node_ids.len()];
    visited[start] = true;
    let mut queue = VecDeque::from([(start, 0usize)]);
    let mut result = Vec::new();
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for edge in &adjacency[node] {
            if edge_kind.is_some_and(|kind| edge.kind != kind) || visited[edge.target] {
                continue;
            }
            visited[edge.target] = true;
            result.push(graph.node_ids[edge.target]);
            queue.push_back((edge.target, depth + 1));
        }
    }
    Ok(result)
}

pub(crate) fn paths(
    graph: &NativeGraph,
    start: u64,
    target: Option<u64>,
    max_hops: usize,
    edge_kinds: Option<&[String]>,
    min_confidence: f32,
) -> PyResult<Vec<Vec<u64>>> {
    let start = start_index(graph, start)?;
    let target = target.map(|id| start_index(graph, id)).transpose()?;
    let allowed: Option<HashSet<&str>> =
        edge_kinds.map(|kinds| kinds.iter().map(String::as_str).collect());
    let mut queue = VecDeque::from([vec![start]]);
    let mut result = Vec::new();
    while let Some(path) = queue.pop_front() {
        let hops = path.len() - 1;
        let node = *path.last().expect("paths are never empty");
        if hops > 0 && target.is_none_or(|target| target == node) {
            result.push(path.iter().map(|&index| graph.node_ids[index]).collect());
        }
        if hops >= max_hops || target == Some(node) {
            continue;
        }
        for edge in &graph.adjacency[node] {
            if edge.confidence < min_confidence
                || allowed
                    .as_ref()
                    .is_some_and(|kinds| !kinds.contains(edge.kind.as_str()))
                || path.contains(&edge.target)
            {
                continue;
            }
            let mut next = path.clone();
            next.push(edge.target);
            queue.push_back(next);
        }
    }
    Ok(result)
}

pub(crate) fn max_depth(graph: &NativeGraph, start: u64, max_hops: usize) -> PyResult<usize> {
    let start = start_index(graph, start)?;
    let mut depth = vec![usize::MAX; graph.node_ids.len()];
    depth[start] = 0;
    let mut queue = VecDeque::from([start]);
    let mut maximum = 0;
    while let Some(node) = queue.pop_front() {
        if depth[node] >= max_hops {
            continue;
        }
        for edge in &graph.adjacency[node] {
            let next_depth = depth[node] + 1;
            if next_depth < depth[edge.target] {
                depth[edge.target] = next_depth;
                maximum = maximum.max(next_depth);
                queue.push_back(edge.target);
            }
        }
    }
    Ok(maximum)
}

#[derive(Clone)]
struct ImpactState {
    cost: f32,
    distance: usize,
    node: usize,
    via: Vec<String>,
}

impl Eq for ImpactState {}
impl PartialEq for ImpactState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}
impl Ord for ImpactState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.distance.cmp(&self.distance))
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for ImpactState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub(crate) fn impact(
    graph: &NativeGraph,
    seed: u64,
    max_hops: usize,
) -> PyResult<Vec<(u64, f32, usize, Vec<String>)>> {
    let seed = start_index(graph, seed)?;
    let mut heap = BinaryHeap::from([ImpactState {
        cost: 0.0,
        distance: 0,
        node: seed,
        via: Vec::new(),
    }]);
    let mut best: HashMap<usize, (f32, usize, Vec<String>)> = HashMap::new();
    while let Some(state) = heap.pop() {
        if state.distance > max_hops
            || best
                .get(&state.node)
                .is_some_and(|(cost, _, _)| *cost <= state.cost)
        {
            continue;
        }
        best.insert(state.node, (state.cost, state.distance, state.via.clone()));
        if state.distance == max_hops {
            continue;
        }
        for edge in &graph.reverse_adjacency[state.node] {
            push_impact(&mut heap, &state, edge, impact_cost(edge));
        }
        for edge in &graph.adjacency[state.node] {
            if let Some(cost) = forward_impact_cost(edge) {
                push_impact(&mut heap, &state, edge, cost);
            }
        }
    }
    best.remove(&seed);
    let mut result: Vec<_> = best
        .into_iter()
        .map(|(node, (cost, distance, via))| (graph.node_ids[node], cost, distance, via))
        .collect();
    result.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
    Ok(result)
}

fn push_impact(
    heap: &mut BinaryHeap<ImpactState>,
    state: &ImpactState,
    edge: &NativeEdge,
    cost: f32,
) {
    let mut via = state.via.clone();
    via.push(edge.kind.clone());
    heap.push(ImpactState {
        cost: state.cost + cost,
        distance: state.distance + 1,
        node: edge.target,
        via,
    });
}

fn impact_cost(edge: &NativeEdge) -> f32 {
    let base = match edge.kind.as_str() {
        "calls" => 1.0,
        "defines" => 1.25,
        "imports" | "depends_on" => 1.6,
        "inherits" | "implements" => 0.75,
        "data_flow" => 0.8,
        "reads_writes" => 0.9,
        "tested_by" => 1.1,
        "member_of" | "entry_of" => 5.0,
        "describes" | "documented_by" => 1.2,
        "mentions" | "illustrates" => 1.8,
        "similar_to" | "rationale_for" => 2.0,
        _ => 1.5,
    };
    base / edge.confidence.max(0.05)
}

fn forward_impact_cost(edge: &NativeEdge) -> Option<f32> {
    let base = match edge.kind.as_str() {
        "calls" => 2.25,
        "imports" | "depends_on" => 2.5,
        "inherits" | "implements" => 1.75,
        "data_flow" | "reads_writes" => 2.0,
        _ => return None,
    };
    Some(base / edge.confidence.max(0.05))
}
