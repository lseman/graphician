//! Batch execution-flow tracing over a compact native graph snapshot.

use crate::graph::NativeGraph;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashSet, VecDeque};

pub(crate) fn trace_flows(
    graph: &NativeGraph,
    entries: &[u64],
    placeholder_ids: &[u64],
    max_depth: usize,
    max_nodes_per_flow: usize,
) -> PyResult<Vec<Vec<(u64, usize)>>> {
    if max_nodes_per_flow == 0 {
        return Ok(vec![Vec::new(); entries.len()]);
    }
    let placeholders: HashSet<usize> = placeholder_ids
        .iter()
        .filter_map(|id| graph.node_index.get(id).copied())
        .collect();
    entries
        .iter()
        .map(|entry| {
            let start = graph
                .node_index
                .get(entry)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("unknown entry node {entry}")))?;
            Ok(trace_one(
                graph,
                start,
                &placeholders,
                max_depth,
                max_nodes_per_flow,
            ))
        })
        .collect()
}

fn trace_one(
    graph: &NativeGraph,
    start: usize,
    placeholders: &HashSet<usize>,
    max_depth: usize,
    max_nodes_per_flow: usize,
) -> Vec<(u64, usize)> {
    let safety_ceiling = max_nodes_per_flow.saturating_mul(10).max(500);
    let mut visited = HashSet::from([start]);
    let mut members = Vec::new();
    let mut queue = VecDeque::from([(start, 0usize)]);

    while let Some((node, depth)) = queue.pop_front() {
        members.push((node, depth));
        if members.len() >= safety_ceiling {
            break;
        }
        if depth >= max_depth {
            continue;
        }
        for edge in &graph.adjacency[node] {
            if !edge.kind.eq_ignore_ascii_case("calls")
                || edge.ambiguous
                || placeholders.contains(&edge.target)
            {
                continue;
            }
            if visited.insert(edge.target) {
                queue.push_back((edge.target, depth + 1));
            }
        }
    }

    if members.len() > max_nodes_per_flow {
        let member_ids: HashSet<usize> = members.iter().map(|(id, _)| *id).collect();
        let mut scored: Vec<(usize, usize, f64)> = members
            .into_iter()
            .map(|(id, depth)| {
                let fanin = graph.reverse_adjacency[id]
                    .iter()
                    .filter(|edge| {
                        edge.kind.eq_ignore_ascii_case("calls") && member_ids.contains(&edge.target)
                    })
                    .count();
                let score = 1.0 / (depth as f64 + 1.0) + (fanin as f64 / 10.0).min(0.5);
                (id, depth, score)
            })
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max_nodes_per_flow);
        members = scored
            .into_iter()
            .map(|(id, depth, _)| (id, depth))
            .collect();
    }

    members
        .into_iter()
        .map(|(index, depth)| (graph.node_ids[index], depth))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_ambiguous_edges_and_placeholders() {
        let graph = NativeGraph::new(
            vec![1, 2, 3, 4],
            vec![
                (1, 2, "calls".into(), "extracted".into()),
                (2, 3, "calls".into(), "ambiguous".into()),
                (2, 4, "calls".into(), "extracted".into()),
            ],
        )
        .unwrap();
        assert_eq!(
            trace_flows(&graph, &[1], &[4], 6, 200).unwrap(),
            vec![vec![(1, 0), (2, 1)]]
        );
    }
}
