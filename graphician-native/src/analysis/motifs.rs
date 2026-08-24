//! Topology-heavy motif matching over a [`NativeGraph`] snapshot.

use crate::graph::NativeGraph;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

pub(crate) fn motif_matches(
    graph: &NativeGraph,
    candidate_ids: Vec<Vec<u64>>,
    pattern_edges: Vec<(usize, usize, Option<String>)>,
    limit: usize,
) -> PyResult<Vec<Vec<u64>>> {
    if candidate_ids.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    if pattern_edges
        .iter()
        .any(|(source, target, _)| *source >= candidate_ids.len() || *target >= candidate_ids.len())
    {
        return Err(PyValueError::new_err(
            "pattern edge references an unknown pattern node",
        ));
    }

    let candidates: Vec<Vec<usize>> = candidate_ids
        .into_iter()
        .map(|ids| {
            ids.into_iter()
                .map(|node_id| {
                    graph.node_index.get(&node_id).copied().ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "motif candidate {node_id} is not present in the native graph"
                        ))
                    })
                })
                .collect()
        })
        .collect::<PyResult<_>>()?;

    let mut current = vec![None; candidates.len()];
    let mut used = HashSet::new();
    let mut results = Vec::new();
    backtrack(
        graph,
        &candidates,
        &pattern_edges,
        0,
        limit,
        &mut current,
        &mut used,
        &mut results,
    );
    Ok(results)
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    graph: &NativeGraph,
    candidates: &[Vec<usize>],
    pattern_edges: &[(usize, usize, Option<String>)],
    depth: usize,
    limit: usize,
    current: &mut [Option<usize>],
    used: &mut HashSet<usize>,
    results: &mut Vec<Vec<u64>>,
) {
    if results.len() >= limit {
        return;
    }
    if depth == candidates.len() {
        results.push(
            current
                .iter()
                .map(|node| graph.node_ids[node.expect("complete motif mapping")])
                .collect(),
        );
        return;
    }

    for &candidate in &candidates[depth] {
        if !used.insert(candidate) {
            continue;
        }
        current[depth] = Some(candidate);
        if assigned_edges_match(graph, current, pattern_edges) {
            backtrack(
                graph,
                candidates,
                pattern_edges,
                depth + 1,
                limit,
                current,
                used,
                results,
            );
        }
        current[depth] = None;
        used.remove(&candidate);
        if results.len() >= limit {
            return;
        }
    }
}

fn assigned_edges_match(
    graph: &NativeGraph,
    current: &[Option<usize>],
    pattern_edges: &[(usize, usize, Option<String>)],
) -> bool {
    pattern_edges.iter().all(|(source, target, kind)| {
        let (Some(source), Some(target)) = (current[*source], current[*target]) else {
            return true;
        };
        graph.adjacency[source].iter().any(|edge| {
            edge.target == target && kind.as_ref().is_none_or(|kind| edge.kind == *kind)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_directed_typed_pattern() {
        let graph = NativeGraph::new(
            vec![1, 2, 3],
            vec![
                (1, 2, "calls".into(), "extracted".into()),
                (1, 3, "imports".into(), "extracted".into()),
            ],
        )
        .unwrap();
        let matches = motif_matches(
            &graph,
            vec![vec![1], vec![2, 3]],
            vec![(0, 1, Some("calls".into()))],
            10,
        )
        .unwrap();
        assert_eq!(matches, vec![vec![1, 2]]);
    }
}
