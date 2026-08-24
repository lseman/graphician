//! Weighted and personalized PageRank over a native graph snapshot.

use crate::graph::NativeGraph;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::collections::HashMap;

pub(crate) fn pagerank(
    graph: &NativeGraph,
    damping: f32,
    iterations: usize,
    seeds: Option<&[(u64, f32)]>,
) -> PyResult<HashMap<u64, f32>> {
    if !(0.0..=1.0).contains(&damping) {
        return Err(PyValueError::new_err("damping must be between 0 and 1"));
    }
    let n = graph.node_ids.len();
    if n == 0 {
        return Ok(HashMap::new());
    }

    let index: HashMap<u64, usize> = graph
        .node_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect();
    let mut teleport = vec![0.0f32; n];
    if let Some(seeds) = seeds {
        for &(id, weight) in seeds {
            if let Some(&node_index) = index.get(&id) {
                teleport[node_index] += weight.max(0.0);
            }
        }
    }
    let teleport_total: f32 = teleport.iter().sum();
    if teleport_total > 0.0 {
        for weight in &mut teleport {
            *weight /= teleport_total;
        }
    } else {
        teleport.fill(1.0 / n as f32);
    }

    let totals: Vec<f32> = graph
        .adjacency
        .iter()
        .map(|neighbors| neighbors.iter().map(|edge| edge.weight).sum())
        .collect();
    let mut ranks = vec![1.0 / n as f32; n];
    for _ in 0..iterations {
        let mut next: Vec<f32> = teleport
            .iter()
            .map(|weight| (1.0 - damping) * weight)
            .collect();
        let mut dangling_mass = 0.0;
        for (source, neighbors) in graph.adjacency.iter().enumerate() {
            if neighbors.iter().all(|edge| edge.ambiguous) || totals[source] <= 0.0 {
                dangling_mass += ranks[source];
                continue;
            }
            for edge in neighbors {
                if edge.ambiguous {
                    continue;
                }
                next[edge.target] += damping * ranks[source] * edge.weight / totals[source];
            }
        }
        for (rank, teleport_weight) in next.iter_mut().zip(&teleport) {
            *rank += damping * dangling_mass * teleport_weight;
        }
        ranks = next;
    }

    Ok(graph.node_ids.iter().copied().zip(ranks).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> NativeGraph {
        NativeGraph::new(
            vec![0, 1, 2],
            vec![
                (0, 1, "calls".into(), "extracted".into()),
                (2, 1, "calls".into(), "extracted".into()),
            ],
        )
        .unwrap()
    }

    #[test]
    fn pagerank_concentrates_on_sink() {
        let ranks = pagerank(&graph(), 0.85, 30, None).unwrap();
        assert!(ranks[&1] > ranks[&0]);
        assert!(ranks[&1] > ranks[&2]);
    }

    #[test]
    fn personalization_biases_seed() {
        let ranks = pagerank(&graph(), 0.85, 30, Some(&[(0, 1.0)])).unwrap();
        assert!(ranks[&0] > ranks[&2]);
    }
}
