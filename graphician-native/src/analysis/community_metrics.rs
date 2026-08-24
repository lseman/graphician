//! One-pass quality metrics for a supplied community partition.

use crate::graph::NativeGraph;
use std::collections::{HashMap, HashSet};

pub(crate) type QualityMetrics = (
    HashMap<i64, f64>,
    usize,
    usize,
    usize,
    usize,
    f64,
    f64,
    f64,
    f64,
    f64,
    usize,
);

pub(crate) fn community_quality_metrics(
    graph: &NativeGraph,
    assignments: Vec<(u64, i64)>,
    resolution: f64,
) -> QualityMetrics {
    let communities: HashMap<u64, i64> = assignments.into_iter().collect();
    let mut sizes: HashMap<i64, usize> = HashMap::new();
    for &community in communities.values() {
        *sizes.entry(community).or_default() += 1;
    }

    let mut internal_pairs: HashMap<i64, HashSet<(u64, u64)>> = HashMap::new();
    let mut internal_edges: HashMap<i64, usize> = HashMap::new();
    let mut external_edges: HashMap<i64, usize> = HashMap::new();
    for (source_index, edges) in graph.adjacency.iter().enumerate() {
        let source_id = graph.node_ids[source_index];
        let source_community = communities.get(&source_id).copied();
        for edge in edges {
            let target_id = graph.node_ids[edge.target];
            let target_community = communities.get(&target_id).copied();
            if let (Some(source), Some(target)) = (source_community, target_community) {
                if source == target {
                    if source_id != target_id {
                        internal_pairs
                            .entry(source)
                            .or_default()
                            .insert((source_id.min(target_id), source_id.max(target_id)));
                    }
                    *internal_edges.entry(source).or_default() += 1;
                }
            }
            if source_community != target_community {
                if let Some(community) = source_community {
                    *external_edges.entry(community).or_default() += 1;
                }
                if let Some(community) = target_community {
                    *external_edges.entry(community).or_default() += 1;
                }
            }
        }
    }

    let mut cohesion = HashMap::new();
    let mut conductances = Vec::with_capacity(sizes.len());
    let mut cohesion_values = Vec::new();
    let mut low_cohesion = 0;
    for (&community, &size) in &sizes {
        let value = if size <= 1 {
            1.0
        } else {
            let possible = size * (size - 1) / 2;
            internal_pairs.get(&community).map_or(0, HashSet::len) as f64 / possible as f64
        };
        cohesion.insert(community, value);
        if size <= 1 {
            conductances.push(0.0);
        } else {
            let internal = internal_edges.get(&community).copied().unwrap_or_default();
            let external = external_edges.get(&community).copied().unwrap_or_default();
            let total = internal + external;
            conductances.push(if total == 0 {
                0.0
            } else {
                external as f64 / total as f64
            });
            cohesion_values.push(value);
            if value < 0.15 {
                low_cohesion += 1;
            }
        }
    }

    let community_count = sizes.len();
    let singleton_count = sizes.values().filter(|&&size| size == 1).count();
    let min_size = sizes.values().copied().min().unwrap_or_default();
    let max_size = sizes.values().copied().max().unwrap_or_default();
    let mean_size = if community_count == 0 {
        0.0
    } else {
        sizes.values().sum::<usize>() as f64 / community_count as f64
    };
    let score = if community_count == 0 || graph.node_ids.is_empty() {
        0.0
    } else {
        let total_nodes = graph.node_ids.len() as f64;
        let sum_sq: f64 = sizes
            .values()
            .map(|&size| (size as f64 / total_nodes).powi(2))
            .sum();
        (sum_sq * resolution).sqrt()
    };
    let mean_conductance = mean(&conductances);
    let max_conductance = conductances.iter().copied().fold(0.0, f64::max);
    let mean_cohesion = mean(&cohesion_values);
    (
        cohesion,
        community_count,
        singleton_count,
        min_size,
        max_size,
        mean_size,
        score,
        mean_conductance,
        max_conductance,
        mean_cohesion,
        low_cohesion,
    )
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_cohesion_and_boundary_conductance() {
        let graph = NativeGraph::new(
            vec![1, 2, 3],
            vec![
                (1, 2, "calls".into(), "extracted".into()),
                (2, 3, "calls".into(), "extracted".into()),
            ],
        )
        .unwrap();
        let metrics = community_quality_metrics(&graph, vec![(1, 0), (2, 0), (3, 1)], 1.0);
        assert_eq!(metrics.0[&0], 1.0);
        assert_eq!(metrics.1, 2);
        assert_eq!(metrics.2, 1);
        assert!((metrics.7 - 0.25).abs() < f64::EPSILON);
    }
}
