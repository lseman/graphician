//! Compact indexed graph shared by native analysis algorithms.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct NativeEdge {
    pub(crate) target: usize,
    pub(crate) weight: f32,
    pub(crate) kind: String,
    pub(crate) confidence: f32,
    pub(crate) ambiguous: bool,
}

/// An immutable graph snapshot optimized for repeated native analysis.
#[pyclass(module = "graphician_native")]
pub struct NativeGraph {
    pub(crate) node_ids: Vec<u64>,
    pub(crate) node_index: HashMap<u64, usize>,
    pub(crate) adjacency: Vec<Vec<NativeEdge>>,
    pub(crate) reverse_adjacency: Vec<Vec<NativeEdge>>,
}

#[pymethods]
impl NativeGraph {
    #[new]
    pub fn new(node_ids: Vec<u64>, edges: Vec<(u64, u64, String, String)>) -> PyResult<Self> {
        let node_index: HashMap<u64, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect();
        if node_index.len() != node_ids.len() {
            return Err(PyValueError::new_err("node_ids must be unique"));
        }

        let mut adjacency = vec![Vec::new(); node_ids.len()];
        let mut reverse_adjacency = vec![Vec::new(); node_ids.len()];
        for (source, target, kind, confidence) in edges {
            let Some(&source_index) = node_index.get(&source) else {
                return Err(PyValueError::new_err(format!(
                    "edge source {source} is not present in node_ids"
                )));
            };
            let Some(&target_index) = node_index.get(&target) else {
                return Err(PyValueError::new_err(format!(
                    "edge target {target} is not present in node_ids"
                )));
            };
            let ambiguous = confidence.eq_ignore_ascii_case("ambiguous");
            let confidence_weight = if confidence.eq_ignore_ascii_case("extracted") {
                1.0
            } else {
                0.0
            };
            adjacency[source_index].push(NativeEdge {
                target: target_index,
                weight: centrality_edge_weight(&kind) * confidence_weight,
                kind: kind.clone(),
                confidence: confidence_weight,
                ambiguous,
            });
            reverse_adjacency[target_index].push(NativeEdge {
                target: source_index,
                weight: centrality_edge_weight(&kind) * confidence_weight,
                kind,
                confidence: confidence_weight,
                ambiguous,
            });
        }

        for neighbors in &mut adjacency {
            neighbors.sort_unstable_by_key(|edge| edge.target);
        }
        for neighbors in &mut reverse_adjacency {
            neighbors.sort_unstable_by_key(|edge| edge.target);
        }
        Ok(Self {
            node_ids,
            node_index,
            adjacency,
            reverse_adjacency,
        })
    }

    #[getter]
    pub fn node_count(&self) -> usize {
        self.node_ids.len()
    }

    #[getter]
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(Vec::len).sum()
    }

    #[pyo3(signature = (damping=0.85, iterations=30, seeds=None))]
    pub fn pagerank(
        &self,
        damping: f32,
        iterations: usize,
        seeds: Option<Vec<(u64, f32)>>,
    ) -> PyResult<HashMap<u64, f32>> {
        crate::analysis::centrality::pagerank(self, damping, iterations, seeds.as_deref())
    }

    #[pyo3(signature = (start, edge_kind=None, reverse=false, max_hops=6, min_confidence=0.0))]
    pub fn traverse(
        &self,
        start: u64,
        edge_kind: Option<&str>,
        reverse: bool,
        max_hops: usize,
        min_confidence: f32,
    ) -> PyResult<Vec<u64>> {
        crate::analysis::traversal::traverse(self, start, edge_kind, reverse, max_hops, min_confidence)
    }

    #[pyo3(signature = (start, target=None, max_hops=6, edge_kinds=None, min_confidence=0.0))]
    pub fn paths(
        &self,
        start: u64,
        target: Option<u64>,
        max_hops: usize,
        edge_kinds: Option<Vec<String>>,
        min_confidence: f32,
    ) -> PyResult<Vec<Vec<u64>>> {
        crate::analysis::traversal::paths(
            self,
            start,
            target,
            max_hops,
            edge_kinds.as_deref(),
            min_confidence,
        )
    }

    #[pyo3(signature = (start, max_hops=20))]
    pub fn max_depth(&self, start: u64, max_hops: usize) -> PyResult<usize> {
        crate::analysis::traversal::max_depth(self, start, max_hops)
    }

    pub fn cyclic_components(&self) -> Vec<Vec<u64>> {
        crate::analysis::structure::cyclic_components(self)
    }

    pub fn core_numbers(&self) -> HashMap<u64, usize> {
        crate::analysis::structure::core_numbers(self)
    }

    pub fn articulation_points(&self) -> Vec<u64> {
        crate::analysis::structure::articulation_points(self)
    }

    #[pyo3(signature = (seed, max_hops=4))]
    pub fn impact(
        &self,
        seed: u64,
        max_hops: usize,
    ) -> PyResult<Vec<(u64, f32, usize, Vec<String>)>> {
        crate::analysis::traversal::impact(self, seed, max_hops)
    }

    #[pyo3(signature = (candidate_ids, pattern_edges, limit=50))]
    pub fn motif_matches(
        &self,
        candidate_ids: Vec<Vec<u64>>,
        pattern_edges: Vec<(usize, usize, Option<String>)>,
        limit: usize,
    ) -> PyResult<Vec<Vec<u64>>> {
        crate::analysis::motifs::motif_matches(self, candidate_ids, pattern_edges, limit)
    }

    #[pyo3(signature = (assignments, resolution=1.0))]
    pub fn community_quality_metrics(
        &self,
        assignments: Vec<(u64, i64)>,
        resolution: f64,
    ) -> crate::analysis::community_metrics::QualityMetrics {
        crate::analysis::community_metrics::community_quality_metrics(self, assignments, resolution)
    }

    #[pyo3(signature = (entries, placeholder_ids, max_depth=6, max_nodes_per_flow=200))]
    pub fn trace_flows(
        &self,
        entries: Vec<u64>,
        placeholder_ids: Vec<u64>,
        max_depth: usize,
        max_nodes_per_flow: usize,
    ) -> PyResult<Vec<Vec<(u64, usize)>>> {
        crate::analysis::flows::trace_flows(
            self,
            &entries,
            &placeholder_ids,
            max_depth,
            max_nodes_per_flow,
        )
    }
}

fn centrality_edge_weight(kind: &str) -> f32 {
    match kind {
        "defines" | "Defines" => 0.7,
        "calls" | "Calls" => 1.0,
        "imports" | "Imports" | "depends_on" | "DependsOn" => 0.55,
        "inherits" | "Inherits" | "implements" | "Implements" => 1.15,
        "data_flow" | "DataFlow" => 0.8,
        "reads_writes" | "ReadsWrites" => 0.9,
        "mentions" | "Mentions" | "describes" | "Describes" | "documented_by" | "DocumentedBy" => {
            0.75
        }
        "similar_to" | "SimilarTo" | "rationale_for" | "RationaleFor" | "illustrates"
        | "Illustrates" => 0.6,
        "tested_by" | "TestedBy" => 0.3,
        "member_of" | "MemberOf" | "entry_of" | "EntryOf" => 0.05,
        _ => 0.5,
    }
}
