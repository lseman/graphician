//! Community detection algorithms ported from Ariadne.
//!
//! Implements Louvain, Leiden, and Infomap community detection with proper edge weights,
//! multi-level aggregation, and Leiden-style refinement.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet, VecDeque};

/// Edge weight by kind — matches Ariadne's edge_kind_weight exactly.
pub fn edge_kind_weight(kind: &str) -> f32 {
    // Match both PascalCase (enum variant names) and lowercase (str representation)
    match kind {
        "Inherits" | "inherits" | "Implements" | "implements" => 1.25,
        "Defines" | "defines" => 0.7,
        "Calls" | "calls" => 0.55,
        "DataFlow" | "data_flow" | "dataflow" => 0.65,
        "ReadsWrites" | "reads_writes" | "readswrites" => 0.85,
        "Mentions" | "mentions" | "Describes" | "describes" | "DocumentedBy" | "documented_by" => {
            0.75
        }
        "TestedBy" | "tested_by" => 0.6,
        "Imports" | "imports" | "DependsOn" | "depends_on" => 0.45,
        "SimilarTo" | "similar_to" | "RationaleFor" | "rationale_for" | "Illustrates"
        | "illustrates" => 0.55,
        "MemberOf" | "member_of" | "EntryOf" | "entry_of" => 0.1,
        _ => 0.5,
    }
}

/// Community detection options — matches Ariadne's CommunityOptions.
#[pyclass(module = "graphician_native")]
#[derive(Debug, Clone)]
pub struct CommunityOptions {
    #[pyo3(get, set)]
    pub resolution: f32,
    #[pyo3(get, set)]
    pub max_passes: usize,
    #[pyo3(get, set)]
    pub max_levels: usize,
    #[pyo3(get, set)]
    pub well_connectedness: f32,
    #[pyo3(get, set)]
    pub min_modularity_gain: f32,
}

#[pymethods]
impl CommunityOptions {
    #[new]
    #[pyo3(signature = (resolution=1.0, max_passes=50, max_levels=10, well_connectedness=1.0, min_modularity_gain=1e-7))]
    fn new(
        resolution: f32,
        max_passes: usize,
        max_levels: usize,
        well_connectedness: f32,
        min_modularity_gain: f32,
    ) -> Self {
        Self {
            resolution,
            max_passes,
            max_levels,
            well_connectedness,
            min_modularity_gain,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CommunityOptions(resolution={}, max_passes={}, max_levels={}, well_connectedness={}, min_modularity_gain={})",
            self.resolution, self.max_passes, self.max_levels, self.well_connectedness, self.min_modularity_gain
        )
    }
}

/// Internal working graph for community algorithms.
#[derive(Clone)]
struct WorkingGraph {
    members: Vec<Vec<String>>,
    adj: Vec<Vec<(usize, f32)>>,
    self_loop: Vec<f32>,
    degree: Vec<f32>,
    total_weight: f32,
}

impl WorkingGraph {
    fn len(&self) -> usize {
        self.members.len()
    }

    fn from_py_data(
        nodes: &Bound<'_, PyList>,
        edges: &Bound<'_, PyList>,
        ambiguous_weight: f32,
    ) -> PyResult<Self> {
        let n = nodes.len();
        if n == 0 {
            return Ok(WorkingGraph {
                members: vec![],
                adj: vec![],
                self_loop: vec![],
                degree: vec![],
                total_weight: 0.0,
            });
        }

        let mut node_to_idx: HashMap<String, usize> = HashMap::with_capacity(n);
        let mut members: Vec<Vec<String>> = Vec::with_capacity(n);
        for i in 0..n {
            let name: String = nodes.get_item(i)?.extract()?;
            node_to_idx.insert(name.clone(), i);
            members.push(vec![name]);
        }

        let mut adj_map: Vec<HashMap<usize, f32>> = vec![HashMap::with_capacity(0); n];
        let mut self_loop = vec![0.0f32; n];

        for edge in edges.iter() {
            let edge_tuple: Bound<'_, PyAny> = edge.extract()?;

            let src: String = edge_tuple.get_item(0)?.extract()?;
            let dst: String = edge_tuple.get_item(1)?.extract()?;
            let kind: String = edge_tuple.get_item(2)?.extract()?;
            let confidence: String = edge_tuple.get_item(3)?.extract()?;

            let src_idx = *node_to_idx.get(&src).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node not found: {}", src))
            })?;
            let dst_idx = *node_to_idx.get(&dst).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Node not found: {}", dst))
            })?;

            let weight = match confidence.as_str() {
                "ambiguous" | "Ambiguous" | "Confidence::Ambiguous" => ambiguous_weight,
                _ => edge_kind_weight(&kind),
            };

            *adj_map[src_idx].entry(dst_idx).or_insert(0.0) += weight;
            if src == dst {
                self_loop[src_idx] += weight * 0.5;
            }
        }

        let adj: Vec<Vec<(usize, f32)>> = adj_map
            .into_iter()
            .map(|m| {
                let mut v: Vec<(usize, f32)> = m.into_iter().collect();
                v.sort_unstable_by_key(|(idx, _)| *idx);
                v
            })
            .collect();

        let degree: Vec<f32> = (0..n)
            .map(|u| adj[u].iter().map(|(_, w)| *w).sum::<f32>() + 2.0 * self_loop[u])
            .collect();
        let total_weight = degree.iter().sum::<f32>() / 2.0;

        Ok(WorkingGraph {
            members,
            adj,
            self_loop,
            degree,
            total_weight,
        })
    }
}

/// Densify labels: remap to contiguous [0, n) range.
pub fn densify_labels(labels: &[usize]) -> Vec<usize> {
    let mut mapping: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|&l| {
            *mapping.entry(l).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect()
}

/// Relabel community assignments to contiguous [0, n) range.
pub fn relabel(mut comm: HashMap<String, usize>) -> HashMap<String, usize> {
    let mut labels: HashMap<usize, usize> = HashMap::new();
    let mut next = 0usize;
    for label in comm.values_mut() {
        *label = *labels.entry(*label).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
    }
    comm
}

/// Aggregate a working graph by partition.
fn aggregate(prev: &WorkingGraph, partition: &[usize]) -> WorkingGraph {
    let dense = densify_labels(partition);
    let new_n = dense.iter().copied().max().map(|x| x + 1).unwrap_or(0);

    let mut new_members: Vec<Vec<String>> = vec![Vec::new(); new_n];
    for (u, members) in prev.members.iter().enumerate() {
        new_members[dense[u]].extend(members.iter().cloned());
    }

    let mut adj_map: Vec<HashMap<usize, f32>> = vec![HashMap::new(); new_n];
    let mut self_loop = vec![0.0f32; new_n];

    for u in 0..prev.len() {
        let cu = dense[u];
        self_loop[cu] += prev.self_loop[u];
        for &(v, w) in &prev.adj[u] {
            let cv = dense[v];
            if cu == cv {
                self_loop[cu] += w * 0.5;
            } else {
                *adj_map[cu].entry(cv).or_insert(0.0) += w;
            }
        }
    }

    let adj: Vec<Vec<(usize, f32)>> = adj_map
        .into_iter()
        .map(|m| {
            let mut v: Vec<(usize, f32)> = m.into_iter().collect();
            v.sort_unstable_by_key(|(idx, _)| *idx);
            v
        })
        .collect();

    let degree: Vec<f32> = (0..new_n)
        .map(|u| adj[u].iter().map(|(_, w)| *w).sum::<f32>() + 2.0 * self_loop[u])
        .collect();
    let total_weight = degree.iter().sum::<f32>() / 2.0;

    WorkingGraph {
        members: new_members,
        adj,
        self_loop,
        degree,
        total_weight,
    }
}

/// Enforce connectivity within each community using BFS.
fn enforce_connected(working: &WorkingGraph, labels: &mut [usize]) {
    let n = working.len();

    let mut undirected: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (u, neighbors) in working.adj.iter().enumerate() {
        for &(v, _) in neighbors {
            undirected[u].push(v);
            undirected[v].push(u);
        }
    }

    let mut by_label: HashMap<usize, Vec<usize>> = HashMap::new();
    for (u, &c) in labels.iter().enumerate() {
        by_label.entry(c).or_default().push(u);
    }

    let mut label_groups: Vec<(usize, Vec<usize>)> = by_label.into_iter().collect();
    label_groups.sort_unstable_by_key(|(l, _)| *l);

    let mut next_label = labels.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mut new_labels: Vec<Option<usize>> = vec![None; n];

    for (_, members) in &label_groups {
        let member_set: HashSet<usize> = members.iter().copied().collect();
        let mut unseen: Vec<usize> = members.clone();
        unseen.sort_unstable();
        let mut unseen_set = member_set.clone();
        let mut first_component = true;

        while let Some(&start) = unseen.iter().find(|u| unseen_set.contains(u)) {
            let component_label = if first_component {
                first_component = false;
                labels[start]
            } else {
                let l = next_label;
                next_label += 1;
                l
            };

            let mut queue = VecDeque::from([start]);
            unseen_set.remove(&start);

            while let Some(u) = queue.pop_front() {
                new_labels[u] = Some(component_label);
                for &v in &undirected[u] {
                    if member_set.contains(&v) && unseen_set.remove(&v) {
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    for (u, label) in new_labels.into_iter().enumerate() {
        if let Some(l) = label {
            labels[u] = l;
        }
    }
}

// ============================================================
// Louvain local move (mirrors Ariadne's implementation exactly)
// ============================================================

fn local_move(working: &WorkingGraph, options: &CommunityOptions) -> Vec<usize> {
    let n = working.len();
    let mut comm: Vec<usize> = (0..n).collect();
    let mut comm_degree: Vec<f32> = working.degree.clone();
    let mut comm_size: Vec<f32> = working.members.iter().map(|m| m.len() as f32).collect();
    let two_m = 2.0 * working.total_weight;
    if two_m <= 0.0 {
        return comm;
    }

    for _ in 0..options.max_passes {
        let mut moved = false;
        for u in 0..n {
            let current = comm[u];
            let node_degree = working.degree[u];
            if node_degree == 0.0 {
                continue;
            }
            let node_mass = working.members[u].len() as f32;

            // Remove u from its current community for the gain calculation.
            comm_degree[current] -= node_degree;
            comm_size[current] -= node_mass;

            let mut weight_to_comm: HashMap<usize, f32> = HashMap::new();
            for &(v, w) in &working.adj[u] {
                *weight_to_comm.entry(comm[v]).or_insert(0.0) += w;
            }

            let mut best = current;
            let mut best_gain = options.min_modularity_gain;
            let stay_weight = weight_to_comm.get(&current).copied().unwrap_or(0.0);
            let stay_gain =
                stay_weight - options.resolution * node_degree * comm_degree[current] / two_m;
            if stay_gain > best_gain {
                best_gain = stay_gain;
                best = current;
            }
            for (&candidate, &edge_weight) in &weight_to_comm {
                if candidate == current {
                    continue;
                }
                let gain =
                    edge_weight - options.resolution * node_degree * comm_degree[candidate] / two_m;
                if gain > best_gain {
                    best_gain = gain;
                    best = candidate;
                }
            }

            comm[u] = best;
            comm_degree[best] += node_degree;
            comm_size[best] += node_mass;
            if best != current {
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }

    comm
}

// ============================================================
// Leiden refinement phase
// ============================================================

fn refinement_phase(
    working: &WorkingGraph,
    partition: &[usize],
    options: &CommunityOptions,
) -> Vec<usize> {
    let n = working.len();
    let two_m = 2.0 * working.total_weight;
    if two_m <= 0.0 {
        return partition.to_vec();
    }

    let mut by_parent: HashMap<usize, Vec<usize>> = HashMap::new();
    for (u, &c) in partition.iter().enumerate() {
        by_parent.entry(c).or_default().push(u);
    }
    let mut parents: Vec<(usize, Vec<usize>)> = by_parent.into_iter().collect();
    parents.sort_by_key(|(p, _)| *p);

    let mut parent_degree: HashMap<usize, f32> = HashMap::new();
    for (u, &c) in partition.iter().enumerate() {
        *parent_degree.entry(c).or_insert(0.0) += working.degree[u];
    }

    let mut label_base = Vec::with_capacity(parents.len());
    let mut cursor = 0usize;
    for (_, members) in &parents {
        label_base.push(cursor);
        cursor += members.len();
    }

    let refined: Vec<Vec<usize>> = parents
        .iter()
        .enumerate()
        .map(|(idx, (_, members))| {
            let base = label_base[idx];
            let parent_total = parent_degree.get(&parents[idx].0).copied().unwrap_or(0.0);

            if members.len() <= 1 {
                return vec![base];
            }

            let member_set: HashSet<usize> = members.iter().copied().collect();
            let mut refined_map: HashMap<usize, usize> = members
                .iter()
                .enumerate()
                .map(|(i, &u)| (u, base + i))
                .collect();
            let mut local_degree: HashMap<usize, f32> = members
                .iter()
                .enumerate()
                .map(|(i, &u)| (base + i, working.degree[u]))
                .collect();
            let mut local_size: HashMap<usize, f32> = members
                .iter()
                .enumerate()
                .map(|(i, &u)| (base + i, working.members[u].len() as f32))
                .collect();

            for _ in 0..options.max_passes {
                let mut moved = false;
                for &u in members {
                    let current = *refined_map.get(&u).unwrap();
                    let node_degree = working.degree[u];
                    let node_mass = working.members[u].len() as f32;
                    if node_degree == 0.0 {
                        continue;
                    }
                    *local_degree.get_mut(&current).unwrap() -= node_degree;
                    *local_size.get_mut(&current).unwrap() -= node_mass;

                    let mut weight_to_comm: HashMap<usize, f32> = HashMap::new();
                    for &(v, w) in &working.adj[u] {
                        if !member_set.contains(&v) {
                            continue;
                        }
                        let r_v = *refined_map.get(&v).unwrap();
                        *weight_to_comm.entry(r_v).or_insert(0.0) += w;
                    }

                    let mut best = current;
                    let mut best_gain = options.min_modularity_gain;
                    let stay_weight = weight_to_comm.get(&current).copied().unwrap_or(0.0);
                    let stay_gain = stay_weight
                        - options.resolution * node_degree * local_degree[&current] / two_m;
                    if stay_gain > best_gain {
                        best_gain = stay_gain;
                        best = current;
                    }

                    for (&target, &weight) in &weight_to_comm {
                        if target == current {
                            continue;
                        }
                        let gain = weight
                            - options.resolution * node_degree * local_degree[&target] / two_m;

                        let threshold = if options.well_connectedness > 0.0
                            && parent_total > 0.0
                            && local_degree[&target] > 0.0
                        {
                            let w_ratio = weight / local_degree[&target];
                            let wc_threshold = options.well_connectedness
                                * (stay_weight / two_m
                                    - node_degree * local_degree[&current] / two_m
                                    + node_mass * local_size[&current] / two_m);
                            gain > best_gain && w_ratio >= wc_threshold
                        } else {
                            gain > best_gain
                        };
                        if threshold {
                            best_gain = gain;
                            best = target;
                        }
                    }

                    if best != current {
                        refined_map.insert(u, best);
                        moved = true;
                    }

                    *local_degree.get_mut(&current).unwrap() += node_degree;
                    *local_size.get_mut(&current).unwrap() += node_mass;
                }
                if !moved {
                    break;
                }
            }
            members
                .iter()
                .map(|&u| *refined_map.get(&u).unwrap())
                .collect()
        })
        .collect();

    let mut result = vec![0usize; n];
    for (idx, (_, members)) in parents.iter().enumerate() {
        for (i, &u) in members.iter().enumerate() {
            result[u] = refined[idx][i];
        }
    }

    enforce_connected(working, &mut result);
    densify_labels(&result)
}

// ============================================================
// Infomap-specific functions
// ============================================================

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn default() -> Self {
        Self { state: 0x5DEECE66D }
    }
    fn next(&mut self) {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    fn gen_range(&mut self, low: usize, high: usize) -> usize {
        self.next();
        low + (self.state as usize % (high - low))
    }
    fn gen_f32(&mut self) -> f32 {
        self.next();
        (((self.state >> 11) as f64) / 9007199254740992.0) as f32
    }
}

#[derive(Debug)]
struct CommunityFlow {
    node_probability: f32,
    exit_probability: f32,
    node_probabilities: Vec<f32>,
}

type CommunityStats = (f32, f32, f32);
type IncomingWeightMap = HashMap<usize, f32>;

fn entropy_term(probability: f32) -> f32 {
    if probability > 0.0 {
        probability * probability.log2()
    } else {
        0.0
    }
}

fn compute_lmdl(working: &WorkingGraph, labels: &[usize], two_m: f32) -> f32 {
    let n = working.len();
    if n == 0 || two_m <= 0.0 {
        return 0.0;
    }

    let flow = compute_community_flow(labels, working, two_m);

    let mut keys: Vec<usize> = flow.keys().copied().collect();
    keys.sort_unstable();
    let ordered: Vec<&CommunityFlow> = keys.iter().map(|k| &flow[k]).collect();

    let q_total: f32 = ordered.iter().map(|f| f.exit_probability).sum();
    let mut length = entropy_term(q_total);

    for community in &ordered {
        length -= entropy_term(community.exit_probability);
    }

    for community in &ordered {
        let p_circle = community.node_probability + community.exit_probability;
        length += entropy_term(p_circle);
        length -= entropy_term(community.exit_probability);
        for &node_probability in &community.node_probabilities {
            length -= entropy_term(node_probability);
        }
    }

    length.max(0.0)
}

fn compute_community_flow(
    labels: &[usize],
    working: &WorkingGraph,
    two_m: f32,
) -> HashMap<usize, CommunityFlow> {
    let mut flow: HashMap<usize, CommunityFlow> = HashMap::new();

    for (u, &l) in labels.iter().enumerate() {
        let entry = flow.entry(l).or_insert(CommunityFlow {
            node_probability: 0.0,
            exit_probability: 0.0,
            node_probabilities: Vec::new(),
        });

        let node_probability = working.degree[u] / two_m;
        entry.node_probability += node_probability;
        entry.node_probabilities.push(node_probability);

        for &(v, w) in &working.adj[u] {
            if labels[v] != l {
                entry.exit_probability += w / two_m;
            }
        }
    }

    flow
}

fn precompute_incremental(
    labels: &[usize],
    working: &WorkingGraph,
    two_m: f32,
) -> (Vec<CommunityStats>, Vec<IncomingWeightMap>) {
    let n = working.len();
    let max_label = labels.iter().copied().max().unwrap_or(0);

    let mut stats: Vec<CommunityStats> = vec![(0.0, 0.0, 0.0); max_label + 1];

    for (u, &l) in labels.iter().enumerate() {
        let p = working.degree[u] / two_m;
        let entry = &mut stats[l];
        entry.0 += p;
        entry.2 += entropy_term(p);
    }

    let mut incoming_to: Vec<IncomingWeightMap> = vec![HashMap::new(); n];
    for v in 0..n {
        let lv = labels[v];
        for &(u, w) in &working.adj[v] {
            *incoming_to[u].entry(lv).or_insert(0.0) += w;
            if labels[u] != lv {
                let entry = &mut stats[lv];
                entry.1 += w / two_m;
            }
        }
    }

    (stats, incoming_to)
}

fn infomap_lmdl_delta(
    labels: &[usize],
    old: usize,
    new: usize,
    u: usize,
    working: &WorkingGraph,
    two_m: f32,
    stats: &[CommunityStats],
    incoming_to: &[IncomingWeightMap],
) -> f32 {
    if old == new {
        return f32::INFINITY;
    }

    let (p_old, exit_old, _) = stats[old];
    let (p_new, exit_new, _) = stats[new];
    let p_u = working.degree[u] / two_m;

    let mut w_out_old = 0.0f32;
    let mut w_out_new = 0.0f32;
    let mut w_out_other = 0.0f32;

    for &(v, w) in &working.adj[u] {
        let label_v = labels[v];
        let w_m = w / two_m;
        match (label_v == old, label_v == new) {
            (true, _) => w_out_old += w_m,
            (_, true) => w_out_new += w_m,
            _ => w_out_other += w_m,
        }
    }

    let w_in_old = incoming_to[u].get(&old).copied().unwrap_or(0.0) / two_m;
    let w_in_new = incoming_to[u].get(&new).copied().unwrap_or(0.0) / two_m;

    let delta_exit_old = w_in_old - w_out_new - w_out_other;
    let delta_exit_new = w_out_old + w_out_other - w_in_new;
    let delta_q_total = delta_exit_old + delta_exit_new;

    let q_total: f32 = stats.iter().map(|s| s.1).sum();
    let q_total_after = q_total + delta_q_total;
    let q_old_before = p_old + exit_old;
    let q_old_after = (p_old - p_u) + (exit_old + delta_exit_old);
    let q_new_before = p_new + exit_new;
    let q_new_after = (p_new + p_u) + (exit_new + delta_exit_new);

    entropy_term(q_total_after)
        - entropy_term(q_total)
        - 2.0 * (entropy_term(exit_old + delta_exit_old) - entropy_term(exit_old))
        - 2.0 * (entropy_term(exit_new + delta_exit_new) - entropy_term(exit_new))
        + (entropy_term(q_old_after) - entropy_term(q_old_before))
        + (entropy_term(q_new_after) - entropy_term(q_new_before))
}

fn random_walk_init(working: &WorkingGraph) -> Vec<usize> {
    let n = working.len();
    let walk_steps = n.max(10) * 5;
    let walk_count = n.max(10);

    let mut rng = LcgRng::default();

    let degree: Vec<f32> = working
        .adj
        .iter()
        .zip(&working.self_loop)
        .map(|(e, sl)| e.iter().map(|(_, w)| *w).sum::<f32>() + 2.0 * sl)
        .collect();

    let mut visits = vec![0u64; n];
    for _ in 0..walk_count {
        let mut node = rng.gen_range(0, n);
        for _ in 0..walk_steps {
            visits[node] += 1;
            let total = degree[node];
            if total <= 0.0 {
                break;
            }
            let mut r = rng.gen_f32() * total;
            let mut next = node;
            for &(v, w) in &working.adj[node] {
                r -= w;
                if r <= 0.0 {
                    next = v;
                    break;
                }
            }
            node = next;
        }
    }

    let mut labels = Vec::with_capacity(n);
    for u in 0..n {
        let mut best_neighbor = u;
        let mut best_visits = visits[u];
        for &(v, _) in &working.adj[u] {
            if visits[v] > best_visits {
                best_visits = visits[v];
                best_neighbor = v;
            }
        }
        labels.push(best_neighbor);
    }

    labels
}

fn infomap_local_move(
    working: &WorkingGraph,
    labels: &[usize],
    two_m: f32,
    max_passes: usize,
) -> (Vec<usize>, f32) {
    let n = working.len();
    let mut current = labels.to_vec();
    let mut best_lmdl = compute_lmdl(working, &current, two_m);

    for _pass in 0..max_passes {
        let mut improved = false;

        let (stats, incoming_to) = precompute_incremental(&current, working, two_m);

        for u in 0..n {
            let old = current[u];

            let neighbor_comms: HashSet<usize> =
                working.adj[u].iter().map(|(v, _)| current[*v]).collect();
            let mut neighbor_comms: Vec<usize> = neighbor_comms.into_iter().collect();
            neighbor_comms.sort_unstable();

            let mut best_new = old;
            let mut best_delta = 0.0f32;

            for &cand in &neighbor_comms {
                if cand == old {
                    continue;
                }

                let delta = infomap_lmdl_delta(
                    &current,
                    old,
                    cand,
                    u,
                    working,
                    two_m,
                    &stats,
                    &incoming_to,
                );

                if delta < best_delta {
                    best_delta = delta;
                    best_new = cand;
                }
            }

            if best_new != old {
                current[u] = best_new;
                improved = true;
            }
        }

        if !improved {
            break;
        }

        best_lmdl = compute_lmdl(working, &current, two_m);
    }

    (current, best_lmdl)
}

fn infomap_refinement(
    working: &WorkingGraph,
    partition: &[usize],
    options: &CommunityOptions,
) -> Vec<usize> {
    let n = working.len();
    let two_m = 2.0 * working.total_weight;

    let mut by_parent: HashMap<usize, Vec<usize>> = HashMap::new();
    for (u, &c) in partition.iter().enumerate() {
        by_parent.entry(c).or_default().push(u);
    }

    let mut parents: Vec<(usize, Vec<usize>)> = by_parent.into_iter().collect();
    parents.sort_by_key(|(p, _)| *p);

    let mut label_base = Vec::with_capacity(parents.len());
    let mut cursor = 0usize;
    for (_, members) in &parents {
        label_base.push(cursor);
        cursor += members.len();
    }

    let total_labels = cursor.max(1);

    let mut refined: Vec<usize> = vec![total_labels; n];

    for (idx, (_, members)) in parents.iter().enumerate() {
        let base = label_base[idx];
        let parent_total: f32 = members.iter().map(|&i| working.degree[i]).sum();

        if members.len() <= 1 {
            refined[members[0]] = base;
            continue;
        }

        let member_set: HashSet<usize> = members.iter().copied().collect();
        let mut local_labels: HashMap<usize, usize> = members
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, base + i))
            .collect();

        let label_degree: HashMap<usize, f32> = members
            .iter()
            .enumerate()
            .map(|(i, &u)| (base + i, working.degree[u]))
            .collect();

        for _pass in 0..options.max_passes {
            let mut moved = false;

            for &u in members {
                let current = *local_labels.get(&u).unwrap();
                let nd = working.degree[u];

                if nd == 0.0 {
                    continue;
                }

                let mut weight_to_comm: HashMap<usize, f32> = HashMap::new();
                for &(v, w) in &working.adj[u] {
                    if !member_set.contains(&v) {
                        continue;
                    }
                    *weight_to_comm
                        .entry(*local_labels.get(&v).unwrap())
                        .or_insert(0.0) += w;
                }

                let mut best = current;
                let mut best_gain = options.min_modularity_gain;

                let stay_weight = weight_to_comm.get(&current).copied().unwrap_or(0.0);
                if stay_weight > best_gain {
                    best_gain = stay_weight;
                    best = current;
                }

                for (&candidate, &edge_weight) in &weight_to_comm {
                    if candidate == current {
                        continue;
                    }

                    let cand_degree = *label_degree.get(&candidate).unwrap_or(&0.0);
                    let threshold = if parent_total > 0.0 {
                        options.well_connectedness * cand_degree * (parent_total - cand_degree)
                            / (two_m * parent_total)
                    } else {
                        0.0
                    };

                    if edge_weight < threshold {
                        continue;
                    }

                    if edge_weight > best_gain {
                        best_gain = edge_weight;
                        best = candidate;
                    }
                }

                local_labels.insert(u, best);
                if best != current {
                    moved = true;
                }
            }

            if !moved {
                break;
            }
        }

        for &u in members {
            refined[u] = *local_labels.get(&u).unwrap();
        }
    }

    enforce_connected(working, &mut refined);
    densify_labels(&refined)
}

// ============================================================
// Multi-level algorithms (mirrors Ariadne's run_multilevel_* exactly)
// ============================================================

/// Run multi-level Louvain — mirrors Ariadne's run_multilevel_louvain.
#[pyfunction]
pub fn community_detection_louvain(
    py: Python,
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
    options: Option<&CommunityOptions>,
) -> PyResult<PyObject> {
    let options = options.unwrap_or(&CommunityOptions {
        resolution: 1.0,
        max_passes: 50,
        max_levels: 10,
        well_connectedness: 1.0,
        min_modularity_gain: 1e-7,
    });

    let mut working = WorkingGraph::from_py_data(nodes, edges, 0.15)?;

    if working.total_weight <= 0.0 {
        let dict = PyDict::new(py);
        for (i, node) in nodes.iter().enumerate() {
            let name: String = node.extract()?;
            dict.set_item(name, i)?;
        }
        return Ok(dict.into());
    }

    // Initialize: each original node maps to its own community
    let mut current: HashMap<String, usize> = working
        .members
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();

    for _level in 0..options.max_levels {
        let partition = local_move(&working, options);
        let distinct: HashSet<usize> = partition.iter().copied().collect();
        let moved = distinct.len() < working.len();

        let aggregation_partition = densify_labels(&partition);

        // Map every original node through its current super-node.
        for (super_idx, members) in working.members.iter().enumerate() {
            for id in members {
                current.insert(id.clone(), aggregation_partition[super_idx]);
            }
        }

        if !moved {
            break;
        }

        working = aggregate(&working, &aggregation_partition);
        if working.len() <= 1 {
            break;
        }
    }

    let final_labels = relabel(current);

    let dict = PyDict::new(py);
    for (node, label) in &final_labels {
        dict.set_item(node, label)?;
    }
    Ok(dict.into())
}

/// Run multi-level Leiden — mirrors Ariadne's run_multilevel_leiden.
#[pyfunction]
pub fn community_detection_leiden(
    py: Python,
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
    options: Option<&CommunityOptions>,
) -> PyResult<PyObject> {
    let options = options.unwrap_or(&CommunityOptions {
        resolution: 1.0,
        max_passes: 50,
        max_levels: 10,
        well_connectedness: 1.0,
        min_modularity_gain: 1e-7,
    });

    let mut working = WorkingGraph::from_py_data(nodes, edges, 0.15)?;

    if working.total_weight <= 0.0 {
        let dict = PyDict::new(py);
        for (i, node) in nodes.iter().enumerate() {
            let name: String = node.extract()?;
            dict.set_item(name, i)?;
        }
        return Ok(dict.into());
    }

    let mut current: HashMap<String, usize> = working
        .members
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();

    for _level in 0..options.max_levels {
        let partition = local_move(&working, options);
        let distinct: HashSet<usize> = partition.iter().copied().collect();
        let moved = distinct.len() < working.len();

        let aggregation_partition = refinement_phase(&working, &partition, options);

        for (super_idx, members) in working.members.iter().enumerate() {
            for id in members {
                current.insert(id.clone(), aggregation_partition[super_idx]);
            }
        }

        if !moved {
            break;
        }

        working = aggregate(&working, &aggregation_partition);
        if working.len() <= 1 {
            break;
        }
    }

    let final_labels = relabel(current);

    let dict = PyDict::new(py);
    for (node, label) in &final_labels {
        dict.set_item(node, label)?;
    }
    Ok(dict.into())
}

/// Run multi-level Infomap — mirrors Ariadne's run_infomap_multilevel.
#[pyfunction]
pub fn community_detection_infomap(
    py: Python,
    nodes: &Bound<'_, PyList>,
    edges: &Bound<'_, PyList>,
    options: Option<&CommunityOptions>,
) -> PyResult<PyObject> {
    let options = options.unwrap_or(&CommunityOptions {
        resolution: 1.0,
        max_passes: 50,
        max_levels: 10,
        well_connectedness: 1.0,
        min_modularity_gain: 1e-7,
    });

    let mut working = WorkingGraph::from_py_data(nodes, edges, 0.15)?;

    if working.total_weight <= 0.0 {
        let dict = PyDict::new(py);
        for (i, node) in nodes.iter().enumerate() {
            let name: String = node.extract()?;
            dict.set_item(name, i)?;
        }
        return Ok(dict.into());
    }

    let original_working = working.clone();
    let two_m = 2.0 * original_working.total_weight;

    let mut current: HashMap<String, usize> = working
        .members
        .iter()
        .flatten()
        .enumerate()
        .map(|(i, id)| (id.clone(), i))
        .collect();

    let mut best_mapping = current.clone();
    let original_labels: Vec<usize> = original_working
        .members
        .iter()
        .flatten()
        .map(|id| *current.get(id).unwrap_or(&0))
        .collect();
    let mut best_lmdl = compute_lmdl(&original_working, &original_labels, two_m);

    for _level in 0..options.max_levels {
        if working.total_weight <= 0.0 || working.len() <= 1 {
            break;
        }

        let two_m_level = 2.0 * working.total_weight;

        // Random walk initialization
        let mut labels = random_walk_init(&working);

        // Greedy local move to minimize LMDL
        let mut prev_pass_lmdl = f32::INFINITY;
        for _pass in 0..options.max_passes {
            let (new_labels, lmdl) =
                infomap_local_move(&working, &labels, two_m_level, options.max_passes);
            labels = new_labels;

            let improved = (prev_pass_lmdl - lmdl).abs() > 1e-8;
            prev_pass_lmdl = lmdl;

            if !improved && _pass >= 2 {
                break;
            }
        }

        // Leiden-style refinement
        let aggregation_partition: Vec<usize> = if options.well_connectedness > 0.0 {
            infomap_refinement(&working, &labels, options)
        } else {
            densify_labels(&labels)
        };

        // Check if partition changed
        let moved = aggregation_partition
            .iter()
            .enumerate()
            .any(|(i, &l)| l != labels[i]);

        // Update original-node → super-node mapping
        let mut candidate_mapping = current.clone();
        for label in candidate_mapping.values_mut() {
            *label = aggregation_partition[*label];
        }

        // Compute LMDL on the candidate mapping
        let candidate_labels: Vec<usize> = original_working
            .members
            .iter()
            .flatten()
            .map(|id| *candidate_mapping.get(id).unwrap_or(&0))
            .collect();
        let candidate_lmdl = compute_lmdl(&original_working, &candidate_labels, two_m);

        if candidate_lmdl + 1e-6 < best_lmdl {
            best_lmdl = candidate_lmdl;
            best_mapping = candidate_mapping.clone();
            current = candidate_mapping;
            if !moved {
                break;
            }
        } else {
            break;
        }

        if !moved {
            break;
        }

        working = aggregate(&working, &aggregation_partition);
        if working.len() <= 1 {
            break;
        }
    }

    let final_labels = relabel(best_mapping);

    let dict = PyDict::new(py);
    for (node, label) in &final_labels {
        dict.set_item(node, label)?;
    }
    Ok(dict.into())
}

#[pymodule]
fn analysis(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CommunityOptions>()?;
    m.add_function(wrap_pyfunction!(community_detection_louvain, &m)?)?;
    m.add_function(wrap_pyfunction!(community_detection_leiden, &m)?)?;
    m.add_function(wrap_pyfunction!(community_detection_infomap, &m)?)?;
    m.add_function(wrap_pyfunction!(densify, &m)?)?;
    m.add(
        "__doc__",
        "Community detection algorithms ported from Ariadne",
    )?;
    Ok(())
}

#[pyfunction]
fn densify(labels: Vec<usize>) -> Vec<usize> {
    densify_labels(&labels)
}
