//! Whole-graph symbol-resolution planning.
//!
//! Native passes return mutation plans rather than mutating Python objects.
//! This keeps Graphician's existing graph interface authoritative while the
//! indexing and graph-wide scans run in Rust.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};

type NodeRecord = (u64, String, String, String);
type EdgeRecord = (u64, u64, u64, String);
type TypeRewire = (u64, u64, u64, String);

type CallNodeRecord = (u64, String, String, String, Option<String>);
type CallEdgeRecord = (
    u64,
    u64,
    u64,
    String,
    String,
    Option<String>,
    Option<String>,
    bool,
);
type ImportedBinding = (String, String, String, String);
type CallResolution = (u64, u64, u64, String, bool);

/// Plan rewrites from `type::<Name>` placeholders to unique real types.
///
/// Nodes are `(id, kind, name, qualified_name)` and edges are
/// `(edge_id, source, target, kind)`. The result contains rewires as
/// `(edge_id, source, replacement_target, kind)` plus placeholder IDs that
/// become isolated once those edges are removed.
#[pyfunction]
pub fn plan_type_resolution(
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
) -> (Vec<TypeRewire>, Vec<u64>) {
    let mut real_types: HashMap<String, Vec<u64>> = HashMap::new();
    let mut placeholders = Vec::new();

    for (id, kind, name, qualified_name) in nodes {
        if kind == "class" && qualified_name.starts_with("type::") {
            placeholders.push((id, qualified_name[6..].to_owned()));
        } else if kind == "class" || kind == "trait" {
            real_types.entry(name).or_default().push(id);
        }
    }

    let replacement_by_placeholder: HashMap<u64, u64> = placeholders
        .iter()
        .filter_map(
            |(placeholder, name)| match real_types.get(name).map(Vec::as_slice) {
                Some([replacement]) if replacement != placeholder => {
                    Some((*placeholder, *replacement))
                }
                _ => None,
            },
        )
        .collect();

    let mut rewires = Vec::new();
    let mut removed_edge_ids = HashSet::new();
    for (edge_id, source, target, kind) in &edges {
        if kind != "inherits" && kind != "implements" {
            continue;
        }
        if let Some(&replacement) = replacement_by_placeholder.get(target) {
            rewires.push((*edge_id, *source, replacement, kind.clone()));
            removed_edge_ids.insert(*edge_id);
        }
    }

    let orphaned = placeholders
        .into_iter()
        .filter(|(placeholder, _)| {
            !edges.iter().any(|(edge_id, source, target, _)| {
                !removed_edge_ids.contains(edge_id)
                    && (source == placeholder || target == placeholder)
            })
        })
        .map(|(placeholder, _)| placeholder)
        .collect();

    (rewires, orphaned)
}

#[derive(Debug)]
struct CallNode {
    kind: String,
    name: String,
    qualified_name: String,
    source_uri: Option<String>,
}

fn common_prefix_len(left: &str, right: &str) -> usize {
    left.split("::")
        .zip(right.split("::"))
        .take_while(|(left, right)| left == right)
        .count()
}

fn module_stem(uri: &str) -> Option<String> {
    let path = Path::new(uri);
    let stem = path.file_stem()?.to_str()?;
    if matches!(stem, "mod" | "index" | "__init__" | "lib" | "main") {
        path.parent()?.file_name()?.to_str().map(str::to_owned)
    } else {
        Some(stem.to_owned())
    }
}

fn normalized_without_extension(path: &str) -> PathBuf {
    let mut value = PathBuf::from(path);
    value.set_extension("");
    if value.file_name().and_then(|part| part.to_str()) == Some("__init__") {
        value.pop();
    }
    value
}

fn normalized_parts(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_owned),
            Component::ParentDir => Some("..".to_owned()),
            Component::CurDir => Some(".".to_owned()),
            Component::RootDir | Component::Prefix(_) => None,
        })
        .collect()
}

fn module_matches_source(module: &str, caller_uri: &str, candidate_uri: &str) -> bool {
    let candidate = normalized_without_extension(candidate_uri);
    let dots = module.len() - module.trim_start_matches('.').len();
    let module_parts: Vec<_> = module
        .trim_start_matches('.')
        .split('.')
        .filter(|part| !part.is_empty())
        .collect();

    if dots > 0 {
        let mut expected = PathBuf::from(caller_uri);
        expected.pop();
        for _ in 0..dots.saturating_sub(1) {
            expected.pop();
        }
        expected.extend(&module_parts);
        let expected_parts = normalized_parts(&expected);
        let candidate_parts = normalized_parts(&candidate);
        return candidate_parts.starts_with(&expected_parts);
    }
    if module_parts.is_empty() {
        return false;
    }
    let candidate_parts = normalized_parts(&candidate);
    candidate_parts.windows(module_parts.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(module_parts.iter().copied())
    })
}

/// Plan all call-placeholder rewrites using Graphician's seven resolution tiers.
///
/// Edge records contain `(id, src, dst, kind, confidence, scope,
/// receiver_type, suppressed_name)`. Receiver type inference remains in the
/// Python adapter because it consumes language-specific source text; all
/// graph-wide indexing, selection, ranking, and deduplication happen here.
#[pyfunction]
pub fn plan_call_resolution(
    nodes: Vec<CallNodeRecord>,
    edges: Vec<CallEdgeRecord>,
    import_tokens: Vec<(String, Vec<String>)>,
    imported_bindings: Vec<ImportedBinding>,
) -> (Vec<CallResolution>, Vec<u64>) {
    let node_order: Vec<u64> = nodes.iter().map(|record| record.0).collect();
    let nodes: HashMap<u64, CallNode> = nodes
        .into_iter()
        .map(|(id, kind, name, qualified_name, source_uri)| {
            (
                id,
                CallNode {
                    kind,
                    name,
                    qualified_name,
                    source_uri,
                },
            )
        })
        .collect();
    let mut by_name: HashMap<String, Vec<u64>> = HashMap::new();
    for id in node_order {
        let node = &nodes[&id];
        if matches!(node.kind.as_str(), "function" | "method" | "class" | "type")
            && !node.qualified_name.starts_with("call::")
        {
            by_name.entry(node.name.clone()).or_default().push(id);
        }
    }

    let import_tokens: HashMap<String, HashSet<String>> = import_tokens
        .into_iter()
        .map(|(uri, values)| (uri, values.into_iter().collect()))
        .collect();
    let mut bindings: HashMap<(String, String), Vec<(String, String)>> = HashMap::new();
    for (uri, local, module, original) in imported_bindings {
        bindings
            .entry((uri, local))
            .or_default()
            .push((module, original));
    }

    let mut existing_calls = HashSet::new();
    let mut incoming_call_count: HashMap<u64, usize> = HashMap::new();
    for (_, source, target, kind, confidence, _, _, _) in &edges {
        if kind == "calls" {
            *incoming_call_count.entry(*target).or_default() += 1;
            if confidence != "ambiguous" {
                existing_calls.insert((*source, *target));
            }
        }
    }

    let mut resolutions = Vec::new();
    let mut stale = Vec::new();
    for (edge_id, source, target, kind, _, scope, receiver_type, suppressed) in &edges {
        if kind != "calls" {
            continue;
        }
        let Some(target_node) = nodes.get(target) else {
            continue;
        };
        let Some(bare) = target_node.qualified_name.strip_prefix("call::") else {
            continue;
        };
        let source_node = nodes.get(source);
        let source_uri = source_node.and_then(|node| node.source_uri.as_deref());
        let imported = source_uri
            .and_then(|uri| bindings.get(&(uri.to_owned(), bare.to_owned())))
            .map(Vec::as_slice)
            .unwrap_or(&[]);

        let mut candidate_names = vec![bare];
        candidate_names.extend(imported.iter().map(|(_, original)| original.as_str()));
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();
        for name in candidate_names {
            if let Some(values) = by_name.get(name) {
                for &candidate in values {
                    if seen.insert(candidate) {
                        candidates.push(candidate);
                    }
                }
            }
        }
        if candidates.is_empty() {
            if *suppressed {
                stale.push(*edge_id);
            }
            continue;
        }

        let mut selected: Option<(u64, &str, bool)> = None;
        if candidates.len() == 1 && !suppressed {
            selected = Some((candidates[0], "unique_name", true));
        }
        if selected.is_none() {
            if let Some(uri) = source_uri {
                let local: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        nodes
                            .get(candidate)
                            .and_then(|node| node.source_uri.as_deref())
                            == Some(uri)
                    })
                    .collect();
                if let [candidate] = local.as_slice() {
                    selected = Some((*candidate, "file_local", true));
                }
            }
        }
        if selected.is_none() {
            if let Some(scope) = scope {
                let scoped: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        nodes
                            .get(candidate)
                            .is_some_and(|node| node.qualified_name.contains(scope))
                    })
                    .collect();
                if let [candidate] = scoped.as_slice() {
                    selected = Some((*candidate, "scoped", false));
                } else if scoped.len() > 1 {
                    let caller_qname = source_node.map_or("", |node| node.qualified_name.as_str());
                    let mut best = None;
                    let mut best_prefix = 0;
                    for candidate in scoped {
                        let prefix = common_prefix_len(
                            caller_qname,
                            &nodes.get(&candidate).unwrap().qualified_name,
                        );
                        if best.is_none() || prefix > best_prefix {
                            best = Some(candidate);
                            best_prefix = prefix;
                        }
                    }
                    selected = best.map(|candidate| (candidate, "scoped_prefix", false));
                }
            }
        }
        if selected.is_none() {
            if let Some(receiver_type) = receiver_type {
                let receiver_candidates: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        nodes.get(candidate).is_some_and(|node| {
                            node.qualified_name
                                .rsplit("::")
                                .nth(1)
                                .is_some_and(|owner| owner == receiver_type)
                        })
                    })
                    .collect();
                if let [candidate] = receiver_candidates.as_slice() {
                    selected = Some((*candidate, "receiver", false));
                }
            }
        }
        if selected.is_none() {
            if let Some(uri) = source_uri {
                let imported_by_name: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        let Some(candidate_node) = nodes.get(candidate) else {
                            return false;
                        };
                        let Some(candidate_uri) = candidate_node.source_uri.as_deref() else {
                            return false;
                        };
                        imported.iter().any(|(module, original)| {
                            candidate_node.name == *original
                                && module_matches_source(module, uri, candidate_uri)
                        })
                    })
                    .collect();
                if let [candidate] = imported_by_name.as_slice() {
                    selected = Some((*candidate, "imported_symbol", false));
                }
            }
        }
        if selected.is_none() {
            if let Some(tokens) = source_uri.and_then(|uri| import_tokens.get(uri)) {
                let imported_candidates: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        nodes
                            .get(candidate)
                            .and_then(|node| node.source_uri.as_deref())
                            .and_then(module_stem)
                            .is_some_and(|stem| tokens.contains(&stem.to_lowercase()))
                    })
                    .collect();
                if let [candidate] = imported_candidates.as_slice() {
                    selected = Some((*candidate, "import_scoped", false));
                }
            }
        }
        if selected.is_none() {
            if let Some(uri) = source_uri {
                let source_dir = Path::new(uri).parent();
                let same_dir: Vec<_> = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        nodes
                            .get(candidate)
                            .and_then(|node| node.source_uri.as_deref())
                            .and_then(|candidate_uri| Path::new(candidate_uri).parent())
                            == source_dir
                    })
                    .collect();
                if let [candidate] = same_dir.as_slice() {
                    selected = Some((*candidate, "same_dir", false));
                }
            }
        }
        if selected.is_none() {
            if *suppressed {
                stale.push(*edge_id);
                continue;
            }
            let scored: Vec<_> = candidates
                .iter()
                .copied()
                .map(|c| {
                    (
                        c,
                        incoming_call_count.get(&c).copied().unwrap_or(0),
                    )
                })
                .collect();
            let max_score = scored
                .iter()
                .map(|(_, s)| *s)
                .max()
                .unwrap_or(0);
            // Require: (1) at least 2 incoming calls to be considered
            // popular enough, (2) winner must have at least 2x the
            // runner-up to avoid picking stdlib functions that barely
            // edge out project functions with the same name.
            if max_score >= 2 {
                let runners_up: Vec<_> = scored
                    .iter()
                    .copied()
                    .filter(|(_, s)| *s < max_score)
                    .map(|(c, _)| c)
                    .collect();
                let runner_up_score = runners_up
                    .iter()
                    .find_map(|c| incoming_call_count.get(c).copied())
                    .unwrap_or(0);
                if max_score >= runner_up_score * 2 {
                    let winners: Vec<_> = scored
                        .iter()
                        .copied()
                        .filter(|(_, s)| *s == max_score)
                        .map(|(c, _)| c)
                        .collect();
                    if let [winner] = winners.as_slice() {
                        selected = Some((*winner, "freq_prior", false));
                    }
                }
            }
        }

        if let Some((candidate, tag, structural)) = selected {
            stale.push(*edge_id);
            if !existing_calls.contains(&(*source, candidate)) {
                resolutions.push((*edge_id, *source, candidate, tag.to_owned(), structural));
            }
        }
    }
    (resolutions, stale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_unique_type_rewire_and_orphan_removal() {
        let nodes = vec![
            (1, "class".into(), "Gateway".into(), "type::Gateway".into()),
            (2, "class".into(), "Stripe".into(), "pkg::Stripe".into()),
            (3, "trait".into(), "Gateway".into(), "pkg::Gateway".into()),
        ];
        let edges = vec![(10, 2, 1, "implements".into())];

        let (rewires, orphaned) = plan_type_resolution(nodes, edges);

        assert_eq!(rewires, vec![(10, 2, 3, "implements".into())]);
        assert_eq!(orphaned, vec![1]);
    }

    #[test]
    fn preserves_ambiguous_and_non_supertype_placeholders() {
        let nodes = vec![
            (1, "class".into(), "Base".into(), "type::Base".into()),
            (2, "class".into(), "Child".into(), "pkg::Child".into()),
            (3, "class".into(), "Base".into(), "a::Base".into()),
            (4, "trait".into(), "Base".into(), "b::Base".into()),
        ];
        let edges = vec![(10, 2, 1, "defines".into())];

        let (rewires, orphaned) = plan_type_resolution(nodes, edges);

        assert!(rewires.is_empty());
        assert!(orphaned.is_empty());
    }
}
