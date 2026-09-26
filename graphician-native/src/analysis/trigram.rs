//! Trigram-based candidate prefilter for fast graph search.

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Extract character trigrams from text.
pub fn trigrams(text: &str) -> HashSet<String> {
    if text.len() < 3 {
        if text.is_empty() {
            return HashSet::new();
        }
        return std::iter::once(text.to_string()).collect();
    }
    text
        .as_bytes()
        .windows(3)
        .map(|w| String::from_utf8_lossy(w).into_owned())
        .collect()
}

/// Build a trigram → node-index postings map from searchable texts.
#[pyfunction]
pub fn build_trigram_index(node_texts: Vec<String>) -> (Vec<String>, HashMap<String, Vec<usize>>) {
    let mut postings: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, text) in node_texts.iter().enumerate() {
        for gram in trigrams(text) {
            postings.entry(gram).or_default().push(i);
        }
    }
    (node_texts, postings)
}

/// Find candidate node indices whose searchable text could contain any needle.
///
/// Uses trigram intersection (smallest-first) to narrow the candidate set.
/// Returns None when the index isn't selective enough (fallback to full scan).
#[pyfunction]
pub fn trigram_candidates(
    node_texts: Vec<String>,
    postings: HashMap<String, Vec<usize>>,
    needles: Vec<String>,
    guard_frac: f64,
) -> Option<Vec<usize>> {
    let n = node_texts.len();
    if n == 0 || needles.is_empty() {
        return Some(vec![]);
    }

    let thresh = (n as f64 * guard_frac).ceil() as usize;

    // Check if any needle is too short or has common trigrams
    for s in &needles {
        let s_lower = s.to_lowercase();
        let tgs = trigrams(&s_lower);
        if tgs.is_empty() {
            continue;
        }
        // Check if rarest trigram is still too common
        let min_count = tgs
            .iter()
            .filter_map(|g| postings.get(g.as_str()).map(|v| v.len()))
            .min();
        if let Some(min_count) = min_count {
            if min_count > thresh {
                return None;
            }
        }
    }

    // Intersect posting lists per needle (smallest-first)
    let mut cand: HashSet<usize> = HashSet::new();
    for s in &needles {
        let s_lower = s.to_lowercase();
        let tgs = trigrams(&s_lower);
        if tgs.is_empty() {
            continue;
        }

        // Collect posting lists and sort by size (smallest first)
        let mut posting_lists: Vec<&Vec<usize>> = tgs
            .iter()
            .filter_map(|g| postings.get(g.as_str()))
            .collect();
        posting_lists.sort_by_key(|list| list.len());

        if posting_lists.is_empty() {
            continue;
        }

        // Intersect smallest-first
        let mut hit: HashSet<usize> = posting_lists[0].iter().cloned().collect();
        for other in &posting_lists[1..] {
            hit.retain(|x| other.contains(x));
            if hit.is_empty() {
                break;
            }
        }
        cand.extend(hit);
    }

    if cand.is_empty() {
        return Some(vec![]);
    }

    let mut result: Vec<usize> = cand.into_iter().collect();
    result.sort_unstable();
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigrams_extract_correctly() {
        assert_eq!(trigrams("").len(), 0);
        assert_eq!(trigrams("ab").len(), 1);
        assert_eq!(trigrams("abc"), vec!["abc".to_string()].into_iter().collect());
        assert_eq!(
            trigrams("abcd"),
            vec!["abc".to_string(), "bcd".to_string()].into_iter().collect()
        );
    }

    #[test]
    fn trigram_candidates_intersects() {
        let texts = vec![
            "cache manager".to_string(),
            "cache service".to_string(),
            "database pool".to_string(),
        ];
        let (_, postings) = build_trigram_index(texts.clone());

        // "cache" has trigrams: cac, aca, che
        let result = trigram_candidates(texts, postings, vec!["cache".to_string()], 0.1).unwrap();
        // Should match indices 0 and 1 (both contain "cache")
        assert_eq!(result.len(), 2);
        assert!(result.contains(&0));
        assert!(result.contains(&1));
    }
}
