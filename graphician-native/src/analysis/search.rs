//! Batch fuzzy-string scoring for in-memory search.

use pyo3::prelude::*;
use std::collections::HashSet;

#[pyfunction]
pub fn fuzzy_score_matrix(queries: Vec<String>, targets: Vec<String>) -> Vec<Vec<f64>> {
    queries
        .iter()
        .map(|query| {
            targets
                .iter()
                .map(|target| fuzzy_score(query, target))
                .collect()
        })
        .collect()
}

fn fuzzy_score(query: &str, target: &str) -> f64 {
    if query.is_empty() || target.is_empty() {
        return 0.0;
    }
    let compact_query = compact(query);
    let compact_target = compact(target);
    [
        ratio(query, target),
        ratio(&compact_query, &compact_target),
        partial_ratio(&compact_query, &compact_target),
        token_sort_ratio(query, target),
        token_set_ratio(query, target),
        acronym_ratio(query, target),
        subsequence_ratio(&compact_query, &compact_target),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

fn chars(value: &str) -> Vec<char> {
    value.chars().collect()
}

fn levenshtein(left: &[char], right: &[char]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (left_index, left_char) in left.iter().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right.iter().enumerate() {
            current[right_index + 1] = (previous[right_index + 1] + 1)
                .min(current[right_index] + 1)
                .min(previous[right_index] + usize::from(left_char != right_char));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

fn ratio(left: &str, right: &str) -> f64 {
    let left = chars(left);
    let right = chars(right);
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    1.0 - levenshtein(&left, &right) as f64 / left.len().max(right.len()) as f64
}

fn partial_ratio(left: &str, right: &str) -> f64 {
    let left = chars(left);
    let right = chars(right);
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let (needle, haystack) = if left.len() <= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    if needle.len() >= haystack.len() {
        return 1.0
            - levenshtein(needle, haystack) as f64 / needle.len().max(haystack.len()) as f64;
    }
    haystack
        .windows(needle.len())
        .map(|window| 1.0 - levenshtein(needle, window) as f64 / needle.len() as f64)
        .fold(0.0, f64::max)
}

fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn sorted_tokens(value: &str) -> Vec<&str> {
    let mut tokens: Vec<_> = value.split_whitespace().collect();
    tokens.sort_unstable();
    tokens
}

fn token_sort_ratio(left: &str, right: &str) -> f64 {
    ratio(
        &sorted_tokens(left).join(" "),
        &sorted_tokens(right).join(" "),
    )
}

fn token_set_ratio(left: &str, right: &str) -> f64 {
    let left_tokens = unique_sorted_tokens(left);
    let right_tokens = unique_sorted_tokens(right);
    let right_set: HashSet<_> = right_tokens.iter().copied().collect();
    let common = left_tokens
        .iter()
        .filter(|token| right_set.contains(**token))
        .copied()
        .collect::<Vec<_>>()
        .join(" ");
    if common.is_empty() {
        0.0
    } else {
        ratio(&common, left).max(ratio(&common, right))
    }
}

fn unique_sorted_tokens(value: &str) -> Vec<&str> {
    let mut seen = HashSet::new();
    sorted_tokens(value)
        .into_iter()
        .filter(|token| seen.insert(*token))
        .collect()
}

fn acronym_ratio(query: &str, candidate: &str) -> f64 {
    let acronym: String = candidate
        .split_whitespace()
        .filter_map(|token| token.chars().next())
        .collect();
    ratio(&compact(query), &acronym)
}

fn subsequence_ratio(query: &str, candidate: &str) -> f64 {
    let query = chars(query);
    let candidate = chars(candidate);
    let mut query_index = 0;
    for character in &candidate {
        if query.get(query_index) == Some(character) {
            query_index += 1;
            if query_index == query.len() {
                return query.len() as f64 / candidate.len().max(1) as f64;
            }
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_exact_partial_and_acronym_matches() {
        assert_eq!(fuzzy_score("request handler", "request handler"), 1.0);
        assert_eq!(fuzzy_score("handler", "requesthandler"), 1.0);
        assert!(fuzzy_score("rh", "request handler") > 0.9);
    }
}
