//! Native MinHash/LSH candidate generation for node deduplication.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const MINHASH_PRIME: u64 = 65_521;

#[derive(Debug)]
struct MinHash {
    a: Vec<u64>,
    b: Vec<u64>,
    signature: Vec<u32>,
}

impl MinHash {
    fn new(num_permutations: usize) -> Self {
        let mut state = 0x6c62_272e_07bb_0142;
        let mut seeds = Vec::with_capacity(num_permutations * 2);
        for _ in 0..num_permutations * 2 {
            seeds.push(splitmix64(&mut state));
        }
        let a = (0..num_permutations)
            .map(|index| seeds[index] ^ index as u64)
            .collect();
        let b = (0..num_permutations)
            .map(|index| seeds[index] ^ (index as u64).wrapping_mul(31))
            .collect();
        Self {
            a,
            b,
            signature: vec![u32::MAX; num_permutations],
        }
    }

    fn from_label(label: &str, shingle_size: usize, num_permutations: usize) -> Self {
        let mut minhash = Self::new(num_permutations);
        for item in shingles(label, shingle_size) {
            minhash.update(item.as_bytes());
        }
        minhash
    }

    fn update(&mut self, data: &[u8]) {
        for index in 0..self.signature.len() {
            let h1 = fnv1a_with_salt(data, self.a[index]);
            let h2 = fnv1a_with_salt(data, self.b[index]);
            let hash = ((u128::from(self.a[index]) * u128::from(h1 % MINHASH_PRIME)
                + u128::from(self.b[index])
                + u128::from(h2))
                % u128::from(MINHASH_PRIME)) as u64;
            self.signature[index] = self.signature[index].min(hash as u32);
        }
    }

    fn jaccard(&self, other: &Self) -> f64 {
        if self.signature.is_empty() {
            return 0.0;
        }
        let matches = self
            .signature
            .iter()
            .zip(&other.signature)
            .filter(|(left, right)| left == right)
            .count();
        matches as f64 / self.signature.len() as f64
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fnv1a_with_salt(data: &[u8], salt: u64) -> u64 {
    data.iter().fold(salt, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn shingles(label: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = label.chars().collect();
    if chars.len() < size {
        return vec![label.to_owned()];
    }
    chars
        .windows(size)
        .map(|window| window.iter().collect())
        .collect()
}

/// Return MinHash/LSH candidate pairs with the same parameters as the Python pipeline.
#[pyfunction]
#[pyo3(signature = (labels, shingle_size, num_permutations, num_bands, row_length, jaccard_threshold))]
pub fn dedup_candidate_pairs(
    labels: Vec<(u64, String)>,
    shingle_size: usize,
    num_permutations: usize,
    num_bands: usize,
    row_length: usize,
    jaccard_threshold: f64,
) -> PyResult<Vec<(u64, u64, f64)>> {
    if shingle_size == 0 {
        return Err(PyValueError::new_err(
            "shingle_size must be greater than zero",
        ));
    }
    if num_permutations == 0 {
        return Err(PyValueError::new_err(
            "num_permutations must be greater than zero",
        ));
    }
    if num_bands == 0 || row_length == 0 {
        return Err(PyValueError::new_err(
            "num_bands and row_length must be greater than zero",
        ));
    }
    let unique_ids: HashSet<u64> = labels.iter().map(|(node_id, _)| *node_id).collect();
    if unique_ids.len() != labels.len() {
        return Err(PyValueError::new_err("node IDs must be unique"));
    }

    let signatures: HashMap<u64, MinHash> = labels
        .into_iter()
        .map(|(node_id, label)| {
            (
                node_id,
                MinHash::from_label(&label, shingle_size, num_permutations),
            )
        })
        .collect();
    let mut tables: Vec<HashMap<Vec<u32>, Vec<u64>>> =
        (0..num_bands).map(|_| HashMap::new()).collect();
    for (&node_id, signature) in &signatures {
        for (band, table) in tables.iter_mut().enumerate() {
            let start = band * row_length;
            let end = start + row_length;
            if end <= signature.signature.len() {
                table
                    .entry(signature.signature[start..end].to_vec())
                    .or_default()
                    .push(node_id);
            }
        }
    }

    let mut seen = HashSet::new();
    let mut pairs = Vec::new();
    for (&left_id, left_signature) in &signatures {
        let mut candidates = HashSet::new();
        for (band, table) in tables.iter().enumerate() {
            let start = band * row_length;
            let end = start + row_length;
            if end <= left_signature.signature.len() {
                if let Some(ids) = table.get(&left_signature.signature[start..end]) {
                    candidates.extend(ids.iter().copied());
                }
            }
        }
        for right_id in candidates {
            if left_id == right_id {
                continue;
            }
            let pair = if left_id < right_id {
                (left_id, right_id)
            } else {
                (right_id, left_id)
            };
            if !seen.insert(pair) {
                continue;
            }
            let score = left_signature.jaccard(&signatures[&right_id]);
            if score >= jaccard_threshold {
                pairs.push((pair.0, pair.1, score));
            }
        }
    }
    pairs.sort_unstable_by_key(|(left, right, _)| (*left, *right));
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_labels_are_candidates() {
        let pairs = dedup_candidate_pairs(
            vec![
                (7, "database connection".into()),
                (9, "database connection".into()),
            ],
            3,
            64,
            12,
            5,
            0.7,
        )
        .unwrap();
        assert_eq!(pairs, vec![(7, 9, 1.0)]);
    }

    #[test]
    fn validates_parameters_and_ids() {
        assert!(dedup_candidate_pairs(vec![(1, "a".into())], 0, 8, 2, 4, 0.5).is_err());
        assert!(
            dedup_candidate_pairs(vec![(1, "a".into()), (1, "b".into())], 1, 8, 2, 4, 0.5,)
                .is_err()
        );
    }
}
