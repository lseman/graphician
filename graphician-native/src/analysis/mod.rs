//! Community detection algorithms ported from Ariadne.
//!
//! This module implements Louvain, Leiden, and Infomap community detection
//! with proper edge weights, multi-level aggregation, and Leiden-style refinement.
//! All algorithms are in core.rs with PyO3 bindings.

pub mod centrality;
pub mod community_metrics;
pub mod core;
pub mod dedup;
pub mod motifs;
pub mod search;
pub mod structure;
pub mod traversal;
