//! SQLite persistence kernels exposed through Graphician's Python adapter.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::collections::HashSet;

type PersistNode = (
    u64,
    String,
    String,
    String,
    Option<String>,
    Option<i64>,
    Option<i64>,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);
type PersistEdge = (
    u64,
    u64,
    String,
    f64,
    String,
    String,
    Option<String>,
    Option<String>,
);
type LoadedNode = (
    u64,
    String,
    String,
    String,
    Option<String>,
    Option<i64>,
    Option<i64>,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);
type LoadedEdge = (
    u64,
    u64,
    String,
    f64,
    String,
    String,
    Option<String>,
    Option<String>,
);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct EdgeKey {
    source: i64,
    target: i64,
    kind: String,
    confidence_bits: u64,
    confidence_class: String,
    properties: String,
    valid_from: Option<String>,
    valid_to: Option<String>,
}

#[derive(Debug)]
struct StoredNode {
    id: i64,
    kind: String,
    name: String,
    source_uri: Option<String>,
    line_start: Option<i64>,
    line_end: Option<i64>,
    properties: String,
    source_text: Option<String>,
    valid_from: Option<String>,
    valid_to: Option<String>,
}

fn runtime_error(error: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(format!("native SQLite persistence failed: {error}"))
}

/// Atomically replace the active graph in a canonical Graphician database.
#[pyfunction]
pub fn save_graph_sqlite(
    path: &str,
    nodes: Vec<PersistNode>,
    edges: Vec<PersistEdge>,
    file_hashes: Option<Vec<(String, String)>>,
    now_iso: &str,
    now_unix: i64,
) -> PyResult<()> {
    let mut connection = Connection::open(path).map_err(runtime_error)?;
    connection
        .busy_timeout(std::time::Duration::from_secs(30))
        .map_err(runtime_error)?;
    let transaction = connection.transaction().map_err(runtime_error)?;
    transaction
        .execute("DELETE FROM edges", [])
        .map_err(runtime_error)?;
    transaction
        .execute("DELETE FROM nodes", [])
        .map_err(runtime_error)?;

    let mut graph_to_database = HashMap::with_capacity(nodes.len());
    {
        let mut statement = transaction
            .prepare(
                "INSERT INTO nodes
                 (kind, name, qualified_name, source_uri, line_start, line_end,
                  properties, source_text, valid_from, valid_to)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .map_err(runtime_error)?;
        for (
            graph_id,
            kind,
            name,
            qualified_name,
            source_uri,
            line_start,
            line_end,
            properties,
            source_text,
            valid_from,
            valid_to,
        ) in nodes
        {
            statement
                .execute(params![
                    kind,
                    name,
                    qualified_name,
                    source_uri,
                    line_start,
                    line_end,
                    properties,
                    source_text,
                    valid_from,
                    valid_to,
                ])
                .map_err(runtime_error)?;
            graph_to_database.insert(graph_id, transaction.last_insert_rowid());
        }
    }

    {
        let mut statement = transaction
            .prepare(
                "INSERT INTO edges
                 (kind, confidence, conf_class, properties, valid_from, valid_to,
                  src_id, dst_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            )
            .map_err(runtime_error)?;
        for (
            source,
            target,
            kind,
            confidence,
            confidence_class,
            properties,
            valid_from,
            valid_to,
        ) in edges
        {
            let source_id = graph_to_database.get(&source).ok_or_else(|| {
                PyRuntimeError::new_err(format!("edge source {source} is missing"))
            })?;
            let target_id = graph_to_database.get(&target).ok_or_else(|| {
                PyRuntimeError::new_err(format!("edge target {target} is missing"))
            })?;
            statement
                .execute(params![
                    kind,
                    confidence,
                    confidence_class,
                    properties,
                    valid_from,
                    valid_to,
                    source_id,
                    target_id,
                ])
                .map_err(runtime_error)?;
        }
    }

    transaction
        .execute("DELETE FROM nodes_fts", [])
        .map_err(runtime_error)?;
    transaction
        .execute(
            "INSERT INTO nodes_fts (rowid, kind, name, qualified_name)
             SELECT id, kind, name, qualified_name FROM nodes",
            [],
        )
        .map_err(runtime_error)?;
    if let Some(file_hashes) = file_hashes {
        transaction
            .execute("DELETE FROM file_state", [])
            .map_err(runtime_error)?;
        let mut statement = transaction
            .prepare(
                "INSERT OR REPLACE INTO file_state (path, hash, indexed_at_unix)
                 VALUES (?, ?, ?)",
            )
            .map_err(runtime_error)?;
        for (path, hash) in file_hashes {
            statement
                .execute(params![path, hash, now_unix])
                .map_err(runtime_error)?;
        }
    }
    for (key, value) in [
        ("node_count", graph_to_database.len().to_string()),
        ("edge_count", edges_len(&transaction)?.to_string()),
        ("last_updated", now_iso.to_owned()),
    ] {
        transaction
            .execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                params![key, value],
            )
            .map_err(runtime_error)?;
    }
    transaction.commit().map_err(runtime_error)
}

fn edges_len(connection: &Connection) -> PyResult<i64> {
    connection
        .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
        .map_err(runtime_error)
}

/// Synchronize a graph while retaining stable rows and refreshing only deltas.
#[pyfunction]
pub fn save_graph_incremental_sqlite(
    path: &str,
    nodes: Vec<PersistNode>,
    edges: Vec<PersistEdge>,
    file_hashes: Option<Vec<(String, String)>>,
    now_iso: &str,
    now_unix: i64,
) -> PyResult<()> {
    let mut connection = Connection::open(path).map_err(runtime_error)?;
    connection
        .busy_timeout(std::time::Duration::from_secs(30))
        .map_err(runtime_error)?;
    let transaction = connection.transaction().map_err(runtime_error)?;

    let existing_nodes: HashMap<String, StoredNode> = {
        let mut statement = transaction
            .prepare(
                "SELECT id, kind, name, qualified_name, source_uri, line_start,
                        line_end, properties, source_text, valid_from, valid_to FROM nodes",
            )
            .map_err(runtime_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(3)?,
                    StoredNode {
                        id: row.get(0)?,
                        kind: row.get(1)?,
                        name: row.get(2)?,
                        source_uri: row.get(4)?,
                        line_start: row.get(5)?,
                        line_end: row.get(6)?,
                        properties: row.get(7)?,
                        source_text: row.get(8)?,
                        valid_from: row.get(9)?,
                        valid_to: row.get(10)?,
                    },
                ))
            })
            .map_err(runtime_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(runtime_error)?;
        rows.into_iter().collect()
    };
    let target_qnames: HashSet<&str> = nodes.iter().map(|node| node.3.as_str()).collect();
    let removed_node_ids: HashSet<i64> = existing_nodes
        .iter()
        .filter(|(qualified_name, _)| !target_qnames.contains(qualified_name.as_str()))
        .map(|(_, node)| node.id)
        .collect();
    let mut changed_node_ids = HashSet::new();
    let mut graph_to_database = HashMap::with_capacity(nodes.len());

    for (
        graph_id,
        kind,
        name,
        qualified_name,
        source_uri,
        line_start,
        line_end,
        properties,
        source_text,
        valid_from,
        valid_to,
    ) in nodes
    {
        let database_id = if let Some(existing) = existing_nodes.get(&qualified_name) {
            let changed = existing.kind != kind
                || existing.name != name
                || existing.source_uri != source_uri
                || existing.line_start != line_start
                || existing.line_end != line_end
                || existing.properties != properties
                || existing.source_text != source_text
                || existing.valid_from != valid_from
                || existing.valid_to != valid_to;
            if changed {
                transaction
                    .execute(
                        "UPDATE nodes SET kind=?, name=?, source_uri=?, line_start=?, line_end=?,
                         properties=?, source_text=?, valid_from=?, valid_to=? WHERE id=?",
                        params![
                            kind,
                            name,
                            source_uri,
                            line_start,
                            line_end,
                            properties,
                            source_text,
                            valid_from,
                            valid_to,
                            existing.id,
                        ],
                    )
                    .map_err(runtime_error)?;
                changed_node_ids.insert(existing.id);
            }
            existing.id
        } else {
            transaction
                .execute(
                    "INSERT INTO nodes
                     (kind, name, qualified_name, source_uri, line_start, line_end,
                      properties, source_text, valid_from, valid_to)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params![
                        kind,
                        name,
                        qualified_name,
                        source_uri,
                        line_start,
                        line_end,
                        properties,
                        source_text,
                        valid_from,
                        valid_to,
                    ],
                )
                .map_err(runtime_error)?;
            let id = transaction.last_insert_rowid();
            changed_node_ids.insert(id);
            id
        };
        graph_to_database.insert(graph_id, database_id);
    }

    let mut existing_edges: HashMap<EdgeKey, Vec<i64>> = {
        let mut statement = transaction
            .prepare(
                "SELECT id, src_id, dst_id, kind, confidence, conf_class,
                        properties, valid_from, valid_to FROM edges",
            )
            .map_err(runtime_error)?;
        let rows = statement
            .query_map([], |row| {
                let confidence: f64 = row.get(4)?;
                Ok((
                    EdgeKey {
                        source: row.get(1)?,
                        target: row.get(2)?,
                        kind: row.get(3)?,
                        confidence_bits: confidence.to_bits(),
                        confidence_class: row.get(5)?,
                        properties: row.get(6)?,
                        valid_from: row.get(7)?,
                        valid_to: row.get(8)?,
                    },
                    row.get::<_, i64>(0)?,
                ))
            })
            .map_err(runtime_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(runtime_error)?;
        let mut grouped: HashMap<EdgeKey, Vec<i64>> = HashMap::new();
        for (key, id) in rows {
            grouped.entry(key).or_default().push(id);
        }
        grouped
    };

    for (source, target, kind, confidence, confidence_class, properties, valid_from, valid_to) in
        edges
    {
        let source_id = *graph_to_database
            .get(&source)
            .ok_or_else(|| PyRuntimeError::new_err(format!("edge source {source} is missing")))?;
        let target_id = *graph_to_database
            .get(&target)
            .ok_or_else(|| PyRuntimeError::new_err(format!("edge target {target} is missing")))?;
        let key = EdgeKey {
            source: source_id,
            target: target_id,
            kind,
            confidence_bits: confidence.to_bits(),
            confidence_class,
            properties,
            valid_from,
            valid_to,
        };
        if let Some(retained) = existing_edges.get_mut(&key) {
            if retained.pop().is_some() {
                continue;
            }
        }
        transaction
            .execute(
                "INSERT INTO edges
                 (src_id, dst_id, kind, confidence, conf_class, properties,
                  valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                params![
                    key.source,
                    key.target,
                    key.kind,
                    f64::from_bits(key.confidence_bits),
                    key.confidence_class,
                    key.properties,
                    key.valid_from,
                    key.valid_to,
                ],
            )
            .map_err(runtime_error)?;
    }
    for edge_id in existing_edges.into_values().flatten() {
        transaction
            .execute("DELETE FROM edges WHERE id=?", [edge_id])
            .map_err(runtime_error)?;
    }

    for node_id in changed_node_ids.iter().chain(&removed_node_ids) {
        transaction
            .execute("DELETE FROM nodes_fts WHERE rowid=?", [node_id])
            .map_err(runtime_error)?;
        transaction
            .execute("DELETE FROM embeddings WHERE node_id=?", [node_id])
            .map_err(runtime_error)?;
    }
    for node_id in &removed_node_ids {
        transaction
            .execute("DELETE FROM nodes WHERE id=?", [node_id])
            .map_err(runtime_error)?;
    }
    for node_id in &changed_node_ids {
        transaction
            .execute(
                "INSERT INTO nodes_fts (rowid, kind, name, qualified_name)
                 SELECT id, kind, name, qualified_name FROM nodes WHERE id=?",
                [node_id],
            )
            .map_err(runtime_error)?;
    }

    if let Some(file_hashes) = file_hashes {
        transaction
            .execute("DELETE FROM file_state", [])
            .map_err(runtime_error)?;
        for (file_path, hash) in file_hashes {
            transaction
                .execute(
                    "INSERT INTO file_state (path, hash, indexed_at_unix) VALUES (?, ?, ?)",
                    params![file_path, hash, now_unix],
                )
                .map_err(runtime_error)?;
        }
    }
    for (key, value) in [
        ("node_count", graph_to_database.len().to_string()),
        ("edge_count", edges_len(&transaction)?.to_string()),
        ("last_updated", now_iso.to_owned()),
    ] {
        transaction
            .execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                params![key, value],
            )
            .map_err(runtime_error)?;
    }
    transaction.commit().map_err(runtime_error)
}

/// Read the complete active graph using one native SQLite scan.
#[pyfunction]
pub fn load_graph_sqlite(path: &str) -> PyResult<(Vec<LoadedNode>, Vec<LoadedEdge>)> {
    let connection = Connection::open(path).map_err(runtime_error)?;
    connection
        .busy_timeout(std::time::Duration::from_secs(30))
        .map_err(runtime_error)?;
    let nodes = {
        let mut statement = connection
            .prepare(
                "SELECT id, kind, name, qualified_name, source_uri, line_start,
                        line_end, properties, source_text, valid_from, valid_to
                 FROM nodes ORDER BY id",
            )
            .map_err(runtime_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                ))
            })
            .map_err(runtime_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(runtime_error)?;
        rows
    };
    let edges = {
        let mut statement = connection
            .prepare(
                "SELECT src_id, dst_id, kind, confidence, conf_class, properties,
                        valid_from, valid_to FROM edges ORDER BY id",
            )
            .map_err(runtime_error)?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, i64>(1)? as u64,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })
            .map_err(runtime_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(runtime_error)?;
        rows
    };
    Ok((nodes, edges))
}
