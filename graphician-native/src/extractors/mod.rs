//! Shared utilities for all language extractors.
//! Re-exports helpers from lib.rs.

pub use crate::{
    child_by_field, extract_decorators, extract_name, node_text, node_text_str,
    should_suppress_call_placeholder, truncated_source_text,
};
pub use crate::{is_test_file_path, is_test_name};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyList;
use tree_sitter::Node;

pub use crate::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};

/// Extract trimmed text from a node.
pub fn text(node: &Node, source: &[u8]) -> String {
    node_text_str(node, source).trim().to_string()
}

/// Get direct children of a node.
pub fn children<'a>(node: &Node<'a>) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor).collect()
}

/// Build a scoped qualified name.
pub fn scoped_qname(file_qn: &str, scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        format!("{}::{}", file_qn, name)
    } else {
        format!("{}::{}::{}", file_qn, scope.join("::"), name)
    }
}

/// Convert ExtractionResult to Python dict.
pub fn result_to_dict(py: Python, result: &ExtractionResult) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    let nodes_list = PyList::empty(py);
    for node in &result.nodes {
        let dict = PyDict::new(py);
        dict.set_item("kind", &node.kind)?;
        dict.set_item("qualified_name", &node.qualified_name)?;
        dict.set_item("name", &node.name)?;
        dict.set_item("source_uri", node.source_uri.as_deref().unwrap_or(""))?;
        dict.set_item("line_start", node.line_start)?;
        dict.set_item("line_end", node.line_end)?;
        dict.set_item("source_text", node.source_text.as_deref().unwrap_or(""))?;
        let props_dict = PyDict::new(py);
        for (k, v) in &node.properties {
            if v == "true" {
                props_dict.set_item(k, true)?;
            } else if v == "false" {
                props_dict.set_item(k, false)?;
            } else {
                props_dict.set_item(k, v)?;
            }
        }
        dict.set_item("properties", props_dict)?;
        nodes_list.append(dict)?;
    }
    let edges_list = PyList::empty(py);
    for edge in &result.edges {
        let dict = PyDict::new(py);
        dict.set_item("src_qn", &edge.src_qn)?;
        dict.set_item("dst_qn", &edge.dst_qn)?;
        dict.set_item("kind", &edge.kind)?;
        dict.set_item("conf_class", &edge.conf_class)?;
        dict.set_item("confidence", edge.confidence)?;
        let props_dict = PyDict::new(py);
        for (k, v) in &edge.properties {
            props_dict.set_item(k, v)?;
        }
        dict.set_item("properties", props_dict)?;
        edges_list.append(dict)?;
    }
    let calls_list = PyList::empty(py);
    for call in &result.calls {
        let dict = PyDict::new(py);
        dict.set_item("caller_qn", &call.caller_qn)?;
        dict.set_item("callee_qn", &call.callee_qn)?;
        dict.set_item("receiver", call.receiver.as_deref().unwrap_or(""))?;
        calls_list.append(dict)?;
    }
    output.set_item("nodes", nodes_list)?;
    output.set_item("edges", edges_list)?;
    output.set_item("calls", calls_list)?;
    Ok(output.into_any().unbind().into())
}

// Language-specific extractors
pub mod cpp;
pub mod go;
pub mod java;
pub mod javascript;
pub mod rust;
pub mod typescript;
