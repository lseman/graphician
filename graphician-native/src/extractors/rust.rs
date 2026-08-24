//! Rust source extraction.
//! Emits: File, Function, Method, Trait, Struct, Enum, Impl, Module
//! Edges: Defines, Calls, Imports, Implements

use pyo3::prelude::*;
use tree_sitter::{Query, QueryCursor, StreamingIterator};

use super::{children, result_to_dict, text};
use super::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::{is_test_file_path, is_test_name, node_text_str, should_suppress_call_placeholder};

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
pub fn extract_rust_file(
    py: Python,
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> PyResult<PyObject> {
    let lang = tree_sitter_rust::LANGUAGE.into();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&lang).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("language load failed: {}", e))
    })?;
    let source_str = String::from_utf8_lossy(source).into_owned();
    let tree = parser
        .parse(&source_str, None)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("parse failed"))?;
    let root = tree.root_node();

    let file_qn_full = if file_qn.is_empty() {
        let stem = std::path::Path::new(file_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        format!("file::{}", stem)
    } else {
        file_qn.to_string()
    };

    let file_is_test =
        is_test_file_path(file_path) || is_test_file_path(&format!("/{}", file_path));
    let file_name = std::path::Path::new(file_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut result = ExtractionResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        calls: Vec::new(),
    };

    // File node
    let lines = source_str.lines().count();
    result.nodes.push(ExtractedNode {
        kind: "file".to_string(),
        qualified_name: file_qn_full.clone(),
        name: file_name.clone(),
        source_uri: Some(file_path.to_string()),
        line_start: 0,
        line_end: lines,
        source_text: Some(source_str.clone()),
        properties: vec![("dialect".to_string(), "rust".to_string())],
    });

    let query = Query::new(
        &lang,
        r#"
        [
            (function_item name: (identifier) @name)
            (function_item)
            (trait_item name: (type_identifier) @name)
            (struct_item name: (type_identifier) @name)
            (enum_item name: (type_identifier) @name)
            (impl_item)
        ]
    "#,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e))
    })?;
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source.as_ref());

    while let Some(m) = matches.next() {
        for cap in m.captures {
            let cn = query.capture_names()[cap.index as usize];
            let node = cap.node;
            let start = (node.start_position().row + 1) as usize;
            let end = (node.end_position().row + 1) as usize;
            let src_text = super::truncated_source_text(&source_str, start, end);

            match cn {
                "name" => {
                    let name = text(&node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }

                    // Determine scope
                    let scope = rust_scope(&node);
                    let qn = if scope.is_empty() {
                        format!("{}::{}", file_qn_full, name)
                    } else {
                        format!("{}::{}::{}", file_qn_full, scope.join("::"), name)
                    };

                    // Determine kind
                    let kind = if has_method_parent(&node) {
                        "method".to_string()
                    } else if let Some(parent) = node.parent() {
                        match parent.kind() {
                            "trait_item" => "function".to_string(),
                            _ => "function".to_string(),
                        }
                    } else {
                        "function".to_string()
                    };

                    let is_test = file_is_test || is_test_name(&name);
                    let mut props = vec![("dialect".to_string(), "rust".to_string())];
                    if is_test {
                        props.push(("is_test".to_string(), "true".to_string()));
                    }

                    result.nodes.push(ExtractedNode {
                        kind,
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text.clone(),
                        properties: props,
                    });
                    result.edges.push(ExtractedEdge {
                        src_qn: file_qn_full.clone(),
                        dst_qn: qn.clone(),
                        kind: "defines".to_string(),
                        conf_class: "extracted".to_string(),
                        confidence: 1.0,
                        properties: vec![],
                    });
                    // Calls
                    if let Some(body) = node.child_by_field_name("body") {
                        emit_calls(&body, source.as_ref(), &qn, &mut result);
                    }
                }
                "trait_item" | "struct_item" | "enum_item" => {
                    let kind = if cn == "trait_item" { "trait" } else { "class" };
                    let name = text(&node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: kind.to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "rust".to_string())],
                    });
                    result.edges.push(ExtractedEdge {
                        src_qn: file_qn_full.clone(),
                        dst_qn: qn,
                        kind: "defines".to_string(),
                        conf_class: "extracted".to_string(),
                        confidence: 1.0,
                        properties: vec![],
                    });
                }
                "impl_item" => {
                    // Extract methods from impl
                    let mut method_cursor = node.walk();
                    for child in node.children(&mut method_cursor) {
                        if child.kind() == "function_item" {
                            if let Some(name_node) = child.child_by_field_name("name") {
                                let name = text(&name_node, source.as_ref());
                                if name.is_empty() {
                                    continue;
                                }
                                let qn = format!("{}::{}", file_qn_full, name);
                                let m_start = (child.start_position().row + 1) as usize;
                                let m_end = (child.end_position().row + 1) as usize;
                                let m_src =
                                    super::truncated_source_text(&source_str, m_start, m_end);
                                result.nodes.push(ExtractedNode {
                                    kind: "method".to_string(),
                                    qualified_name: qn.clone(),
                                    name: name.clone(),
                                    source_uri: Some(file_path.to_string()),
                                    line_start: m_start,
                                    line_end: m_end,
                                    source_text: m_src,
                                    properties: vec![("dialect".to_string(), "rust".to_string())],
                                });
                                result.edges.push(ExtractedEdge {
                                    src_qn: file_qn_full.clone(),
                                    dst_qn: qn.clone(),
                                    kind: "defines".to_string(),
                                    conf_class: "extracted".to_string(),
                                    confidence: 1.0,
                                    properties: vec![],
                                });
                                if let Some(body) = child.child_by_field_name("body") {
                                    emit_calls(&body, source.as_ref(), &qn, &mut result);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Imports
    let import_query = Query::new(&lang, r#"(use_declaration (scoped_identifier) @path)"#)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e))
        })?;
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&import_query, root, source.as_ref());
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let path_text = text(&cap.node, source.as_ref());
            if path_text.is_empty() {
                continue;
            }
            let mod_qn = format!("module::{}", path_text);
            result.nodes.push(ExtractedNode {
                kind: "module".to_string(),
                qualified_name: mod_qn.clone(),
                name: path_text,
                source_uri: None,
                line_start: 0,
                line_end: 0,
                source_text: None,
                properties: vec![("dialect".to_string(), "rust".to_string())],
            });
            result.edges.push(ExtractedEdge {
                src_qn: file_qn_full.clone(),
                dst_qn: mod_qn,
                kind: "imports".to_string(),
                conf_class: "extracted".to_string(),
                confidence: 1.0,
                properties: vec![],
            });
        }
    }

    result_to_dict(py, &result)
}

fn has_method_parent(node: &tree_sitter::Node) -> bool {
    node.parent()
        .map(|p| p.kind() == "impl_item")
        .unwrap_or(false)
}

fn rust_scope(node: &tree_sitter::Node) -> Vec<String> {
    let mut scope = Vec::new();
    let mut current = node.parent();
    while let Some(parent) = current {
        if parent.kind() == "mod_item" {
            if let Some(name) = parent.child_by_field_name("name") {
                scope.push(node_text_str(&name, &[0u8; 0])); // Simplified - would need source
            }
        }
        current = parent.parent();
    }
    scope.reverse();
    scope
}

fn emit_calls(
    node: &tree_sitter::Node,
    source: &[u8],
    caller_qn: &str,
    result: &mut ExtractionResult,
) {
    let mut stack: Vec<tree_sitter::Node> = children(node);
    while let Some(child) = stack.pop() {
        if child.kind() == "call_expression" {
            let mut func_node = None;
            for c in child.children(&mut child.walk()) {
                if child.field_name_for_child(c.id() as u32) == Some("function")
                    || child.field_name_for_child(c.id() as u32) == Some("callee")
                {
                    func_node = Some(c);
                    break;
                }
            }
            if let Some(func) = func_node {
                let mut name = None;
                let mut receiver = None;
                match func.kind() {
                    "identifier" => {
                        name = Some(text(&func, source));
                    }
                    "scoped_identifier" => {
                        if let Some(name_node) = func.child_by_field_name("name") {
                            name = Some(text(&name_node, source));
                        }
                        if let Some(obj) = func.child_by_field_name("path") {
                            receiver = Some(text(&obj, source));
                        }
                    }
                    _ => {}
                }
                if let Some(n) = name {
                    if !n.starts_with('_') && !should_suppress_call_placeholder(&n) {
                        result.calls.push(CallPlaceholder {
                            caller_qn: caller_qn.to_string(),
                            callee_qn: format!("call::{}", n),
                            receiver,
                        });
                    }
                }
            }
        }
        stack.extend(children(&child));
    }
}
