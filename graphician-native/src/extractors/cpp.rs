//! C/C++ source extraction.
//! Emits: File, Namespace, Class, Struct, Function, Method, Module

use pyo3::prelude::*;
use tree_sitter::{Query, QueryCursor, StreamingIterator};

use super::{children, result_to_dict, text};
use super::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::{is_test_file_path, is_test_name, should_suppress_call_placeholder};

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
pub fn extract_cpp_file(
    py: Python,
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> PyResult<PyObject> {
    let lang = tree_sitter_cpp::LANGUAGE.into();
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

    let file_is_test = is_test_file_path(file_path);
    let file_name = std::path::Path::new(file_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut result = ExtractionResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        calls: Vec::new(),
    };

    let lines = source_str.lines().count();
    result.nodes.push(ExtractedNode {
        kind: "file".to_string(),
        qualified_name: file_qn_full.clone(),
        name: file_name,
        source_uri: Some(file_path.to_string()),
        line_start: 0,
        line_end: lines,
        source_text: Some(source_str.clone()),
        properties: vec![("dialect".to_string(), "cpp".to_string())],
    });

    walk_scope(
        &root,
        &source_str,
        &file_qn_full,
        file_path,
        &[],
        file_is_test,
        source,
        &mut result,
    );

    // Namespace imports
    let ns_query = Query::new(
        &lang,
        r#"(namespace_definition name: (namespace_identifier) @name)"#,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e))
    })?;
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&ns_query, root, source.as_ref());
    while let Some(m) = matches.next() {
        for cap in m.captures {
            let name = text(&cap.node, source.as_ref());
            if !name.is_empty() {
                let ns_qn = format!("namespace::{}", name);
                result.nodes.push(ExtractedNode {
                    kind: "module".to_string(),
                    qualified_name: ns_qn.clone(),
                    name: name,
                    source_uri: None,
                    line_start: 0,
                    line_end: 0,
                    source_text: None,
                    properties: vec![("dialect".to_string(), "cpp".to_string())],
                });
                result.edges.push(ExtractedEdge {
                    src_qn: file_qn_full.clone(),
                    dst_qn: ns_qn,
                    kind: "imports".to_string(),
                    conf_class: "extracted".to_string(),
                    confidence: 1.0,
                    properties: vec![],
                });
            }
        }
    }

    result_to_dict(py, &result)
}

fn walk_scope(
    node: &tree_sitter::Node,
    source_str: &str,
    file_qn: &str,
    file_path: &str,
    scope: &[String],
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "class_specifier" | "struct_specifier" => {
                let kind = if child.kind() == "struct_specifier" {
                    "class"
                } else {
                    "class"
                };
                let name_node = child
                    .child_by_field_name("name")
                    .or_else(|| child.child_by_field_name("declarator"));
                let Some(name_node) = name_node else {
                    continue;
                };
                let name = text(&name_node, source);
                if name.is_empty() {
                    continue;
                }
                let qn = if scope.is_empty() {
                    format!("{}::{}", file_qn, name)
                } else {
                    format!("{}::{}::{}", file_qn, scope.join("::"), name)
                };
                let start = (child.start_position().row + 1) as usize;
                let end = (child.end_position().row + 1) as usize;
                let src_text = super::truncated_source_text(source_str, start, end);

                result.nodes.push(ExtractedNode {
                    kind: kind.to_string(),
                    qualified_name: qn.clone(),
                    name: name.clone(),
                    source_uri: Some(file_path.to_string()),
                    line_start: start,
                    line_end: end,
                    source_text: src_text,
                    properties: vec![("dialect".to_string(), "cpp".to_string())],
                });
                result.edges.push(ExtractedEdge {
                    src_qn: file_qn.to_string(),
                    dst_qn: qn.clone(),
                    kind: "defines".to_string(),
                    conf_class: "extracted".to_string(),
                    confidence: 1.0,
                    properties: vec![],
                });

                // Base classes
                if let Some(base_clause) = child.child_by_field_name("base_clause") {
                    for c in children(&base_clause) {
                        if c.kind() == "qualified_identifier" {
                            let parts: Vec<String> = c
                                .children(&mut c.walk())
                                .filter(|n| n.kind() == "identifier")
                                .map(|n| text(&n, source))
                                .collect();
                            if !parts.is_empty() {
                                let base = parts.join("::");
                                let base_qn = format!("type::{}", base);
                                result.nodes.push(ExtractedNode {
                                    kind: "class".to_string(),
                                    qualified_name: base_qn.clone(),
                                    name: base,
                                    source_uri: None,
                                    line_start: 0,
                                    line_end: 0,
                                    source_text: None,
                                    properties: vec![("dialect".to_string(), "cpp".to_string())],
                                });
                                result.edges.push(ExtractedEdge {
                                    src_qn: qn.clone(),
                                    dst_qn: base_qn,
                                    kind: "inherits".to_string(),
                                    conf_class: "extracted".to_string(),
                                    confidence: 1.0,
                                    properties: vec![],
                                });
                            }
                        }
                    }
                }

                if let Some(body) = child.child_by_field_name("body") {
                    let mut child_scope = scope.to_vec();
                    child_scope.push(name);
                    walk_scope(
                        &body,
                        source_str,
                        file_qn,
                        file_path,
                        &child_scope,
                        file_is_test,
                        source,
                        result,
                    );
                }
            }
            "function_definition" => {
                let Some(declarator) = child.child_by_field_name("declarator") else {
                    continue;
                };
                let name_node = declarator
                    .child_by_field_name("name")
                    .or_else(|| declarator.child_by_field_name("declarator"));
                let Some(name_node) = name_node else {
                    continue;
                };
                let name = text(&name_node, source);
                if name.is_empty() {
                    continue;
                }
                let is_test = file_is_test || is_test_name(&name);
                let qn = if scope.is_empty() {
                    format!("{}::{}", file_qn, name)
                } else {
                    format!("{}::{}::{}", file_qn, scope.join("::"), name)
                };
                let start = (child.start_position().row + 1) as usize;
                let end = (child.end_position().row + 1) as usize;
                let src_text = super::truncated_source_text(source_str, start, end);

                let mut props = vec![("dialect".to_string(), "cpp".to_string())];
                if is_test {
                    props.push(("is_test".to_string(), "true".to_string()));
                }

                result.nodes.push(ExtractedNode {
                    kind: "function".to_string(),
                    qualified_name: qn.clone(),
                    name: name.clone(),
                    source_uri: Some(file_path.to_string()),
                    line_start: start,
                    line_end: end,
                    source_text: src_text,
                    properties: props,
                });
                result.edges.push(ExtractedEdge {
                    src_qn: file_qn.to_string(),
                    dst_qn: qn.clone(),
                    kind: "defines".to_string(),
                    conf_class: "extracted".to_string(),
                    confidence: 1.0,
                    properties: vec![],
                });
                if let Some(body) = child.child_by_field_name("body") {
                    emit_calls(&body, source, &qn, result);
                }
            }
            "namespace_definition" => {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = text(&name_node, source);
                    if !name.is_empty() {
                        let ns_qn = if scope.is_empty() {
                            format!("{}::{}", file_qn, name)
                        } else {
                            format!("{}::{}::{}", file_qn, scope.join("::"), name)
                        };
                        let start = (child.start_position().row + 1) as usize;
                        let end = (child.end_position().row + 1) as usize;
                        result.nodes.push(ExtractedNode {
                            kind: "module".to_string(),
                            qualified_name: ns_qn.clone(),
                            name: name.clone(),
                            source_uri: Some(file_path.to_string()),
                            line_start: start,
                            line_end: end,
                            source_text: None,
                            properties: vec![("dialect".to_string(), "cpp".to_string())],
                        });
                        result.edges.push(ExtractedEdge {
                            src_qn: file_qn.to_string(),
                            dst_qn: ns_qn.clone(),
                            kind: "defines".to_string(),
                            conf_class: "extracted".to_string(),
                            confidence: 1.0,
                            properties: vec![],
                        });
                        if let Some(body) = child.child_by_field_name("body") {
                            let mut child_scope = scope.to_vec();
                            child_scope.push(name);
                            walk_scope(
                                &body,
                                source_str,
                                &ns_qn,
                                file_path,
                                &child_scope,
                                file_is_test,
                                source,
                                result,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
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
            if let Some(func) = child.child_by_field_name("function") {
                let mut name = None;
                let receiver = None;
                match func.kind() {
                    "identifier" => {
                        name = Some(text(&func, source));
                    }
                    "qualified_identifier" => {
                        let parts: Vec<String> = func
                            .children(&mut func.walk())
                            .filter(|n| n.kind() == "identifier")
                            .map(|n| text(&n, source))
                            .collect();
                        if !parts.is_empty() {
                            name = Some(parts.join("::"));
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
