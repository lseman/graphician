//! JavaScript extraction (.js/.mjs/.cjs; .jsx via TSX grammar)
//! Emits: File, Class, Function, Method, Module

use pyo3::prelude::*;
use tree_sitter::{Query, QueryCursor, StreamingIterator};

use super::{children, result_to_dict, text};
use super::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::{
    extract_decorators, is_test_file_path, is_test_name, should_suppress_call_placeholder,
};

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
pub fn extract_javascript_file(
    py: Python,
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> PyResult<PyObject> {
    let is_jsx = file_path.ends_with(".jsx");
    let lang = if is_jsx {
        tree_sitter_typescript::LANGUAGE_TSX.into()
    } else {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    };

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
        properties: vec![("dialect".to_string(), "javascript".to_string())],
    });

    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        match child.kind() {
            "class_declaration" => emit_class(
                &child,
                &file_qn_full,
                file_path,
                file_is_test,
                source,
                &mut result,
            ),
            "function_declaration" => emit_fn(
                &child,
                &file_qn_full,
                file_path,
                file_is_test,
                source,
                &mut result,
            ),
            "method_definition" => {
                // Top-level method definition (not in class body)
                emit_method(
                    &child,
                    &file_qn_full,
                    file_path,
                    file_is_test,
                    source,
                    &mut result,
                );
            }
            "export_statement" | "export_declaration" => {
                if let Some(decl) = child.child_by_field_name("declaration") {
                    match decl.kind() {
                        "class_declaration" => emit_class(
                            &decl,
                            &file_qn_full,
                            file_path,
                            file_is_test,
                            source,
                            &mut result,
                        ),
                        "function_declaration" => emit_fn(
                            &decl,
                            &file_qn_full,
                            file_path,
                            file_is_test,
                            source,
                            &mut result,
                        ),
                        _ => {}
                    }
                }
            }
            "import_statement" => emit_import(&child, &file_qn_full, source, &mut result),
            "variable_declaration" | "lexical_declaration" => {
                emit_var_functions(
                    &child,
                    &file_qn_full,
                    file_path,
                    file_is_test,
                    source,
                    &mut result,
                );
            }
            _ => {}
        }
    }

    result_to_dict(py, &result)
}

fn emit_class(
    node: &tree_sitter::Node,
    file_qn: &str,
    file_path: &str,
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = text(&name_node, source);
    if name.is_empty() {
        return;
    }
    let qn = format!("{}::{}", file_qn, name);
    let start = (node.start_position().row + 1) as usize;
    let end = (node.end_position().row + 1) as usize;
    let src_text =
        super::truncated_source_text(&String::from_utf8_lossy(source).into_owned(), start, end);

    let mut props = vec![("dialect".to_string(), "javascript".to_string())];
    let decs = extract_decorators(node, source);
    if !decs.is_empty() {
        if !decs.is_empty() {
            use serde_json::json;
            props.push(("decorators".to_string(), json!(decs).to_string()));
        }
    }

    result.nodes.push(ExtractedNode {
        kind: "class".to_string(),
        qualified_name: qn.clone(),
        name,
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

    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            if child.kind() == "method_definition" {
                emit_method(&child, &qn, file_path, file_is_test, source, result);
            }
        }
    }
}

fn emit_fn(
    node: &tree_sitter::Node,
    file_qn: &str,
    file_path: &str,
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let Some(name_node) = node.child_by_field_name("name") else {
        return;
    };
    let name = text(&name_node, source);
    if name.is_empty() {
        return;
    }
    let qn = format!("{}::{}", file_qn, name);
    let is_test = file_is_test || is_test_name(&name);
    let start = (node.start_position().row + 1) as usize;
    let end = (node.end_position().row + 1) as usize;
    let src_text =
        super::truncated_source_text(&String::from_utf8_lossy(source).into_owned(), start, end);

    let mut props = vec![("dialect".to_string(), "javascript".to_string())];
    if is_test {
        props.push(("is_test".to_string(), "true".to_string()));
    }

    result.nodes.push(ExtractedNode {
        kind: "function".to_string(),
        qualified_name: qn.clone(),
        name,
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
    if let Some(body) = node.child_by_field_name("body") {
        emit_calls(&body, source, &qn, result);
    }
}

fn emit_method(
    node: &tree_sitter::Node,
    parent_qn: &str,
    file_path: &str,
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let name_node = node
        .child_by_field_name("name")
        .or_else(|| node.child_by_field_name("property"));
    let Some(name_node) = name_node else {
        return;
    };
    let name = text(&name_node, source);
    if name.is_empty() {
        return;
    }
    // The dedicated JavaScript parser keeps methods in the file namespace
    // while retaining the class -> method Defines edge.
    let file_qn = parent_qn.split("::").take(2).collect::<Vec<_>>().join("::");
    let qn = format!("{}::{}", file_qn, name);
    let is_test = file_is_test || is_test_name(&name);
    let start = (node.start_position().row + 1) as usize;
    let end = (node.end_position().row + 1) as usize;
    let src_text =
        super::truncated_source_text(&String::from_utf8_lossy(source).into_owned(), start, end);

    let mut props = vec![("dialect".to_string(), "javascript".to_string())];
    if is_test {
        props.push(("is_test".to_string(), "true".to_string()));
    }

    result.nodes.push(ExtractedNode {
        kind: "method".to_string(),
        qualified_name: qn.clone(),
        name,
        source_uri: Some(file_path.to_string()),
        line_start: start,
        line_end: end,
        source_text: src_text,
        properties: props,
    });
    result.edges.push(ExtractedEdge {
        src_qn: parent_qn.to_string(),
        dst_qn: qn.clone(),
        kind: "defines".to_string(),
        conf_class: "extracted".to_string(),
        confidence: 1.0,
        properties: vec![],
    });
    if let Some(body) = node.child_by_field_name("body") {
        emit_calls(&body, source, &qn, result);
    }
}

fn emit_var_functions(
    node: &tree_sitter::Node,
    file_qn: &str,
    file_path: &str,
    _file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let mut cursor = node.walk();
    for decl in node.children(&mut cursor) {
        if decl.kind() != "variable_declarator" {
            continue;
        }
        if let Some(name_node) = decl.child_by_field_name("name") {
            let name = text(&name_node, source);
            if name.is_empty() {
                continue;
            }
            if let Some(value) = decl.child_by_field_name("value") {
                if matches!(value.kind(), "arrow_function" | "function_expression") {
                    let qn = format!("{}::{}", file_qn, name);
                    let start = (decl.start_position().row + 1) as usize;
                    let end = (decl.end_position().row + 1) as usize;
                    result.nodes.push(ExtractedNode {
                        kind: "function".to_string(),
                        qualified_name: qn.clone(),
                        name,
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: None,
                        properties: vec![("dialect".to_string(), "javascript".to_string())],
                    });
                    result.edges.push(ExtractedEdge {
                        src_qn: file_qn.to_string(),
                        dst_qn: qn.clone(),
                        kind: "defines".to_string(),
                        conf_class: "extracted".to_string(),
                        confidence: 1.0,
                        properties: vec![],
                    });
                    if let Some(body) = value.child_by_field_name("body") {
                        emit_calls(&body, source, &qn, result);
                    }
                }
            }
        }
    }
}

fn emit_import(
    node: &tree_sitter::Node,
    file_qn: &str,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    // Query for ESM import statements
    let esm_query = Query::new(
        &tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        r#"(import_statement (string) @path)"#,
    )
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("invalid query"))
    .ok();

    if let Some(query) = esm_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, *node, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let path_text = text(&cap.node, source);
                if path_text.is_empty() {
                    continue;
                }
                let mod_qn = format!("module::{}", path_text);
                if !result.nodes.iter().any(|n| n.qualified_name == mod_qn) {
                    result.nodes.push(ExtractedNode {
                        kind: "module".to_string(),
                        qualified_name: mod_qn.clone(),
                        name: path_text,
                        source_uri: None,
                        line_start: 0,
                        line_end: 0,
                        source_text: None,
                        properties: vec![("dialect".to_string(), "javascript".to_string())],
                    });
                    result.edges.push(ExtractedEdge {
                        src_qn: file_qn.to_string(),
                        dst_qn: mod_qn,
                        kind: "imports".to_string(),
                        conf_class: "extracted".to_string(),
                        confidence: 1.0,
                        properties: vec![],
                    });
                }
            }
        }
    }

    // Query for require() calls
    let require_query = Query::new(
        &tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        r#"(call_expression
            function: (identifier) @require_fn
            arguments: (arguments (string) @module_path)
        )"#,
    )
    .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("invalid query"))
    .ok();

    if let Some(require_query) = require_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&require_query, *node, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let cn = require_query.capture_names()[cap.index as usize];
                if cn == "module_path" {
                    let path_text = text(&cap.node, source);
                    if path_text.is_empty() {
                        continue;
                    }
                    let mod_qn = format!("module::{}", path_text);
                    if !result.nodes.iter().any(|n| n.qualified_name == mod_qn) {
                        result.nodes.push(ExtractedNode {
                            kind: "module".to_string(),
                            qualified_name: mod_qn.clone(),
                            name: path_text,
                            source_uri: None,
                            line_start: 0,
                            line_end: 0,
                            source_text: None,
                            properties: vec![("dialect".to_string(), "javascript".to_string())],
                        });
                        result.edges.push(ExtractedEdge {
                            src_qn: file_qn.to_string(),
                            dst_qn: mod_qn,
                            kind: "imports".to_string(),
                            conf_class: "extracted".to_string(),
                            confidence: 1.0,
                            properties: vec![],
                        });
                    }
                }
            }
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
            let func_node = child
                .child_by_field_name("function")
                .or_else(|| child.child_by_field_name("callee"));
            if let Some(func) = func_node {
                let mut name = None;
                let mut receiver = None;
                match func.kind() {
                    "identifier" => {
                        name = Some(text(&func, source));
                    }
                    "member_expression" => {
                        if let Some(prop) = func.child_by_field_name("property") {
                            name = Some(text(&prop, source));
                        }
                        if let Some(obj) = func.child_by_field_name("object") {
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
