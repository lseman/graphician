//! Go source extraction.
//! Emits: File, Function, Method, Struct, Interface, TypeAlias, Constant, Variable, Module, Test
//! Edges: Defines, Calls, Imports, Implements, Receiver

use pyo3::prelude::*;
use tree_sitter::{Query, QueryCursor, StreamingIterator};

use super::{result_to_dict, text};
use super::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::{is_test_file_path, is_test_name, should_suppress_call_placeholder};

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
pub fn extract_go_file(
    py: Python,
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> PyResult<PyObject> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_go::LANGUAGE.into())
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "language load failed: {}",
                e
            ))
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

    // File node
    let lines = source_str.lines().count();
    result.nodes.push(ExtractedNode {
        kind: "file".to_string(),
        qualified_name: file_qn_full.clone(),
        name: file_name,
        source_uri: Some(file_path.to_string()),
        line_start: 0,
        line_end: lines,
        source_text: Some(source_str.clone()),
        properties: vec![("dialect".to_string(), "go".to_string())],
    });

    // Primary extraction query
    let query = Query::new(
        &tree_sitter_go::LANGUAGE.into(),
        r#"
        [
            (function_declaration) @function
            (method_declaration) @method
            (struct_declaration) @struct
            (interface_declaration) @interface
            (type_declaration name: (identifier)) @type_decl
            (var_declaration spec: (identifier) @var) @var_decl
            (const_declaration spec: (identifier) @const) @const_decl
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
                "function" | "method" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }

                    let is_test = file_is_test || is_test_name(&name);
                    let is_method = cn == "method";
                    let kind = if is_method { "method" } else { "function" };

                    let mut props = vec![("dialect".to_string(), "go".to_string())];
                    if is_test {
                        props.push(("is_test".to_string(), "true".to_string()));
                    }

                    // For methods, extract receiver type
                    let receiver_type = if is_method {
                        extract_go_receiver(&node, source.as_ref())
                    } else {
                        None
                    };

                    let qn = if is_method {
                        match &receiver_type {
                            Some(rt) => format!("{}::{}::{}", file_qn_full, rt, name),
                            None => format!("{}::{}", file_qn_full, name),
                        }
                    } else {
                        format!("{}::{}", file_qn_full, name)
                    };

                    result.nodes.push(ExtractedNode {
                        kind: kind.to_string(),
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

                    // If we extracted a receiver type, add it as a node
                    if let Some(rt) = receiver_type {
                        let type_qn = format!("{}::{}", file_qn_full, rt);
                        if !result.nodes.iter().any(|n| n.qualified_name == type_qn) {
                            result.nodes.push(ExtractedNode {
                                kind: "class".to_string(),
                                qualified_name: type_qn.clone(),
                                name: rt.clone(),
                                source_uri: None,
                                line_start: 0,
                                line_end: 0,
                                source_text: None,
                                properties: vec![("dialect".to_string(), "go".to_string())],
                            });
                            result.edges.push(ExtractedEdge {
                                src_qn: file_qn_full.clone(),
                                dst_qn: type_qn,
                                kind: "defines".to_string(),
                                conf_class: "extracted".to_string(),
                                confidence: 1.0,
                                properties: vec![],
                            });
                        }
                    }
                }
                "struct" => {
                    let Some(name_node) = node
                        .child_by_field_name("type")
                        .and_then(|t| t.child_by_field_name("name"))
                    else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: "class".to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "go".to_string())],
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
                "interface" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: "interface".to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "go".to_string())],
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
                "type_decl" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: "type".to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "go".to_string())],
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
                "var" => {
                    let name = text(
                        &node.child_by_field_name("spec").unwrap_or(node),
                        source.as_ref(),
                    );
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: "variable".to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "go".to_string())],
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
                "const" => {
                    let name = text(
                        &node.child_by_field_name("spec").unwrap_or(node),
                        source.as_ref(),
                    );
                    if name.is_empty() {
                        continue;
                    }
                    let qn = format!("{}::{}", file_qn_full, name);
                    result.nodes.push(ExtractedNode {
                        kind: "constant".to_string(),
                        qualified_name: qn.clone(),
                        name: name.clone(),
                        source_uri: Some(file_path.to_string()),
                        line_start: start,
                        line_end: end,
                        source_text: src_text,
                        properties: vec![("dialect".to_string(), "go".to_string())],
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
                _ => {}
            }
        }
    }

    // Extract imports
    extract_imports(
        &root,
        &file_qn_full,
        file_path,
        source.as_ref(),
        &mut result,
    );

    result_to_dict(py, &result)
}

/// Extract Go imports (package imports and aliased imports)
fn extract_imports(
    root: &tree_sitter::Node,
    file_qn: &str,
    _file_path: &str,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let import_query = Query::new(
        &tree_sitter_go::LANGUAGE.into(),
        r#"
        (import_declaration
            path: (import_spec path: (interpreted_string_literal) @path)
        ) @import
        (import_declaration
            path: (import_spec
                name: (identifier) @alias
                path: (interpreted_string_literal) @path
            )
        ) @import2
        "#,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e)))
    .ok();

    if let Some(import_query) = import_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&import_query, *root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let cn = import_query.capture_names()[cap.index as usize];
                if matches!(cn, "path") {
                    let path_text = text(&cap.node, source);
                    if !path_text.is_empty() {
                        // Extract module name from path
                        let mod_name = path_text.trim_matches('"');
                        let mod_qn = format!("module::{}", mod_name);
                        if !result.nodes.iter().any(|n| n.qualified_name == mod_qn) {
                            result.nodes.push(ExtractedNode {
                                kind: "module".to_string(),
                                qualified_name: mod_qn.clone(),
                                name: mod_name.to_string(),
                                source_uri: None,
                                line_start: 0,
                                line_end: 0,
                                source_text: None,
                                properties: vec![("dialect".to_string(), "go".to_string())],
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
}

/// Extract receiver type from a method declaration
fn extract_go_receiver(node: &tree_sitter::Node, source: &[u8]) -> Option<String> {
    // method_declaration has a "parameter" field for the receiver
    let param_field = node.child_by_field_name("parameter")?;
    for child in param_field.children(&mut node.walk()) {
        if child.kind() == "parameter_declaration" {
            // Get the type
            if let Some(type_field) = child.child_by_field_name("type") {
                return Some(text(&type_field, source));
            }
        }
    }
    None
}

/// Emit call placeholders from a node
fn emit_calls(
    node: &tree_sitter::Node,
    source: &[u8],
    caller_qn: &str,
    result: &mut ExtractionResult,
) {
    let mut stack: Vec<tree_sitter::Node> = node.children(&mut node.walk()).collect();
    while let Some(child) = stack.pop() {
        if child.kind() == "call_expression" {
            let func_node = child.child_by_field_name("function");
            if let Some(func) = func_node {
                let mut name = None;
                match func.kind() {
                    "identifier" => {
                        name = Some(text(&func, source));
                    }
                    "selector_expression" => {
                        if let Some(sel_name) = func.child_by_field_name("field") {
                            name = Some(text(&sel_name, source));
                        }
                    }
                    _ => {}
                }
                if let Some(n) = name {
                    if !n.starts_with('_') && !should_suppress_call_placeholder(&n) {
                        result.calls.push(CallPlaceholder {
                            caller_qn: caller_qn.to_string(),
                            callee_qn: format!("call::{}", n),
                            receiver: None,
                        });
                    }
                }
            }
        }
        stack.extend(child.children(&mut child.walk()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_test_file() {
        assert!(is_test_file_path("foo_test.go"));
        assert!(!is_test_file_path("foo.go"));
    }

    #[test]
    fn test_is_test_name() {
        assert!(is_test_name("TestFoo"));
        assert!(is_test_name("BenchmarkBar"));
        assert!(!is_test_name("Foo"));
    }

    #[test]
    fn test_suppression() {
        assert!(should_suppress_call_placeholder("print"));
        assert!(should_suppress_call_placeholder("len"));
        assert!(!should_suppress_call_placeholder("login"));
    }
}
