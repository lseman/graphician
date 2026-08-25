//! Rust source extraction.
//! Emits: File, Function, Method, Trait, Struct, Enum, Impl, Const, Static, TypeAlias, Union, Macro, Module
//! Edges: Defines, Calls, Imports, Implements, Uses

use pyo3::prelude::*;
use tree_sitter::{Query, QueryCursor, StreamingIterator};

use super::{children, result_to_dict, text};
use super::{CallPlaceholder, ExtractedEdge, ExtractedNode, ExtractionResult};
use crate::{is_test_file_path, is_test_name, should_suppress_call_placeholder};

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
pub fn extract_rust_file(
    py: Python,
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> PyResult<PyObject> {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
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
        properties: vec![("dialect".to_string(), "rust".to_string())],
    });

    // Primary extraction query
    let query = Query::new(
        &tree_sitter_rust::LANGUAGE.into(),
        r#"
        [
            (function_item) @function
            (trait_item) @trait
            (struct_item) @struct
            (enum_item) @enum
            (mod_item) @module
            (union_item) @union
            (const_item) @const
            (static_item) @static
            (type_item) @type_alias
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
                "function" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
                    if name.is_empty() {
                        continue;
                    }

                    // Determine scope
                    let scope = rust_scope(&node, source.as_ref());
                    let qn = if scope.is_empty() {
                        format!("{}::{}", file_qn_full, name)
                    } else {
                        format!("{}::{}::{}", file_qn_full, scope.join("::"), name)
                    };

                    // Determine kind
                    let kind = if has_method_parent(&node) {
                        "method"
                    } else {
                        "function"
                    };

                    let is_test = file_is_test || is_test_name(&name);
                    let mut props = vec![("dialect".to_string(), "rust".to_string())];
                    if is_test {
                        props.push(("is_test".to_string(), "true".to_string()));
                    }

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
                }
                "trait" | "struct" | "enum" => {
                    let kind = match cn {
                        "trait" => "trait",
                        "enum" => "type",
                        _ => "class",
                    };
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
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
                "union" => {
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
                "const" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
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
                "static" => {
                    let Some(name_node) = node.child_by_field_name("name") else {
                        continue;
                    };
                    let name = text(&name_node, source.as_ref());
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
                "type_alias" => {
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
                _ => {}
            }
        }
    }

    // Extract impl blocks and methods
    impl_extract(
        &root,
        &file_qn_full,
        file_path,
        source.as_ref(),
        &mut result,
    );

    // Extract macro_rules! definitions
    extract_macros(
        &root,
        &file_qn_full,
        file_path,
        source.as_ref(),
        &mut result,
    );

    // Extract use imports
    emit_imports_rust(&root, &file_qn_full, source.as_ref(), &mut result);

    result_to_dict(py, &result)
}

/// Extract impl blocks with their methods and trait bounds
fn impl_extract(
    root: &tree_sitter::Node,
    file_qn: &str,
    file_path: &str,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let impl_query = Query::new(
        &tree_sitter_rust::LANGUAGE.into(),
        r#"
        (impl_item
            subject: (type) @type
            body: (declaration_list
                (function_item) @method
            )
        ) @impl_block
        (impl_item
            trait: (scoped_identifier) @trait
            subject: (type) @type2
            body: (declaration_list
                (function_item) @method2
            )
        ) @impl_block2
        "#,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e)))
    .ok();

    if let Some(impl_query) = impl_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&impl_query, *root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let cn = impl_query.capture_names()[cap.index as usize];
                if matches!(cn, "impl_block" | "impl_block2") {
                    let impl_node = cap.node;
                    // Get the type being implemented
                    let type_field = impl_node
                        .child_by_field_name("subject")
                        .or_else(|| impl_node.child_by_field_name("type"));
                    if let Some(type_node) = type_field {
                        let type_name = text(&type_node, source);
                        if !type_name.is_empty() {
                            let qn = format!("{}::{}", file_qn, type_name);
                            // Track unique impl blocks
                            if !result
                                .nodes
                                .iter()
                                .any(|n| n.qualified_name == qn && n.kind == "class")
                            {
                                let start = (impl_node.start_position().row + 1) as usize;
                                let end = (impl_node.end_position().row + 1) as usize;
                                result.nodes.push(ExtractedNode {
                                    kind: "class".to_string(),
                                    qualified_name: qn.clone(),
                                    name: type_name.to_string(),
                                    source_uri: Some(file_path.to_string()),
                                    line_start: start,
                                    line_end: end,
                                    source_text: super::truncated_source_text(
                                        &String::from_utf8_lossy(source),
                                        start,
                                        end,
                                    ),
                                    properties: vec![("dialect".to_string(), "rust".to_string())],
                                });
                                result.edges.push(ExtractedEdge {
                                    src_qn: file_qn.to_string(),
                                    dst_qn: qn.clone(),
                                    kind: "defines".to_string(),
                                    conf_class: "extracted".to_string(),
                                    confidence: 1.0,
                                    properties: vec![],
                                });
                            }
                            // Check for trait impl
                            if let Some(trait_node) = impl_node.child_by_field_name("trait") {
                                if let Some(trait_path) = trait_node.child_by_field_name("path") {
                                    let trait_text = text(&trait_path, source);
                                    if !trait_text.is_empty() {
                                        let trait_qn = format!("trait::{}", trait_text);
                                        result.nodes.push(ExtractedNode {
                                            kind: "trait".to_string(),
                                            qualified_name: trait_qn.clone(),
                                            name: trait_text.to_string(),
                                            source_uri: None,
                                            line_start: 0,
                                            line_end: 0,
                                            source_text: None,
                                            properties: vec![(
                                                "dialect".to_string(),
                                                "rust".to_string(),
                                            )],
                                        });
                                        result.edges.push(ExtractedEdge {
                                            src_qn: qn.clone(),
                                            dst_qn: trait_qn,
                                            kind: "implements".to_string(),
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
                // Note: methods inside impl blocks are already extracted by the
                // primary query (via has_method_parent), so we skip them here
                // to avoid duplicates.
            }
        }
    }
}

/// Extract macro_rules! definitions
fn extract_macros(
    root: &tree_sitter::Node,
    file_qn: &str,
    file_path: &str,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    let macro_query = Query::new(
        &tree_sitter_rust::LANGUAGE.into(),
        r#"(macro_definition name: (identifier) @name) @macro"#,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e)))
    .ok();

    if let Some(macro_query) = macro_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&macro_query, *root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let cn = macro_query.capture_names()[cap.index as usize];
                if cn == "macro" {
                    if let Some(name_node) = cap.node.child_by_field_name("name") {
                        let name = text(&name_node, source);
                        if !name.is_empty() {
                            let start = (cap.node.start_position().row + 1) as usize;
                            let end = (cap.node.end_position().row + 1) as usize;
                            let qn = format!("{}::{}", file_qn, name);
                            result.nodes.push(ExtractedNode {
                                kind: "macro".to_string(),
                                qualified_name: qn.clone(),
                                name: name.clone(),
                                source_uri: Some(file_path.to_string()),
                                line_start: start,
                                line_end: end,
                                source_text: super::truncated_source_text(
                                    &String::from_utf8_lossy(source),
                                    start,
                                    end,
                                ),
                                properties: vec![("dialect".to_string(), "rust".to_string())],
                            });
                            result.edges.push(ExtractedEdge {
                                src_qn: file_qn.to_string(),
                                dst_qn: qn,
                                kind: "defines".to_string(),
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

/// Improved import extraction for Rust use statements
fn emit_imports_rust(
    root: &tree_sitter::Node,
    file_qn: &str,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    // Query for use statements with various patterns
    let use_query = Query::new(
        &tree_sitter_rust::LANGUAGE.into(),
        r#"(use_declaration path: (scoped_identifier path: (ident)* @path name: (ident) @name))"#,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("invalid query: {}", e)))
    .ok();

    if let Some(use_query) = use_query {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&use_query, *root, source);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                let cn = use_query.capture_names()[cap.index as usize];
                if matches!(cn, "path" | "name") {
                    let path_text = text(&cap.node, source);
                    if !path_text.is_empty() && !path_text.starts_with('"') {
                        let mod_qn = format!("module::{}", path_text);
                        if !result.nodes.iter().any(|n| n.qualified_name == mod_qn) {
                            result.nodes.push(ExtractedNode {
                                kind: "module".to_string(),
                                qualified_name: mod_qn.clone(),
                                name: path_text.to_string(),
                                source_uri: None,
                                line_start: 0,
                                line_end: 0,
                                source_text: None,
                                properties: vec![("dialect".to_string(), "rust".to_string())],
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

fn extract_decorators(_node: &tree_sitter::Node, _source: &[u8]) -> Vec<String> {
    Vec::new()
}

fn has_method_parent(node: &tree_sitter::Node) -> bool {
    // Check grandparent because impl_item -> declaration_list -> function_item
    if let Some(parent) = node.parent() {
        if let Some(grandparent) = parent.parent() {
            if matches!(grandparent.kind(), "impl_item") {
                return true;
            }
        }
    }
    false
}

fn rust_scope(node: &tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut scope = Vec::new();
    let mut current = node.parent();
    while let Some(parent) = current {
        match parent.kind() {
            "mod_item" => {
                let name_node = parent.child_by_field_name("name");
                if let Some(name_node) = name_node {
                    scope.push(text(&name_node, source));
                }
            }
            "use_declaration" => {
                // Skip use declarations
            }
            "impl_item" => {
                // Get the type being implemented (e.g., "A" from "impl A")
                let type_node = parent
                    .child_by_field_name("subject")
                    .or_else(|| parent.child_by_field_name("type"));
                if let Some(type_node) = type_node {
                    let type_name = text(&type_node, source);
                    if !type_name.is_empty() {
                        scope.push(type_name.to_string());
                    }
                }
            }
            "trait_item" => {
                // Get the trait name
                let name_node = parent.child_by_field_name("name");
                if let Some(name_node) = name_node {
                    scope.push(text(&name_node, source));
                }
            }
            _ => {}
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
            let func_node = child.child_by_field_name("function");
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
                        if let Some(path) = func.child_by_field_name("path") {
                            receiver = Some(text(&path, source));
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
        } else if child.kind() == "macro_invocation" {
            // Extract calls from inside macro invocations (e.g., assert!(login()))
            emit_macro_calls(&child, source, caller_qn, result);
        }
        stack.extend(children(&child));
    }
}

/// Extract call placeholders from inside macro invocations.
/// Handles patterns like: macro_name!(identifier(), identifier2())
fn emit_macro_calls(
    node: &tree_sitter::Node,
    source: &[u8],
    caller_qn: &str,
    result: &mut ExtractionResult,
) {
    let mut stack: Vec<tree_sitter::Node> = children(node);
    while let Some(child) = stack.pop() {
        if child.kind() == "token_tree" {
            let kids: Vec<tree_sitter::Node> = child.children(&mut child.walk()).collect();
            for i in 0..kids.len().saturating_sub(1) {
                if kids[i].kind() != "identifier" {
                    continue;
                }
                let nxt = &kids[i + 1];
                if nxt.kind() != "token_tree" {
                    continue;
                }
                let inner: Vec<tree_sitter::Node> = nxt.children(&mut nxt.walk()).collect();
                if inner.is_empty() || inner[0].kind() != "(" {
                    continue;
                }
                let name = text(&kids[i], source);
                // Skip Rust keywords that can appear as macro arguments
                if matches!(
                    name.to_lowercase().as_str(),
                    "return"
                        | "if"
                        | "else"
                        | "let"
                        | "match"
                        | "for"
                        | "while"
                        | "loop"
                        | "in"
                        | "mut"
                        | "ref"
                        | "as"
                        | "move"
                ) {
                    continue;
                }
                if !name.starts_with('_') && !should_suppress_call_placeholder(&name) {
                    result.calls.push(CallPlaceholder {
                        caller_qn: caller_qn.to_string(),
                        callee_qn: format!("call::{}", name),
                        receiver: None,
                    });
                }
            }
        }
        stack.extend(child.children(&mut child.walk()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{is_generic_name, should_suppress};

    #[test]
    fn should_suppress_python_builtins() {
        assert!(should_suppress_call_placeholder("print"));
        // len is no longer suppressed — stub resolution handles it
        assert!(should_suppress_call_placeholder("range"));
        assert!(should_suppress_call_placeholder("super"));
        assert!(should_suppress_call_placeholder("isinstance"));
    }

    #[test]
    fn should_suppress_rust_std() {
        assert!(should_suppress_call_placeholder("and_then"));
        // unwrap/collect/to_string moved to stub coverage — no longer suppressed
        assert!(should_suppress_call_placeholder("expect"));
    }

    #[test]
    fn should_suppress_tree_sitter_api() {
        assert!(should_suppress_call_placeholder("child_by_field_name"));
        assert!(should_suppress_call_placeholder("root_node"));
        assert!(should_suppress_call_placeholder("start_position"));
        assert!(should_suppress_call_placeholder("walk"));
    }

    #[test]
    fn should_not_suppress_project_functions() {
        assert!(!should_suppress_call_placeholder("login"));
        assert!(!should_suppress_call_placeholder("extract_file"));
        assert!(!should_suppress_call_placeholder("add_node"));
        assert!(!should_suppress_call_placeholder("add_edge"));
    }

    #[test]
    fn is_generic_name_flags_common_verbs() {
        // get/new removed — stub coverage handles them
        assert!(is_generic_name("execute"));
        assert!(is_generic_name("select"));
        assert!(is_generic_name("merge"));
        assert!(is_generic_name("path"));
    }

    #[test]
    fn is_generic_name_does_not_flag_strict_list() {
        assert!(!is_generic_name("child_by_field_name"));
        assert!(!is_generic_name("printf"));
    }

    #[test]
    fn combined_suppression_catches_both_tiers() {
        // Strict list — unwrap/collect no longer suppressed, use and_then
        assert!(should_suppress("and_then"));
        // get/new removed from generic name — stub coverage handles them
        assert!(should_suppress("execute"));
        // Real project function — should NOT be suppressed
        assert!(!should_suppress("login"));
        assert!(!should_suppress("extract_file"));
    }
}
