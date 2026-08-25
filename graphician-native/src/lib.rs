//! High-performance Python code extraction using tree-sitter Rust bindings.
pub mod analysis;
pub mod graph;
pub mod persistence;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde_json::json;
use tree_sitter::{Node, Query, QueryCursor, StreamingIterator};

/// Safe text extraction from a tree-sitter node via byte-slicing.
pub fn node_text<'a>(node: &Node<'a>, source: &'a [u8]) -> &'a [u8] {
    if node.byte_range().is_empty() {
        &[]
    } else {
        &source[node.byte_range()]
    }
}

pub fn node_text_str<'a>(node: &Node<'a>, source: &'a [u8]) -> String {
    String::from_utf8_lossy(node_text(node, source)).into_owned()
}

pub fn extract_name(node: &Node, source: &[u8]) -> Option<String> {
    let text = node_text(node, source);
    if text.is_empty() {
        return None;
    }
    let s = String::from_utf8_lossy(text);
    if s.trim().is_empty() {
        None
    } else {
        Some(s.into_owned())
    }
}

pub fn child_by_field<'a>(node: &'a Node<'a>, field: &str) -> Option<Node<'a>> {
    node.child_by_field_name(field)
}

fn is_protocol_class(bases: Option<&Node>, source: &[u8]) -> bool {
    let bases = match bases {
        Some(node) => node,
        None => return false,
    };
    let protocol_bases = [
        "Protocol",
        "ABC",
        "typing.Protocol",
        "typing_extensions.Protocol",
    ];
    for base in bases.children(&mut bases.walk()) {
        match base.kind() {
            "identifier" => {
                let name = node_text_str(&base, source);
                if protocol_bases.contains(&name.as_str()) {
                    return true;
                }
            }
            "attribute" => {
                if let Some(attr) = child_by_field(&base, "attribute") {
                    let name = node_text_str(&attr, source);
                    if protocol_bases.contains(&name.as_str()) {
                        return true;
                    }
                }
            }
            "dotted_name" => {
                let name = node_text_str(&base, source);
                if protocol_bases.contains(&name.as_str()) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn is_typeddict_class(bases: Option<&Node>, source: &[u8]) -> bool {
    let bases = match bases {
        Some(node) => node,
        None => return false,
    };
    for base in bases.children(&mut bases.walk()) {
        match base.kind() {
            "identifier" | "dotted_name" => {
                let name = node_text_str(&base, source);
                if name.contains("TypedDict") {
                    return true;
                }
            }
            "attribute" => {
                if let Some(attr) = child_by_field(&base, "attribute") {
                    let name = node_text_str(&attr, source);
                    if name.contains("TypedDict") {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn is_typevar_definition(body: Option<&Node>, source: &[u8]) -> bool {
    let body = match body {
        Some(b) => b,
        None => return false,
    };
    for child in body.children(&mut body.walk()) {
        if child.kind() == "expression_statement" {
            if let Some(expr) = child.children(&mut child.walk()).next() {
                if expr.kind() == "call" {
                    if let Some(func) = child_by_field(&expr, "function") {
                        let name = node_text_str(&func, source);
                        if matches!(
                            name.as_str(),
                            "TypeVar" | "ParamSpec" | "TypeVarTuple" | "GenericAlias"
                        ) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Extract decorator text from a decorator node.
/// Handles simple identifiers (@dataclass) and attributes (pytest.fixture).
pub fn extract_decorator_name(dec: &Node, source: &[u8]) -> String {
    for c in dec.children(&mut dec.walk()) {
        if c.kind() == "@" {
            continue;
        }
        // Try to get the full dotted name
        match c.kind() {
            "identifier" => return node_text_str(&c, source).trim().to_string(),
            "attribute" => {
                // Build dotted name: object.attribute
                let parts: Vec<String> = c
                    .children(&mut c.walk())
                    .filter(|ch| ch.kind() != "." && ch.kind() != "@")
                    .map(|ch| node_text_str(&ch, source))
                    .collect();
                if !parts.is_empty() {
                    return parts.join(".");
                }
            }
            "call" => {
                // Handle @decorator(args) - extract decorator name only
                if let Some(func) = child_by_field(&c, "function") {
                    match func.kind() {
                        "identifier" => return node_text_str(&func, source).trim().to_string(),
                        "attribute" => {
                            let parts: Vec<String> = func
                                .children(&mut func.walk())
                                .filter(|ch| ch.kind() != "." && ch.kind() != "@")
                                .map(|ch| node_text_str(&ch, source))
                                .collect();
                            if !parts.is_empty() {
                                return parts.join(".");
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    String::new()
}

pub fn extract_decorators(node: &Node, source: &[u8]) -> Vec<String> {
    let mut decorators = Vec::new();
    for child in node.children(&mut node.walk()) {
        // Handle both decorated_definition and decorated node types
        if child.kind() == "decorated_definition" || child.kind() == "decorated" {
            for c in child.children(&mut child.walk()) {
                if c.kind() == "decorator" {
                    let name = extract_decorator_name(&c, source);
                    if !name.is_empty() {
                        decorators.push(name);
                    }
                }
            }
        } else if child.kind() == "decorator" {
            // Direct decorator child (fallback)
            let name = extract_decorator_name(&child, source);
            if !name.is_empty() {
                decorators.push(name);
            }
        }
    }
    decorators
}

#[derive(Debug, Clone)]
pub struct ExtractedNode {
    kind: String,
    qualified_name: String,
    name: String,
    source_uri: Option<String>,
    line_start: usize,
    line_end: usize,
    source_text: Option<String>,
    properties: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct ExtractedEdge {
    src_qn: String,
    dst_qn: String,
    kind: String,
    conf_class: String,
    confidence: f64,
    properties: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct CallPlaceholder {
    caller_qn: String,
    callee_qn: String,
    receiver: Option<String>,
}

#[derive(Debug)]
pub struct ExtractionResult {
    nodes: Vec<ExtractedNode>,
    edges: Vec<ExtractedEdge>,
    calls: Vec<CallPlaceholder>,
}

/// Heuristics for recognising test code.
pub fn is_test_file_path(path: &str) -> bool {
    let s = path.replace('\\', "/");

    // Directory components
    if s.contains("/tests/")
        || s.contains("/test/")
        || s.starts_with("tests/")
        || s.starts_with("test/")
        || s.contains("/__tests__/")
        || s.starts_with("__tests__/")
        || s.contains("/spec/")
        || s.starts_with("spec/")
    {
        return true;
    }

    let stem = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    if stem.starts_with("test_")
        || stem.ends_with("_test")
        || stem.ends_with("_spec")
        || stem.ends_with(".test")
        || stem.ends_with(".spec")
    {
        return true;
    }

    let stem_lower = stem.to_ascii_lowercase();
    if stem_lower.starts_with("test_helper") || stem_lower.starts_with("test_helpers") {
        return true;
    }

    // Extension-gated suffix conventions
    match ext {
        "java" | "cs" | "php" => stem.ends_with("Test") || stem.ends_with("Tests"),
        "kt" | "swift" => {
            stem.ends_with("Test") || stem.ends_with("Tests") || stem.ends_with("Spec")
        }
        "scala" => stem.ends_with("Spec") || stem.ends_with("Suite") || stem.ends_with("Test"),
        "dart" => stem_lower.starts_with("test_") || stem_lower.ends_with("_test"),
        "lua" => {
            stem_lower.starts_with("test_")
                || stem_lower.ends_with("_test")
                || stem_lower.ends_with("_spec")
        }
        _ => false,
    }
}

/// Test the *symbol name*. True if the function/method name follows a
/// test convention.
pub fn is_test_name(name: &str) -> bool {
    if name.starts_with("test_") || name.starts_with("Test_") {
        return true;
    }
    if name.ends_with("_test") || name.ends_with("_spec") {
        return true;
    }
    // `XxxTest` — only if there's something *before* `Test`
    if name.len() > 4 && name.ends_with("Test") {
        return true;
    }
    // BDD style: `should*`, `it*`, `given*` followed by capital
    let starts_camel = |prefix: &str| -> bool {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.chars().next())
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false)
    };
    if starts_camel("should") || starts_camel("it") || starts_camel("given") {
        return true;
    }
    // `Test[A-Z]…`: TestLogin, TestAuthRefresh, etc.
    if name
        .strip_prefix("Test")
        .and_then(|rest| rest.chars().next())
        .map(|c| c.is_ascii_uppercase())
        .unwrap_or(false)
    {
        return true;
    }
    false
}

pub fn should_suppress_call_placeholder(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        // Python builtins and common constructors.
        "abs" | "all" | "any" | "bool" | "bytes" | "callable"
            | "dict" | "dir" | "enumerate" | "float"
            | "getattr" | "hasattr" | "hash" | "id" | "int" | "isinstance"
            | "iter" | "list" | "max" | "min" | "next"
            | "open" | "print" | "range" | "repr" | "reversed" | "round"
            | "set" | "sorted" | "str" | "sum" | "super" | "tuple"
            | "type" | "vars" | "zip"
        // Rust/std/common fluent API calls (still suppressed).
        | "and_then" | "as_bytes" | "as_deref" | "chars"
        | "clamp" | "ends_with" | "err" | "filter_map"
        | "flat_map" | "fold" | "get_mut" | "into_iter"
        | "is_empty" | "is_none" | "is_some_and"
        | "iter_mut" | "lines" | "map_err" | "none" | "ok" | "ok_or"
        | "ok_or_else" | "or_default" | "push_str" | "rsplit" | "some"
        | "splitn" | "starts_with" | "to_owned"
        | "to_string_lossy" | "from_str" | "trim" | "with_capacity"
        // std::collections / std::vec methods
        | "truncate"
        | "reserve" | "concat" | "to_vec"
        // std::io / std::fs methods
        | "read_to_string" | "read_to_end" | "remove_dir_all"
        | "remove_file" | "create_dir_all" | "exists" | "write_all"
        | "flush"
        // std trait methods
        | "fg"
        // std::env
        | "temp_dir" | "args"
        // std::string methods
        | "strip_prefix" | "to_ascii_lowercase" | "trim_matches"
        // std::option
        | "is_some" | "or_else"
        // std::collections
        | "values_mut" | "borrow"
        // std::collections
        | "pop_front" | "push_back" | "to_lowercase" | "to_uppercase"
        | "split_whitespace" | "saturating_sub" | "to_string_pretty"
        | "as_array" | "as_u64"
        // std::path
        | "file_name" | "file_stem"
        // std::num
        | "wrapping_add"
        // std::fs
        | "current_dir"
        // serde_json
        | "as_bool" | "as_f64" | "as_object"
        // external lib symbols
        | "render_widget" | "highlight_style" | "border_style" | "borders"
        | "checkAvailable" | "percentage" | "strip_suffix"
        | "trim_end_matches" | "chunks" | "or_insert" | "or_insert_with"
        // SQLite rusqlite bindings
        | "query_map" | "prepare" | "transaction" | "selected"
        | "query_row" | "add_modifier"
        // std::time methods
        | "duration_since" | "as_nanos"
        // std::num
        | "saturating_add" | "wrapping_mul"
        // std::path
        | "extension"
        // Confidence enum variant leaking as unresolved
        | "inferred"
        // Common graph-library traversal/mutation helpers.
        | "node_weight" | "node_weight_mut"
        // std::fs::DirEntry / std::path methods
        | "to_path_buf" | "is_dir" | "is_file" | "filter_entry"
        // C/C++ and libc-style calls.
        | "malloc" | "free" | "printf" | "fprintf" | "memcpy" | "memset"
        | "strlen" | "strcmp" | "std"
        // tree-sitter Node API
        | "child_by_field_name" | "children" | "end_position"
        | "is_named" | "language_typescript" | "language_tsx" | "root_node"
        | "start_position" | "walk" | "utf8_text"
        // tree-sitter Parser API
        | "set_language" | "included_ranges"
        // Additional tree-sitter / std methods
        | "rev" | "nth" | "to_str" | "last_mut" | "trim_start"
        | "from_utf8" | "windows" | "end_byte" | "start_byte"
        | "is_ascii_digit" | "is_lowercase" | "is_uppercase"
        | "is_alphanumeric" | "new_ext" | "reverse" | "next_back"
        | "as_array_mut"
        // Ariadne internal methods
        | "resolve_mentions" | "original_nodes" | "edges_mut" | "qname_index"
        // Additional stdlib/external
        | "is_err" | "is_ok" | "read_dir" | "from_secs" | "trim_start_matches"
        | "from_utf8_lossy" | "Object" | "json"
        // Python builtins not in original list
        | "property" | "staticmethod" | "classmethod" | "issubclass"
        | "setattr" | "delattr" | "bytearray" | "frozenset"
        | "divmod" | "chr" | "ord" | "hex" | "oct" | "bin" | "format"
        | "input" | "compile" | "eval" | "exec" | "help" | "memoryview"
        | "slice" | "complex" | "object" | "breakpoint"
    )
}

/// Common short English verbs/nouns that dominate noise in *most*
/// codebases as stdlib/fluent-API calls, but collide with real method
/// names often enough (`Repository::select`, `Service::execute`,
/// `Config::merge`) that they must not be dropped before the project's
/// symbol table exists.
pub fn is_generic_name(name: &str) -> bool {
    let name = name.trim();
    if name.is_empty() {
        return true;
    }
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "find"
            | "select"
            | "execute"
            | "merge"
            | "load"
            | "write"
            | "read"
            | "path"
            | "string"
            | "index"
            | "take"
            | "has"
            | "display"
            | "now"
            | "entry"
            | "default"
            | "count"
            | "first"
            | "last"
            | "position"
            | "split"
            | "replace"
            | "clear"
            | "values"
            | "node"
            | "text"
            | "parse"
            | "kind"
            | "parent"
            | "language"
            | "status"
            | "watch"
            | "commit"
            | "block"
            | "attr"
    )
}

/// Combined suppression: strict list or generic name.
pub fn should_suppress(name: &str) -> bool {
    should_suppress_call_placeholder(name) || is_generic_name(name)
}

/// Import text capped at 10KB to bound memory for large nodes.
pub fn truncated_source_text(source: &str, line_start: usize, line_end: usize) -> Option<String> {
    if line_start == 0 || line_end == 0 || line_end <= line_start {
        return None;
    }
    let lines: Vec<&str> = source.lines().collect();
    // tree-sitter rows are 1-indexed (start) → subtract 1 for 0-indexed vec.
    let s = (line_start - 1).min(lines.len());
    let e = line_end.min(lines.len());
    if s >= lines.len() || s >= e {
        return None;
    }
    let text: String = lines[s..e].join("\n");
    if text.is_empty() {
        return None;
    }
    if text.len() > 10_000 {
        Some(text.get(..10_000).unwrap_or(&text).to_string())
    } else {
        Some(text)
    }
}

/// Extract imports using a tree-sitter Query — faster and more robust
/// than manual child iteration.
fn emit_imports(root: &Node, source: &[u8], file_qn: &str, result: &mut ExtractionResult) {
    let query = Query::new(
        &tree_sitter_python::LANGUAGE.into(),
        r#"
        [
          (import_statement name: (dotted_name) @path)
          (import_statement (aliased_import (dotted_name) @path))
          (import_from_statement module_name: (dotted_name) @path)
        ]
        "#,
    )
    .expect("import query is valid");
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, *root, source);
    while let Some(m) = matches.next() {
        for cap in m.captures {
            if cap.node.kind() != "dotted_name" {
                continue;
            }
            let path_text = node_text_str(&cap.node, source);
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
                properties: vec![("dialect".to_string(), "python".to_string())],
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

fn emit_calls(
    node: &Node,
    source: &[u8],
    caller_qn: &str,
    result: &mut ExtractionResult,
    suppress_types: &[&str],
) {
    let mut stack: Vec<Node> = node.children(&mut node.walk()).collect();
    while let Some(child) = stack.pop() {
        if suppress_types.contains(&child.kind()) {
            continue;
        }
        if child.kind() == "call" {
            let mut func_node = None;
            for c in child.children(&mut child.walk()) {
                if child.field_name_for_child(c.id() as u32) == Some("function") {
                    func_node = Some(c);
                    break;
                }
            }
            if func_node.is_none() {
                func_node = child.children(&mut child.walk()).next();
            }
            if let Some(func) = func_node {
                let mut name = None;
                let mut receiver = None;
                match func.kind() {
                    "identifier" => {
                        name = Some(node_text_str(&func, source));
                    }
                    "attribute" => {
                        if let Some(attr) = child_by_field(&func, "attribute") {
                            name = Some(node_text_str(&attr, source));
                        }
                        if let Some(obj) = child_by_field(&func, "object") {
                            receiver = Some(node_text_str(&obj, source));
                        }
                    }
                    _ => {}
                }
                if let Some(n) = name {
                    if !n.starts_with('_') && !should_suppress_call_placeholder(&n) {
                        let callee_qn = format!("call::{}", n);
                        result.calls.push(CallPlaceholder {
                            caller_qn: caller_qn.to_string(),
                            callee_qn,
                            receiver,
                        });
                    }
                }
            }
        }
        stack.extend(child.children(&mut child.walk()).collect::<Vec<_>>());
    }
}

fn walk_scope(
    node: &Node,
    file_qn: &str,
    path: &str,
    parent_qn: &str,
    scope: &[String],
    parent_is_class: bool,
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
) {
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "decorated_definition" => {
                // Extract decorators from decorated_definition
                let decorators = extract_decorators(&child, source);
                // Find the actual definition (class/function) inside decorated_definition
                for def in child.children(&mut child.walk()) {
                    match def.kind() {
                        "class_definition" => handle_class(
                            &def,
                            file_qn,
                            path,
                            parent_qn,
                            scope,
                            file_is_test,
                            source,
                            result,
                            Some(&decorators),
                        ),
                        "function_definition" => handle_function(
                            &def,
                            file_qn,
                            path,
                            parent_qn,
                            scope,
                            parent_is_class,
                            file_is_test,
                            source,
                            result,
                            Some(&decorators),
                        ),
                        _ => {}
                    }
                }
            }
            "class_definition" => handle_class(
                &child,
                file_qn,
                path,
                parent_qn,
                scope,
                file_is_test,
                source,
                result,
                None,
            ),
            "function_definition" => handle_function(
                &child,
                file_qn,
                path,
                parent_qn,
                scope,
                parent_is_class,
                file_is_test,
                source,
                result,
                None,
            ),
            _ => {}
        }
    }
}

fn handle_class(
    node: &Node,
    file_qn: &str,
    path: &str,
    parent_qn: &str,
    scope: &[String],
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
    provided_decorators: Option<&Vec<String>>,
) {
    let name_node = match child_by_field(node, "name") {
        Some(n) => n,
        None => return,
    };
    let name = match extract_name(&name_node, source) {
        Some(n) => n,
        None => return,
    };
    let child_scope: Vec<String> = scope.iter().cloned().chain(Some(name.clone())).collect();
    let qn = format!("{}::{}", file_qn, child_scope.join("::"));
    let mut props: Vec<(String, String)> = Vec::new();
    if let Some(decs) = provided_decorators {
        if !decs.is_empty() {
            props.push(("decorators".to_string(), format!("{}", json!(decs))));
        }
    }
    let bases = child_by_field(node, "superclasses");
    let mut bases_node: Option<Node> = bases;
    if bases_node.is_none() {
        for c in node.children(&mut node.walk()) {
            if c.kind() == "arguments" || c.kind() == "argument_list" || c.kind() == "base_list" {
                bases_node = Some(c);
                break;
            }
        }
    }
    let body = child_by_field(node, "body");
    let (kind, role) = if is_protocol_class(bases_node.as_ref(), source) {
        ("trait", Some("protocol"))
    } else if is_typeddict_class(bases_node.as_ref(), source) {
        ("type", Some("typeddict"))
    } else {
        ("class", None)
    };
    props.push(("dialect".to_string(), "python".to_string()));
    if let Some(r) = role {
        props.push(("role".to_string(), r.to_string()));
    }
    let node_lines = (node.start_position().row + 1) as usize;
    let node_end = (node.end_position().row + 1) as usize;
    let source_text = truncated_source_text(&node_text_str(node, source), node_lines, node_end);
    result.nodes.push(ExtractedNode {
        kind: kind.to_string(),
        qualified_name: qn.clone(),
        name: name.clone(),
        source_uri: Some(path.to_string()),
        line_start: node_lines,
        line_end: node_end,
        source_text,
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
    if let Some(bases) = bases_node {
        for base in bases.children(&mut bases.walk()) {
            let base_name = match base.kind() {
                "identifier" | "dotted_name" => node_text_str(&base, source)
                    .split('.')
                    .last()
                    .map(|s| s.to_string()),
                "attribute" => {
                    child_by_field(&base, "attribute").map(|a| node_text_str(&a, source))
                }
                _ => None,
            };
            if let Some(bn) = base_name {
                let base_qn = format!("type::{}", bn);
                result.nodes.push(ExtractedNode {
                    kind: "class".to_string(),
                    qualified_name: base_qn.clone(),
                    name: bn.clone(),
                    source_uri: None,
                    line_start: 0,
                    line_end: 0,
                    source_text: None,
                    properties: vec![
                        ("dialect".to_string(), "python".to_string()),
                        ("role".to_string(), "base_class".to_string()),
                    ],
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
    if let Some(body) = body {
        walk_scope(
            &body,
            file_qn,
            path,
            &qn,
            &child_scope,
            true,
            file_is_test,
            source,
            result,
        );
    }
}

fn handle_function(
    node: &Node,
    file_qn: &str,
    path: &str,
    parent_qn: &str,
    scope: &[String],
    parent_is_class: bool,
    file_is_test: bool,
    source: &[u8],
    result: &mut ExtractionResult,
    provided_decorators: Option<&Vec<String>>,
) {
    let name_node = match child_by_field(node, "name") {
        Some(n) => n,
        None => return,
    };
    let name = match extract_name(&name_node, source) {
        Some(n) => n,
        None => return,
    };
    let is_test = file_is_test || is_test_name(&name);
    let child_scope: Vec<String> = scope.iter().cloned().chain(Some(name.clone())).collect();
    let qn = format!("{}::{}", file_qn, child_scope.join("::"));
    let mut props: Vec<(String, String)> = Vec::new();
    if let Some(decs) = provided_decorators {
        if !decs.is_empty() {
            props.push(("decorators".to_string(), format!("{}", json!(decs))));
        }
    }
    if is_test {
        props.push(("is_test".to_string(), "true".to_string()));
    }
    let body = child_by_field(node, "body");
    let kind = if parent_is_class {
        "method".to_string()
    } else if is_typevar_definition(body.as_ref(), source) {
        props.push(("role".to_string(), "typevar".to_string()));
        "type".to_string()
    } else {
        props.push(("dialect".to_string(), "python".to_string()));
        "function".to_string()
    };
    let node_lines = (node.start_position().row + 1) as usize;
    let node_end = (node.end_position().row + 1) as usize;
    let source_text = truncated_source_text(&node_text_str(node, source), node_lines, node_end);
    result.nodes.push(ExtractedNode {
        kind,
        qualified_name: qn.clone(),
        name: name.clone(),
        source_uri: Some(path.to_string()),
        line_start: node_lines,
        line_end: node_end,
        source_text,
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
    if let Some(body) = body {
        emit_calls(
            &body,
            source,
            &qn,
            result,
            &["function_definition", "class_definition"],
        );
        walk_scope(
            &body,
            file_qn,
            path,
            &qn,
            &child_scope,
            false,
            file_is_test,
            source,
            result,
        );
    }
}

fn extract_python(
    source: &[u8],
    file_path: &str,
    file_qn: &str,
) -> Result<ExtractionResult, PyErr> {
    let mut parser = tree_sitter::Parser::new();
    let lang = tree_sitter_python::LANGUAGE.into();
    parser.set_language(&lang).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to set language: {}", e))
    })?;
    let tree = parser.parse(source, None).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to parse source")
    })?;
    let root = tree.root_node();
    let mut result = ExtractionResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        calls: Vec::new(),
    };
    let file_name = std::path::Path::new(file_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let file_qn_full = if file_qn.is_empty() {
        format!("file::{}", file_name)
    } else {
        file_qn.to_string()
    };
    result.nodes.push(ExtractedNode {
        kind: "file".to_string(),
        qualified_name: file_qn_full.clone(),
        name: file_name.clone(),
        source_uri: Some(file_path.to_string()),
        line_start: 0,
        line_end: source.split(|&b| b == b'\n').count() as usize,
        source_text: Some(String::from_utf8_lossy(source).into_owned()),
        properties: vec![("dialect".to_string(), "python".to_string())],
    });
    emit_imports(&root, source, &file_qn_full, &mut result);
    let file_is_test =
        is_test_file_path(file_path) || is_test_file_path(&format!("/{}", file_path));
    walk_scope(
        &root,
        &file_qn_full,
        file_path,
        &file_qn_full,
        &[],
        false,
        file_is_test,
        source,
        &mut result,
    );
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (source, file_path="", file_qn=""))]
fn extract_python_file(source: &[u8], file_path: &str, file_qn: &str) -> PyResult<PyObject> {
    let result = extract_python(source, file_path, file_qn)?;
    Python::with_gil(|py| {
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
                } else if let Ok(n) = v.parse::<i64>() {
                    props_dict.set_item(k, n)?;
                } else if let Ok(f) = v.parse::<f64>() {
                    props_dict.set_item(k, f)?;
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
        let output = PyDict::new(py);
        output.set_item("nodes", nodes_list)?;
        output.set_item("edges", edges_list)?;
        output.set_item("calls", calls_list)?;
        Ok(output.into_any().unbind().into())
    })
}

#[pyfunction]
fn version() -> PyResult<&'static str> {
    Ok(env!("CARGO_PKG_VERSION"))
}

#[pyfunction]
fn available() -> PyResult<bool> {
    Ok(true)
}

/// Extract multiple Python files in parallel using rayon.
///
/// Accepts a list of dicts with "path" and "source" keys (source as bytes).
/// Returns a single dict with combined "nodes", "edges", and "calls".
///
/// Uses rayon's par_iter + try_reduce for efficient tree-reduction:
/// each file is processed on a rayon thread, and results are merged
/// as workers finish, avoiding one large merge at the end.
#[pyfunction]
#[pyo3(signature = (files))]
fn extract_python_files(py: Python, files: &Bound<'_, PyList>) -> PyResult<PyObject> {
    // Convert Python list of dicts to Vec<(path, source_bytes)>
    let file_data: Vec<(String, Vec<u8>)> = files
        .iter()
        .map(|item| {
            let dict = item.downcast::<PyDict>()?;
            let path = dict.get_item("path")?.unwrap().extract::<String>()?;
            let source = dict.get_item("source")?.unwrap().extract::<Vec<u8>>()?;
            Ok((path, source))
        })
        .collect::<PyResult<_>>()?;

    if file_data.is_empty() {
        let output = PyDict::new(py);
        return Ok(output.into_any().unbind().into());
    }

    // Process files in parallel using rayon
    let results: Result<Vec<ExtractionResult>, PyErr> = file_data
        .par_iter()
        .map(|(path, source)| {
            let file_qn = format!("file::{}", path);
            extract_python(source, path, &file_qn)
        })
        .collect();

    let all_results = results?;

    // Merge all results
    let mut merged = ExtractionResult {
        nodes: Vec::new(),
        edges: Vec::new(),
        calls: Vec::new(),
    };

    for result in all_results {
        merged.nodes.extend(result.nodes);
        merged.edges.extend(result.edges);
        merged.calls.extend(result.calls);
    }

    // Convert merged result to Python dict
    Python::with_gil(|py| {
        let nodes_list = PyList::empty(py);
        for node in &merged.nodes {
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
                } else if let Ok(n) = v.parse::<i64>() {
                    props_dict.set_item(k, n)?;
                } else if let Ok(f) = v.parse::<f64>() {
                    props_dict.set_item(k, f)?;
                } else {
                    props_dict.set_item(k, v)?;
                }
            }
            dict.set_item("properties", props_dict)?;
            nodes_list.append(dict)?;
        }
        let edges_list = PyList::empty(py);
        for edge in &merged.edges {
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
        for call in &merged.calls {
            let dict = PyDict::new(py);
            dict.set_item("caller_qn", &call.caller_qn)?;
            dict.set_item("callee_qn", &call.callee_qn)?;
            dict.set_item("receiver", call.receiver.as_deref().unwrap_or(""))?;
            calls_list.append(dict)?;
        }
        let output = PyDict::new(py);
        output.set_item("nodes", nodes_list)?;
        output.set_item("edges", edges_list)?;
        output.set_item("calls", calls_list)?;
        Ok(output.into_any().unbind().into())
    })
}

pub mod extractors;
use extractors::{cpp, go, java, javascript, rust, typescript};

// ============================================================================
// Data Flow Edge Extraction
// ============================================================================

/// Represents a data flow edge between two named nodes.
#[derive(Debug, Clone)]
struct DataFlowEdge {
    source_id: u64,
    target_id: u64,
    source_kind: String,
    target_kind: String,
    source_name: String,
    target_name: String,
    flow_type: String,
    source_line: String,
}

/// Extract data flow edges from a function/method body.
///
/// Scans the source text for assignment patterns, parameter usage,
/// and return statements. Emits DataFlow edges between variables
/// and between variables and their containing function.
///
/// Returns a list of DataFlowEdge dicts for the Python wrapper.
#[pyfunction]
#[pyo3(signature = (source_text, function_id, params=None))]
fn extract_data_flow(
    py: Python,
    source_text: &str,
    function_id: u64,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let params_list: Vec<String> = match params {
        Some(pl) => {
            let list = pl.downcast::<PyList>().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "params must be a list: {}",
                    e
                ))
            })?;
            list.iter()
                .map(|p| p.extract::<String>().unwrap_or_default())
                .collect()
        }
        None => extract_params(source_text),
    };

    let edges: Vec<DataFlowEdge> = {
        let mut edges = extract_assignments(source_text, function_id, &params_list);
        edges.extend(extract_return_flow(source_text, function_id, &params_list));
        edges
    };

    // Convert to Python list of dicts
    let edges_list = PyList::empty(py);
    for edge in &edges {
        let dict = PyDict::new(py);
        dict.set_item("source_id", edge.source_id)?;
        dict.set_item("target_id", edge.target_id)?;
        dict.set_item("source_kind", &edge.source_kind)?;
        dict.set_item("target_kind", &edge.target_kind)?;
        dict.set_item("source_name", &edge.source_name)?;
        dict.set_item("target_name", &edge.target_name)?;
        dict.set_item("flow_type", &edge.flow_type)?;
        dict.set_item("source_text", &edge.source_line)?;
        edges_list.append(dict)?;
    }

    let output = PyDict::new(py);
    output.set_item("edges", edges_list)?;
    output.set_item("count", edges.len())?;
    Ok(output.into_any().unbind().into())
}

/// Extract assignment edges: `let/var/mut x = expr` or `x = expr`.
fn extract_assignments(
    source_text: &str,
    function_id: u64,
    params: &[String],
) -> Vec<DataFlowEdge> {
    let mut edges = Vec::new();
    let lines: Vec<&str> = source_text.lines().collect();

    for line in lines.iter() {
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with("#") || trimmed.starts_with("/*") {
            continue;
        }

        // Pattern: `let [mut] var: Type = expr` or `let [mut] var = expr`
        // or `var = expr` (reassignment)
        if let Some((var_name, expr)) = parse_assignment(trimmed) {
            if var_name.is_empty() || var_name.starts_with('_') {
                continue;
            }

            let var_qn = format!("var::{}::{}", function_id, var_name);
            let var_id = format_qn_to_id(&var_qn);

            // Emit DataFlow from params if the expression references them
            for param in params {
                if expr.contains(param.as_str()) {
                    let param_qn = format!("param::{}::{}", function_id, param);
                    let param_id = format_qn_to_id(&param_qn);
                    edges.push(DataFlowEdge {
                        source_id: param_id,
                        target_id: var_id,
                        source_kind: "param".to_string(),
                        target_kind: "variable".to_string(),
                        source_name: param.clone(),
                        target_name: var_name.clone(),
                        flow_type: "param_flow".to_string(),
                        source_line: trimmed.to_string(),
                    });
                }
            }

            // Emit DataFlow from the containing function to the variable
            edges.push(DataFlowEdge {
                source_id: function_id,
                target_id: var_id,
                source_kind: "function".to_string(),
                target_kind: "variable".to_string(),
                source_name: format!("func::{}", function_id),
                target_name: var_name.clone(),
                flow_type: "assignment".to_string(),
                source_line: trimmed.to_string(),
            });
        }
    }

    edges
}

/// Parse an assignment from a line of code.
/// Returns (variable_name, expression) or None.
/// Handles: let/var/mut x = expr, x = expr, self.field = expr
fn parse_assignment(line: &str) -> Option<(String, String)> {
    let mut s = line.trim().to_string();

    // Strip leading `self.` for Python field assignments
    if s.starts_with("self.") {
        s = s[5..].to_string();
    }

    // Strip `let ` prefix
    if s.starts_with("let ") {
        s = s[4..].to_string();
    }

    // Strip `mut ` prefix (Rust) — after `let` so `let mut x` works
    if s.starts_with("mut ") {
        s = s[4..].to_string();
    }

    // Find the equals sign
    let eq_pos = s.find('=')?;
    let before_eq = s[..eq_pos].trim();
    let after_eq = s[eq_pos + 1..].trim();

    // Handle `var: Type` — strip everything after ':'
    let var_name = if let Some(colon_pos) = before_eq.find(':') {
        before_eq[..colon_pos].trim().to_string()
    } else {
        before_eq.to_string()
    };

    if var_name.is_empty() || var_name.starts_with('_') {
        return None;
    }

    Some((var_name, after_eq.to_string()))
}

/// Extract return value flow: `return expr` where expr references a known node.
fn extract_return_flow(
    source_text: &str,
    function_id: u64,
    params: &[String],
) -> Vec<DataFlowEdge> {
    let mut edges = Vec::new();
    let lines: Vec<&str> = source_text.lines().collect();

    for line in lines.iter() {
        let trimmed = line.trim();

        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with("#") {
            continue;
        }

        // Match: `return expr`, `return expr,` (tuple), `return Some(expr)`
        if let Some(return_expr) = parse_return(trimmed) {
            if return_expr.is_empty() {
                continue;
            }

            let return_qn = format!("return::{}", function_id);
            let return_id = format_qn_to_id(&return_qn);

            // Check if return value references a parameter
            for param in params {
                if return_expr.contains(param.as_str()) {
                    let param_qn = format!("param::{}::{}", function_id, param);
                    let param_id = format_qn_to_id(&param_qn);
                    edges.push(DataFlowEdge {
                        source_id: param_id,
                        target_id: return_id,
                        source_kind: "param".to_string(),
                        target_kind: "return_value".to_string(),
                        source_name: param.clone(),
                        target_name: "return_value".to_string(),
                        flow_type: "return_flow".to_string(),
                        source_line: trimmed.to_string(),
                    });
                }
            }

            // Emit DataFlow from function to its return value
            edges.push(DataFlowEdge {
                source_id: function_id,
                target_id: return_id,
                source_kind: "function".to_string(),
                target_kind: "return_value".to_string(),
                source_name: format!("func::{}", function_id),
                target_name: "return_value".to_string(),
                flow_type: "return_flow".to_string(),
                source_line: trimmed.to_string(),
            });
        }
    }

    edges
}

/// Parse a return statement, extracting the expression after `return`.
fn parse_return(line: &str) -> Option<String> {
    let trimmed = line.trim();

    // Match `return expr`, `return expr;`, `return (expr)`
    if trimmed.starts_with("return ") || trimmed.starts_with("return\t") {
        let after = &trimmed[7..]; // skip "return "
        let expr = after.trim_end_matches(';').trim();
        Some(expr.to_string())
    } else if trimmed == "return" {
        Some(String::new())
    } else {
        None
    }
}

/// Convert a qualified name to a deterministic ID for tracking.
fn format_qn_to_id(qn: &str) -> u64 {
    // Simple hash-based ID for tracking variable nodes
    let mut hash: u64 = 0;
    for byte in qn.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash
}

/// Build a parameter list from a function's source text.
/// Extracts parameter names from function signatures.
pub fn extract_params(source_text: &str) -> Vec<String> {
    let mut params = Vec::new();

    // Find the function signature (first line with `fn` or `def`)
    for line in source_text.lines() {
        let trimmed = line.trim();

        // Rust: `fn name(args) {` or `fn name(&self, args) {`
        if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") {
            if let Some(args) = extract_rust_params(trimmed) {
                params.extend(args);
            }
            break;
        }

        // Python: `def name(args):`
        if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
            if let Some(args) = extract_python_params(trimmed) {
                params.extend(args);
            }
            break;
        }

        // TypeScript/JS: `function name(args)` or class methods
        if trimmed.starts_with("function ")
            || trimmed.starts_with("public ")
            || trimmed.starts_with("private ")
            || trimmed.starts_with("protected ")
        {
            if let Some(args) = extract_ts_params(trimmed) {
                params.extend(args);
            }
            break;
        }
    }

    params
}

fn extract_rust_params(line: &str) -> Option<Vec<String>> {
    let paren_start = line.find('(')?;
    let paren_end = line.find(')')?;
    let args = &line[paren_start + 1..paren_end];

    let mut params = Vec::new();
    for arg in args.split(',') {
        let arg = arg.trim();
        if arg.is_empty() || arg == "self" || arg == "&self" || arg == "&mut self" {
            continue;
        }
        let name = arg
            .split([':', '='])
            .next()
            .unwrap_or(arg)
            .trim()
            .to_string();
        let name = name
            .strip_prefix("&mut ")
            .or_else(|| name.strip_prefix("&"))
            .unwrap_or(&name)
            .trim()
            .to_string();
        if !name.is_empty() && !name.starts_with('_') {
            params.push(name);
        }
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn extract_python_params(line: &str) -> Option<Vec<String>> {
    let paren_start = line.find('(')?;
    let paren_end = line.find(')')?;
    let args = &line[paren_start + 1..paren_end];

    let mut params = Vec::new();
    for arg in args.split(',') {
        let arg = arg.trim();
        if arg.is_empty() || arg == "self" || arg == "cls" {
            continue;
        }
        let name = arg
            .split([':', '='])
            .next()
            .unwrap_or(arg)
            .trim()
            .to_string();
        let name = name.trim().to_string();
        if !name.is_empty() && !name.starts_with('_') {
            params.push(name);
        }
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

fn extract_ts_params(line: &str) -> Option<Vec<String>> {
    let paren_start = line.find('(')?;
    let paren_end = line.find(')')?;
    let args = &line[paren_start + 1..paren_end];

    let mut params = Vec::new();
    for arg in args.split(',') {
        let arg = arg.trim();
        if arg.is_empty() {
            continue;
        }
        let name = arg
            .split([':', '='])
            .next()
            .unwrap_or(arg)
            .trim()
            .to_string();
        let name = name
            .strip_prefix("public ")
            .or_else(|| name.strip_prefix("private "))
            .or_else(|| name.strip_prefix("protected "))
            .unwrap_or(&name)
            .trim()
            .to_string();
        let name = name.strip_suffix('?').unwrap_or(&name).trim().to_string();
        if name == "this" || name.is_empty() || name.starts_with('_') {
            continue;
        }
        params.push(name);
    }

    if params.is_empty() {
        None
    } else {
        Some(params)
    }
}

// Community detection — re-exported from analysis::core
pub use analysis::core::{
    community_detection_infomap, community_detection_leiden, community_detection_louvain,
    CommunityOptions,
};

#[pymodule]
fn graphician_native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_python_file, m)?)?;
    m.add_function(wrap_pyfunction!(extract_python_files, m)?)?;
    m.add_function(wrap_pyfunction!(extract_data_flow, m)?)?;
    m.add_function(wrap_pyfunction!(rust::extract_rust_file, m)?)?;
    m.add_function(wrap_pyfunction!(typescript::extract_typescript_file, m)?)?;
    m.add_function(wrap_pyfunction!(javascript::extract_javascript_file, m)?)?;
    m.add_function(wrap_pyfunction!(java::extract_java_file, m)?)?;
    m.add_function(wrap_pyfunction!(cpp::extract_cpp_file, m)?)?;
    m.add_function(wrap_pyfunction!(go::extract_go_file, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available, m)?)?;
    m.add_function(wrap_pyfunction!(community_detection_louvain, m)?)?;
    m.add_function(wrap_pyfunction!(community_detection_leiden, m)?)?;
    m.add_function(wrap_pyfunction!(community_detection_infomap, m)?)?;
    m.add_function(wrap_pyfunction!(analysis::dedup::dedup_candidate_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(analysis::search::fuzzy_score_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(
        analysis::resolution::plan_type_resolution,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        analysis::resolution::plan_call_resolution,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(persistence::save_graph_sqlite, m)?)?;
    m.add_function(wrap_pyfunction!(
        persistence::save_graph_incremental_sqlite,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(persistence::load_graph_sqlite, m)?)?;
    m.add_class::<CommunityOptions>()?;
    m.add_class::<graph::NativeGraph>()?;
    m.add(
        "__doc__",
        "High-performance multi-language code extraction using tree-sitter Rust bindings",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let source = b"def foo():\n    pass\n";
        let result = extract_python(source, "test.py", "file::test").unwrap();
        assert!(result.nodes.iter().any(|n| n.kind == "function"));
    }

    #[test]
    fn test_class() {
        let source = b"class Foo:\n    def bar(self):\n        pass\n";
        let result = extract_python(source, "test.py", "file::test").unwrap();
        assert!(result.nodes.iter().any(|n| n.kind == "class"));
        assert!(result.nodes.iter().any(|n| n.kind == "method"));
    }

    #[test]
    fn test_imports() {
        let source = b"import os\nfrom typing import List\n";
        let result = extract_python(source, "test.py", "file::test").unwrap();
        let imports: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.kind == "imports")
            .collect();
        assert!(!imports.is_empty());
    }

    // ===== Test detection tests =====

    #[test]
    fn detects_python_test_files() {
        assert!(is_test_file_path("tests/test_auth.py"));
        assert!(is_test_file_path("project/tests/foo.py"));
        assert!(is_test_file_path("src/test_auth.py"));
        assert!(!is_test_file_path("src/auth.py"));
    }

    #[test]
    fn detects_js_test_files() {
        assert!(is_test_file_path("src/auth.test.ts"));
        assert!(is_test_file_path("src/auth.spec.js"));
        assert!(is_test_file_path("__tests__/auth.js"));
    }

    #[test]
    fn detects_common_xunit_and_spec_files() {
        assert!(is_test_file_path("src/FooTest.java"));
        assert!(is_test_file_path("src/FooTests.kt"));
        assert!(is_test_file_path("src/FooSpec.swift"));
        assert!(is_test_file_path("src/FooSuite.scala"));
        assert!(is_test_file_path("src/FooTest.cs"));
        assert!(is_test_file_path("src/FooTest.php"));
        assert!(!is_test_file_path("src/Contest.java"));
        assert!(!is_test_file_path("src/Latest.kt"));
    }

    #[test]
    fn detects_test_names() {
        assert!(is_test_name("test_login"));
        assert!(is_test_name("TestLogin"));
        assert!(is_test_name("login_test"));
        assert!(is_test_name("login_spec"));
        assert!(is_test_name("shouldRejectExpiredTokens"));
        assert!(is_test_name("itReturnsNullForMissingUser"));
        assert!(!is_test_name("login"));
        assert!(!is_test_name("Test")); // bare Test is often a type
        assert!(!is_test_name("Testimony"));
        assert!(!is_test_name("should")); // bare verb
    }

    // ===== Suppression list tests =====

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
