//! High-performance Python code extraction using tree-sitter Rust bindings.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::json;
use tree_sitter::Node;

/// Safe text extraction from a tree-sitter node via byte-slicing.
fn node_text<'a>(node: &Node<'a>, source: &'a [u8]) -> &'a [u8] {
    if node.byte_range().is_empty() {
        &[]
    } else {
        &source[node.byte_range()]
    }
}

fn node_text_str<'a>(node: &Node<'a>, source: &'a [u8]) -> String {
    String::from_utf8_lossy(node_text(node, source)).into_owned()
}

fn extract_name(node: &Node, source: &[u8]) -> Option<String> {
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

fn child_by_field<'a>(node: &'a Node<'a>, field: &str) -> Option<Node<'a>> {
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
                        if matches!(name.as_str(), "TypeVar" | "ParamSpec" | "TypeVarTuple" | "GenericAlias") {
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
fn extract_decorator_name(dec: &Node, source: &[u8]) -> String {
    for c in dec.children(&mut dec.walk()) {
        if c.kind() == "@" {
            continue;
        }
        // Try to get the full dotted name
        match c.kind() {
            "identifier" => return node_text_str(&c, source).trim().to_string(),
            "attribute" => {
                // Build dotted name: object.attribute
                let parts: Vec<String> = c.children(&mut c.walk())
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
                            let parts: Vec<String> = func.children(&mut func.walk())
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

fn extract_decorators(node: &Node, source: &[u8]) -> Vec<String> {
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
struct ExtractedNode {
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
struct ExtractedEdge {
    src_qn: String,
    dst_qn: String,
    kind: String,
    conf_class: String,
    confidence: f64,
    properties: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct CallPlaceholder {
    caller_qn: String,
    callee_qn: String,
    receiver: Option<String>,
}

#[derive(Debug)]
struct ExtractionResult {
    nodes: Vec<ExtractedNode>,
    edges: Vec<ExtractedEdge>,
    calls: Vec<CallPlaceholder>,
}

fn should_suppress(name: &str) -> bool {
    matches!(
        name,
        "print" | "len" | "range" | "super" | "property" | "staticmethod"
        | "classmethod" | "isinstance" | "issubclass" | "type" | "getattr"
        | "setattr" | "hasattr" | "delattr" | "repr" | "str" | "int"
        | "float" | "list" | "dict" | "set" | "tuple" | "bool" | "bytes"
        | "bytearray" | "frozenset" | "enumerate" | "zip" | "map" | "filter"
        | "sorted" | "reversed" | "any" | "all" | "abs" | "round" | "min"
        | "max" | "sum" | "pow" | "divmod" | "chr" | "ord" | "hex" | "oct"
        | "bin" | "format" | "hash" | "id" | "input" | "open" | "compile"
        | "eval" | "exec" | "dir" | "vars" | "help" | "callable"
        | "memoryview" | "slice" | "complex" | "object" | "next" | "iter"
        | "breakpoint"
    )
}

fn emit_imports(
    node: &Node,
    source: &[u8],
    file_qn: &str,
    result: &mut ExtractionResult,
) {
    for child in node.children(&mut node.walk()) {
        match child.kind() {
            "import_statement" => {
                for c in child.children(&mut child.walk()) {
                    if c.kind() == "dotted_name" {
                        let path_text = node_text_str(&c, source);
                        let mod_qn = format!("module::{}", path_text);
                        result.nodes.push(ExtractedNode {
                            kind: "module".to_string(),
                            qualified_name: mod_qn.clone(),
                            name: path_text.clone(),
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
            "import_from_statement" => {
                if let Some(module_name) = child_by_field(&child, "module_name") {
                    let path_text = node_text_str(&module_name, source);
                    let mod_qn = format!("module::{}", path_text);
                    result.nodes.push(ExtractedNode {
                        kind: "module".to_string(),
                        qualified_name: mod_qn.clone(),
                        name: path_text.clone(),
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
            _ => {}
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
                    if !n.starts_with('_') && !should_suppress(&n) {
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
                        "class_definition" => handle_class(&def, file_qn, path, parent_qn, scope, file_is_test, source, result, Some(&decorators)),
                        "function_definition" => handle_function(&def, file_qn, path, parent_qn, scope, parent_is_class, file_is_test, source, result, Some(&decorators)),
                        _ => {}
                    }
                }
            }
            "class_definition" => handle_class(&child, file_qn, path, parent_qn, scope, file_is_test, source, result, None),
            "function_definition" => handle_function(&child, file_qn, path, parent_qn, scope, parent_is_class, file_is_test, source, result, None),
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
    let source_text = if node.byte_range().is_empty() { None } else { Some(node_text_str(node, source)) };
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
                "identifier" | "dotted_name" => node_text_str(&base, source).split('.').last().map(|s| s.to_string()),
                "attribute" => child_by_field(&base, "attribute").map(|a| node_text_str(&a, source)),
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
                    properties: vec![("dialect".to_string(), "python".to_string()), ("role".to_string(), "base_class".to_string())],
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
        walk_scope(&body, file_qn, path, &qn, &child_scope, true, file_is_test, source, result);
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
    let is_test = file_is_test || name.starts_with("test_") || name.ends_with("_test");
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
    let source_text = if node.byte_range().is_empty() { None } else { Some(node_text_str(node, source)) };
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
        emit_calls(&body, source, &qn, result, &["function_definition", "class_definition"]);
        walk_scope(&body, file_qn, path, &qn, &child_scope, false, file_is_test, source, result);
    }
}

fn extract_python(source: &[u8], file_path: &str, file_qn: &str) -> Result<ExtractionResult, PyErr> {
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
    let file_is_test = file_name.starts_with("test_") || file_name.ends_with("_test");
    walk_scope(&root, &file_qn_full, file_path, &file_qn_full, &[], false, file_is_test, source, &mut result);
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

#[pymodule]
fn graphician_extract(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_python_file, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(available, m)?)?;
    m.add("__doc__", "High-performance Python code extraction using tree-sitter Rust bindings")?;
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
        let imports: Vec<_> = result.edges.iter().filter(|e| e.kind == "imports").collect();
        assert!(!imports.is_empty());
    }
}
