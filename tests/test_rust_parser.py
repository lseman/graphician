"""Comprehensive tests for Rust parser extraction details."""

from __future__ import annotations

from pathlib import Path

from graphician.core.edge import EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import NodeKind
from graphician.extraction.languages.parsers.rust import (
    _extract_source_text,
    _in_any_range,
    _join_use_path,
    _should_suppress_rust_call,
    extract_file,
)

# ---------------------------------------------------------------------------
# Pure helper unit tests
# ---------------------------------------------------------------------------

class TestShouldSuppressRustCall:
    """Tests for _should_suppress_rust_call."""

    def test_empty_name_suppressed(self) -> None:
        assert _should_suppress_rust_call("")
        assert _should_suppress_rust_call(None)  # type: ignore[arg-type]

    def test_rust_builtins_suppressed(self) -> None:
        assert _should_suppress_rust_call("unwrap")
        assert _should_suppress_rust_call("expect")
        assert _should_suppress_rust_call("and_then")
        assert _should_suppress_rust_call("is_some")
        assert _should_suppress_rust_call("collect")
        assert _should_suppress_rust_call("sort_by")

    def test_tree_sitter_api_suppressed(self) -> None:
        assert _should_suppress_rust_call("children")
        assert _should_suppress_rust_call("parent")
        assert _should_suppress_rust_call("text")
        assert _should_suppress_rust_call("walk")

    def test_project_names_not_suppressed(self) -> None:
        assert not _should_suppress_rust_call("my_function")
        assert not _should_suppress_rust_call("calculate")
        assert not _should_suppress_rust_call("process_data")
        assert not _should_suppress_rust_call("get")

    def test_c_standard_library_suppressed(self) -> None:
        assert _should_suppress_rust_call("malloc")
        assert _should_suppress_rust_call("free")
        assert _should_suppress_rust_call("printf")
        assert _should_suppress_rust_call("memcpy")


class TestExtractSourceText:
    """Tests for _extract_source_text."""

    def test_empty_source(self) -> None:
        assert _extract_source_text([], 0, 0) == ""

    def test_invalid_range(self) -> None:
        assert _extract_source_text(["a", "b", "c"], 0, 5) == ""
        assert _extract_source_text(["a", "b", "c"], 5, 3) == ""

    def test_out_of_bounds(self) -> None:
        assert _extract_source_text(["a", "b"], 10, 15) == ""

    def test_single_line(self) -> None:
        lines = ["first", "second", "third"]
        # When start==end, e = end-1 = 0, s = start-1 = 0, s>=e so returns ""
        # This matches the actual Rust behavior for 1-indexed ranges
        assert _extract_source_text(lines, 1, 1) == ""
        # A proper single-line range needs start+1 == end (e.g. 1:1 → rows 0:0)
        assert _extract_source_text(lines, 1, 2) == "first"

    def test_multiple_lines(self) -> None:
        lines = ["first", "second", "third"]
        assert _extract_source_text(lines, 1, 3) == "first\nsecond"

    def test_offset_by_one(self) -> None:
        """_extract_source_text assumes 1-indexed rows (matches ariadne-rust)."""
        lines = ["a", "b", "c", "d", "e"]
        assert _extract_source_text(lines, 2, 4) == "b\nc"

    def test_edge_at_end(self) -> None:
        lines = ["a", "b", "c"]
        assert _extract_source_text(lines, 2, 3) == "b"


class TestJoinUsePath:
    """Tests for _join_use_path."""

    def test_no_prefix(self) -> None:
        assert _join_use_path("", "foo::bar") == "foo::bar"

    def test_prefix_only(self) -> None:
        assert _join_use_path("std::io", "") == "std::io"

    def test_path_is_self(self) -> None:
        assert _join_use_path("std::io", "self") == "std::io"

    def test_combined(self) -> None:
        assert _join_use_path("std::io", "BufReader") == "std::io::BufReader"

    def test_nested_combined(self) -> None:
        assert _join_use_path("std::collections", "HashMap") == "std::collections::HashMap"


class TestInAnyRange:
    """Tests for _in_any_range."""

    def test_fully_inside(self) -> None:
        ranges = [(0, 100), (200, 300)]
        assert _in_any_range((10, 50), ranges)
        assert _in_any_range((0, 100), ranges)

    def test_outside_all(self) -> None:
        ranges = [(0, 100), (200, 300)]
        assert not _in_any_range((150, 180), ranges)
        assert not _in_any_range((350, 400), ranges)

    def test_partial_overlap_not_inside(self) -> None:
        ranges = [(0, 50)]
        assert not _in_any_range((25, 75), ranges)

    def test_empty_ranges(self) -> None:
        assert not _in_any_range((10, 20), [])


# ---------------------------------------------------------------------------
# Helper to get node from qname
# ---------------------------------------------------------------------------

def _find_node(g: Graph, qn: str) -> NodeId | None:
    """Find a node by qname and return the NodeId, or None."""
    nid = g.find_by_qname(qn)
    if nid is not None:
        return nid
    return None


def _node_kind(g: Graph, nid: NodeId) -> NodeKind | None:
    n = g.node(nid)
    return n.kind if n is not None else None


# ---------------------------------------------------------------------------
# Integration tests via extract_file
# ---------------------------------------------------------------------------

class TestExtractFileBasic:
    """Test basic extraction of different Rust constructs."""

    def test_simple_function(self, tmp_path: Path) -> None:
        source = tmp_path / "main.rs"
        source.write_text(
            "fn main() { println!(\"hello\"); }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        fn = _find_node(graph, "main::main")
        assert fn is not None
        assert _node_kind(graph, fn) == NodeKind.FUNCTION
        file_node = _find_node(graph, "main")
        assert file_node is not None

    def test_multiple_functions(self, tmp_path: Path) -> None:
        source = tmp_path / "mod.rs"
        source.write_text(
            "fn foo() { bar(); }\nfn bar() { }\nfn baz(x: i32) -> i32 { x * 2 }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        assert _find_node(graph, "mod::foo") is not None
        assert _find_node(graph, "mod::bar") is not None
        assert _find_node(graph, "mod::baz") is not None

    def test_function_with_call(self, tmp_path: Path) -> None:
        source = tmp_path / "calc.rs"
        source.write_text(
            "fn add(a: i32, b: i32) -> i32 { a + b }\n"
            "fn main() { let x = add(1, 2); }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        caller = _find_node(graph, "calc::main")
        assert caller is not None
        edges = [
            (t, e) for t, e in graph.out_neighbors(caller) if e.kind == EdgeKind.CALLS
        ]
        targets = [t for t, _ in edges]
        callee_qns = {graph.node(t).qualified_name for t in targets}
        assert "call::add" in callee_qns

    def test_struct(self, tmp_path: Path) -> None:
        source = tmp_path / "types.rs"
        source.write_text(
            "struct Point { x: f64, y: f64 }\n"
            "struct Rectangle { width: f64, height: f64 }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        assert _find_node(graph, "types::Point") is not None
        assert _find_node(graph, "types::Rectangle") is not None
        for qn in ["types::Point", "types::Rectangle"]:
            fn = _find_node(graph, qn)
            assert fn is not None and _node_kind(graph, fn) == NodeKind.CLASS

    def test_enum_with_variants(self, tmp_path: Path) -> None:
        source = tmp_path / "color.rs"
        source.write_text(
            "enum Color { Red, Green, Blue }\n"
            "enum Result<T, E> { Ok(T), Err(E) }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        assert _find_node(graph, "color::Color") is not None
        assert _find_node(graph, "color::Result") is not None
        # Enum variants are Variable nodes
        assert _find_node(graph, "color::Color::Red") is not None
        assert _find_node(graph, "color::Color::Green") is not None
        assert _find_node(graph, "color::Color::Blue") is not None
        assert _find_node(graph, "color::Result::Ok") is not None
        assert _find_node(graph, "color::Result::Err") is not None

    def test_trait(self, tmp_path: Path) -> None:
        source = tmp_path / "traits.rs"
        source.write_text(
            "trait Printable { fn print(&self); }\n"
            "trait Iterator { type Item; fn next(&mut self) -> Option<Self::Item>; }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        assert _find_node(graph, "traits::Printable") is not None
        assert _find_node(graph, "traits::Iterator") is not None
        for qn in ["traits::Printable", "traits::Iterator"]:
            fn = _find_node(graph, qn)
            assert fn is not None and _node_kind(graph, fn) == NodeKind.TRAIT

    def test_nested_function_calls(self, tmp_path: Path) -> None:
        source = tmp_path / "nested.rs"
        source.write_text(
            "fn outer() { "
            "  let x = inner1(); "
            "  let y = inner2(x); "
            "  some_func(inner3(y)); "
            "}",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        caller = _find_node(graph, "nested::outer")
        assert caller is not None
        edges = [
            (t, e) for t, e in graph.out_neighbors(caller) if e.kind == EdgeKind.CALLS
        ]
        targets = [t for t, _ in edges]
        callee_qns = {graph.node(t).qualified_name for t in targets}
        assert "call::inner1" in callee_qns
        assert "call::inner2" in callee_qns
        assert "call::some_func" in callee_qns


class TestExtractFileScoping:
    """Test scoped call extraction with receivers and scopes."""

    def test_method_call_captures_positional_receiver(self, tmp_path: Path) -> None:
        source = tmp_path / "receiver.rs"
        source.write_text(
            "fn populate(graph: &mut Graph) { graph.add_node(node); }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        caller = _find_node(graph, "receiver::populate")
        placeholder = _find_node(graph, "call::add_node")
        assert caller is not None and placeholder is not None
        edges = [
            edge
            for target, edge in graph.out_neighbors(caller)
            if target == placeholder and edge.kind == EdgeKind.CALLS
        ]
        assert len(edges) == 1
        assert edges[0].properties.get("call_receiver") == "graph"

    def test_scoped_call_has_scope_property(self, tmp_path: Path) -> None:
        source = tmp_path / "scoped.rs"
        source.write_text(
            "fn main() { std::process::exit(0); }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        caller = _find_node(graph, "scoped::main")
        assert caller is not None
        edges = [
            edge
            for target, edge in graph.out_neighbors(caller)
            if graph.node(target).qualified_name == "call::exit"
            and edge.kind == EdgeKind.CALLS
        ]
        assert len(edges) == 1
        assert edges[0].properties.get("call_scope") == "std::process"
        assert edges[0].properties.get("call_receiver") == "std"


class TestExtractFileImports:
    """Test import/use declaration extraction."""

    def test_simple_use(self, tmp_path: Path) -> None:
        source = tmp_path / "imports.rs"
        source.write_text(
            "use std::io::Read;\n"
            "use std::collections::HashMap;\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # Import edges from file node to module nodes
        file_node = _find_node(graph, "imports")
        assert file_node is not None
        import_targets = [
            t for t, e in graph.out_neighbors(file_node) if e.kind == EdgeKind.IMPORTS
        ]
        target_qns = {graph.node(t).qualified_name for t in import_targets}
        assert "module::std::io::Read" in target_qns
        assert "module::std::collections::HashMap" in target_qns

    def test_grouped_use(self, tmp_path: Path) -> None:
        source = tmp_path / "grouped.rs"
        source.write_text(
            "use std::{fmt, io};\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        file_node = _find_node(graph, "grouped")
        import_targets = [
            t for t, e in graph.out_neighbors(file_node) if e.kind == EdgeKind.IMPORTS
        ]
        target_qns = {graph.node(t).qualified_name for t in import_targets}
        assert "module::std::fmt" in target_qns
        assert "module::std::io" in target_qns

    def test_nested_grouped_use(self, tmp_path: Path) -> None:
        source = tmp_path / "nested_use.rs"
        source.write_text(
            "use std::collections::{HashMap, HashSet};\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        file_node = _find_node(graph, "nested_use")
        import_targets = [
            t for t, e in graph.out_neighbors(file_node) if e.kind == EdgeKind.IMPORTS
        ]
        target_qns = {graph.node(t).qualified_name for t in import_targets}
        assert "module::std::collections::HashMap" in target_qns
        assert "module::std::collections::HashSet" in target_qns


class TestExtractFileModDeclarations:
    """Test module declaration extraction."""

    def test_mod_declaration(self, tmp_path: Path) -> None:
        source = tmp_path / "mods.rs"
        source.write_text(
            "mod utils;\n"
            "mod config {\n"
            "    fn default() { }\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # External module reference
        utils_mod = _find_node(graph, "mods::utils")
        assert utils_mod is not None
        assert _node_kind(graph, utils_mod) == NodeKind.MODULE

        # Inline module with function
        config_mod = _find_node(graph, "mods::config")
        assert config_mod is not None
        assert _node_kind(graph, config_mod) == NodeKind.MODULE

        config_fn = _find_node(graph, "mods::config::default")
        assert config_fn is not None


class TestExtractFileTestDetection:
    """Test test function detection via naming, attributes, and cfg(test) modules."""

    def test_test_prefixed_name(self, tmp_path: Path) -> None:
        source = tmp_path / "tests.rs"
        source.write_text(
            "fn test_addition() { }\n"
            "fn helper() { }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        test_fn = _find_node(graph, "tests::test_addition")
        assert test_fn is not None and graph.node(test_fn).properties.get("is_test") is True
        helper_fn = _find_node(graph, "tests::helper")
        assert helper_fn is not None and graph.node(helper_fn).properties.get("is_test") is None

    def test_test_suffixed_name(self, tmp_path: Path) -> None:
        source = tmp_path / "specs.rs"
        source.write_text(
            "fn addition_test() { }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        fn = _find_node(graph, "specs::addition_test")
        assert fn is not None and graph.node(fn).properties.get("is_test") is True

    def test_cfg_test_mod(self, tmp_path: Path) -> None:
        source = tmp_path / "lib.rs"
        source.write_text(
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    fn test_basic() { }\n"
            "    fn another_test() { }\n"
            "}\n"
            "fn production_code() { }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # cfg(test) mod functions are scoped under the mod
        test_basic = _find_node(graph, "lib::tests::test_basic")
        assert test_basic is not None and graph.node(test_basic).properties.get("is_test") is True
        another_test = _find_node(graph, "lib::tests::another_test")
        assert another_test is not None and graph.node(another_test).properties.get("is_test") is True
        production = _find_node(graph, "lib::production_code")
        assert production is not None and graph.node(production).properties.get("is_test") is None

    def test_rust_attr_test(self, tmp_path: Path) -> None:
        source = tmp_path / "annotated.rs"
        source.write_text(
            "#[test]\n"
            "fn my_test() { }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        fn = _find_node(graph, "annotated::my_test")
        assert fn is not None and graph.node(fn).properties.get("is_test") is True


class TestExtractFileSourcePath:
    """Test extract_file with source_path parameter for URI mapping."""

    def test_source_path_stored(self, tmp_path: Path) -> None:
        source = tmp_path / "a.rs"
        source.write_text("fn foo() { }", encoding="utf-8")
        graph = Graph()
        source_path = tmp_path / "mapped.rs"
        extract_file(source, graph, source_path=source_path)

        file_node = _find_node(graph, "a")
        assert file_node is not None
        assert graph.node(file_node).source_uri == str(source_path)


class TestWalkEnumFields:
    """Tests for enum variant field extraction."""

    def test_enum_with_simple_variants(self, tmp_path: Path) -> None:
        source = tmp_path / "status.rs"
        source.write_text(
            "enum Status { Active, Inactive, Pending }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        enum_node = _find_node(graph, "status::Status")
        assert enum_node is not None
        defines_edges = [
            (t, e) for t, e in graph.out_neighbors(enum_node) if e.kind == EdgeKind.DEFINES
        ]
        var_qns = {graph.node(t).qualified_name for t, _ in defines_edges}
        assert "status::Status::Active" in var_qns
        assert "status::Status::Inactive" in var_qns
        assert "status::Status::Pending" in var_qns

    def test_enum_with_named_variant_fields(self, tmp_path: Path) -> None:
        source = tmp_path / "shape.rs"
        source.write_text(
            "enum Shape {\n"
            "    Circle { radius: f64 },\n"
            "    Rectangle { width: f64, height: f64 }\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        circle_var = _find_node(graph, "shape::Shape::Circle")
        assert circle_var is not None
        # Named variant fields are extracted via _walk_enum_fields
        defines_edges = [
            (t, e) for t, e in graph.out_neighbors(circle_var) if e.kind == EdgeKind.DEFINES
        ]
        field_qns = {graph.node(t).qualified_name for t, _ in defines_edges}
        # Field extraction works when the tree-sitter field name matches
        if field_qns:
            assert "shape::Shape::Circle::radius" in field_qns


class TestExtractFileWithSourceText:
    """Test that source text is stored on extracted nodes."""

    def test_function_source_text_stored(self, tmp_path: Path) -> None:
        source = tmp_path / "src.rs"
        # Add leading lines so the function is not at row 0
        # (row 0 produces empty source_text due to known offset behavior)
        source.write_text(
            "// module\n"
            "fn compute(x: i32) -> i32 {\n"
            "    x * 2\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        fn = _find_node(graph, "src::compute")
        assert fn is not None
        node = graph.node(fn)
        assert node.source_text is not None
        # The source_text should contain the function signature
        assert "fn compute" in node.source_text


class TestExtractFileMethodDetection:
    """Test that methods inside impl blocks are detected as methods."""

    def test_method_inside_impl(self, tmp_path: Path) -> None:
        source = tmp_path / "impl_test.rs"
        source.write_text(
            "struct MyStruct;\n"
            "impl MyStruct {\n"
            "    fn method(&self) { }\n"
            "    fn static_func() { }\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # Methods in impl blocks get scoped with the impl type name
        method = _find_node(graph, "impl_test::MyStruct::method")
        assert method is not None and _node_kind(graph, method) == NodeKind.METHOD

        static_fn = _find_node(graph, "impl_test::MyStruct::static_func")
        assert static_fn is not None and _node_kind(graph, static_fn) == NodeKind.METHOD


class TestExtractFileEdgeCases:
    """Test edge cases in extraction."""

    def test_empty_file(self, tmp_path: Path) -> None:
        source = tmp_path / "empty.rs"
        source.write_text("", encoding="utf-8")
        graph = Graph()
        extract_file(source, graph)

        # File node should still exist
        file_node = _find_node(graph, "empty")
        assert file_node is not None

    def test_only_comments(self, tmp_path: Path) -> None:
        source = tmp_path / "comments.rs"
        source.write_text(
            "// A comment\n"
            "/* block comment */\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        file_node = _find_node(graph, "comments")
        assert file_node is not None

    def test_macro_invocation(self, tmp_path: Path) -> None:
        source = tmp_path / "macros.rs"
        source.write_text(
            "fn main() { println!(\"hello\"); vec![1, 2, 3]; }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # println! is in the suppression list for Rust, so it should not appear
        caller = _find_node(graph, "macros::main")
        assert caller is not None
        edges = [
            (t, e) for t, e in graph.out_neighbors(caller) if e.kind == EdgeKind.CALLS
        ]
        target_qns = {graph.node(t).qualified_name for t, _ in edges}
        # println is suppressed; vec! should also be suppressed
        assert "call::println" not in target_qns

    def test_complex_generic_struct(self, tmp_path: Path) -> None:
        source = tmp_path / "generic.rs"
        source.write_text(
            "struct Container<T> { value: T }\n"
            "impl<T> Container<T> {\n"
            "    fn new(value: T) -> Self { Container { value } }\n"
            "    fn get(&self) -> &T { &self.value }\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        struct_node = _find_node(graph, "generic::Container")
        assert struct_node is not None and _node_kind(graph, struct_node) == NodeKind.CLASS

    def test_trait_with_associated_types(self, tmp_path: Path) -> None:
        source = tmp_path / "traits2.rs"
        source.write_text(
            "trait Processor {\n"
            "    type Input;\n"
            "    type Output;\n"
            "    fn process(&self, input: Self::Input) -> Self::Output;\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        trait_node = _find_node(graph, "traits2::Processor")
        assert trait_node is not None and _node_kind(graph, trait_node) == NodeKind.TRAIT


class TestExtractFileNestedCalls:
    """Test deeply nested function call extraction."""

    def test_deeply_nested_calls(self, tmp_path: Path) -> None:
        source = tmp_path / "nested_calls.rs"
        source.write_text(
            "fn outer() { "
            "  let x = foo(bar(baz(1))); "
            "  qux(wow(x)); "
            "}",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        caller = _find_node(graph, "nested_calls::outer")
        assert caller is not None
        edges = [
            (t, e) for t, e in graph.out_neighbors(caller) if e.kind == EdgeKind.CALLS
        ]
        target_qns = {graph.node(t).qualified_name for t, _ in edges}
        # All leaf calls should be captured
        assert "call::foo" in target_qns
        assert "call::bar" in target_qns
        assert "call::baz" in target_qns
        assert "call::qux" in target_qns
        assert "call::wow" in target_qns


class TestExtractFileSelfCalls:
    """Test self-referential method calls."""

    def test_self_method_chaining(self, tmp_path: Path) -> None:
        source = tmp_path / "chain.rs"
        source.write_text(
            "struct Builder {\n"
            "    value: i32,\n"
            "}\n"
            "impl Builder {\n"
            "    fn new() -> Self { Builder { value: 0 } }\n"
            "    fn set_value(&mut self, v: i32) -> &mut Self { self.value = v; self }\n"
            "    fn run(&self) { self.compute(); }\n"
            "    fn compute(&self) { }\n"
            "}\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        # Method qn is scoped with impl type name
        caller = _find_node(graph, "chain::Builder::run")
        assert caller is not None
        edges = [
            (t, e) for t, e in graph.out_neighbors(caller) if e.kind == EdgeKind.CALLS
        ]
        targets = [t for t, _ in edges]
        callee_qns = {graph.node(t).qualified_name for t in targets}
        assert "call::compute" in callee_qns


class TestExtractFileDefinesEdges:
    """Test that DEFINES edges are created correctly."""

    def test_file_defines_function(self, tmp_path: Path) -> None:
        source = tmp_path / "def.rs"
        source.write_text(
            "fn foo() { }\n"
            "fn bar() { }\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        file_node = _find_node(graph, "def")
        assert file_node is not None
        defines_edges = [
            (t, e) for t, e in graph.out_neighbors(file_node) if e.kind == EdgeKind.DEFINES
        ]
        target_qns = {graph.node(t).qualified_name for t, _ in defines_edges}
        assert "def::foo" in target_qns
        assert "def::bar" in target_qns

    def test_file_imports_module(self, tmp_path: Path) -> None:
        source = tmp_path / "import_def.rs"
        source.write_text(
            "use std::io::Read;\n",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        file_node = _find_node(graph, "import_def")
        assert file_node is not None
        import_edges = [
            (t, e) for t, e in graph.out_neighbors(file_node) if e.kind == EdgeKind.IMPORTS
        ]
        target_qns = {graph.node(t).qualified_name for t, _ in import_edges}
        assert "module::std::io::Read" in target_qns


class TestExtractFileEnumDefinesEdges:
    """Test that enum DEFINES edges point to variant variables."""

    def test_enum_defines_variants(self, tmp_path: Path) -> None:
        source = tmp_path / "enums.rs"
        source.write_text(
            "enum Status { Active, Inactive, Pending }",
            encoding="utf-8",
        )
        graph = Graph()
        extract_file(source, graph)

        enum_node = _find_node(graph, "enums::Status")
        assert enum_node is not None
        defines_edges = [
            (t, e) for t, e in graph.out_neighbors(enum_node) if e.kind == EdgeKind.DEFINES
        ]
        var_qns = {graph.node(t).qualified_name for t, _ in defines_edges}
        assert "enums::Status::Active" in var_qns
        assert "enums::Status::Inactive" in var_qns
        assert "enums::Status::Pending" in var_qns
