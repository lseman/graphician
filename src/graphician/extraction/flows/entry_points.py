"""Entry point detection for flow tracing."""

from __future__ import annotations

from ...core.node import Node, NodeKind
from ...core.edge import EdgeKind
from .trace import _is_test_node


def detect_entry_points(graph) -> list:
    """Detect entry points across all nodes."""
    entries = []
    for nid, node in graph.nodes():
        if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD):
            continue
        if node.qualified_name.startswith("call::"):
            continue
        if _is_test_node(node):
            entries.append(nid)
            continue
        if node.name in ("main", "__main__"):
            entries.append(nid)
            continue
        if _is_framework_entry(node):
            entries.append(nid)
            continue
        has_caller = any(
            edge.kind == EdgeKind.CALLS
            for _, edge in graph.in_neighbors(nid)
        )
        if not has_caller:
            entries.append(nid)
    return entries


def _is_framework_entry(node: Node) -> bool:
    """Detect framework-decorated entry points."""
    return (
        _is_python_framework_entry(node)
        or _is_js_ts_framework_entry(node)
        or _is_generic_event_entry(node)
    )


def _is_python_framework_entry(node: Node) -> bool:
    """Python framework entry detection."""
    name = node.name
    qn = node.qualified_name
    if name.startswith("route_") or name.startswith("api_"):
        return True
    if name in ("get", "post", "put", "delete", "patch") and (
        "router" in qn or "route" in qn
    ):
        return True
    decorators = node.properties.get("decorators")
    if isinstance(decorators, list):
        all_dec = " ".join(str(d).lower() for d in decorators)
        python_patterns = {
            "@route", "@api", "@endpoint", "@handler", "@command",
            "@task", "@job", "@cron", "@app.route", "@blueprint",
            "@api_view", "@login_required", "@permission",
            "@celery", "@shared_task", "@app.task",
            "@csrf_exempt", "@csrf_protect",
            "@pytest", "@fixture", "@parametrize",
            "@click", "@group()", "@option",
            "@graphql", "@resolver",
        }
        return any(p in all_dec for p in python_patterns)
    return False


def _is_js_ts_framework_entry(node: Node) -> bool:
    """JS/TS framework entry detection."""
    name = node.name
    if any(name.startswith(p) for p in (
        "handle_", "on_", "serve_", "middleware_", "route_",
        "endpoint_", "controller_", "action_", "callback_",
    )):
        return True
    decorators = node.properties.get("decorators")
    if isinstance(decorators, list):
        all_dec = " ".join(str(d).lower() for d in decorators)
        js_patterns = {
            "@component", "@directive", "@pipe", "@injectable",
            "@controller", "@get", "@post", "@put", "@delete",
            "@test", "@spec", "@describe",
        }
        return any(p in all_dec for p in js_patterns)
    return False


def _is_generic_event_entry(node: Node) -> bool:
    """Generic event-driven entry patterns."""
    name = node.name
    if name.startswith("on_") and len(name) > 3:
        return True
    if name.startswith("handle_") and len(name) > 7:
        return True
    if name.startswith("dispatch_") and len(name) > 9:
        return True
    if name.startswith("emit_") and len(name) > 5:
        return True
    if name.startswith("trigger_") and len(name) > 8:
        return True
    return False
