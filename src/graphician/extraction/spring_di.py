"""Spring dependency-injection resolution for Java.

Java's field/constructor-parameter extraction (``extract/languages/parsers/java.py``)
captures ``declared_type`` + ``annotations`` + optional ``qualifier`` on
``Variable`` nodes for ``@Autowired`` fields and constructor parameters,
but doesn't know which concrete class satisfies an interface-typed
dependency — that requires looking at the whole graph (every class
``Implements``-ing the interface), so it runs as a post-build enrichment
pass.

For each injected field/parameter whose declared type resolves to a
``Trait`` node (Java interfaces extract as ``Trait``), finds candidate
``Class`` nodes that ``Implements`` it and carry a Spring stereotype
annotation (``@Component``, ``@Service``, ``@Repository``,
``@Controller``, ``@RestController``). If exactly one candidate remains
(after ``@Qualifier`` filtering, when present), adds a ``DependsOn`` edge
from the injecting class to the implementation. Ambiguous cases (zero
or multiple surviving candidates) are skipped, not guessed.
"""

from __future__ import annotations

from collections import defaultdict

from ..core.edge import Confidence, Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import Node, NodeKind

STEREOTYPE_ANNOTATIONS: tuple[str, ...] = (
    "Component",
    "Service",
    "Repository",
    "Controller",
    "RestController",
)


def resolve_spring_injections(graph: Graph) -> int:
    """Resolve Spring dependency injections into ``DependsOn`` edges.

    Returns the number of injection edges added.
    """
    injection_sites = _collect_injection_sites(graph)
    traits_by_name = _build_traits_by_name(graph)
    added = 0

    for site in injection_sites:
        owning_class = _owning_class(graph, site.field_id)
        if owning_class is None:
            continue

        bare_type = _bare_type_name(site.declared_type)
        iface_candidates = traits_by_name.get(bare_type)
        if not iface_candidates:
            continue
        # Ambiguous interface name (two same-named traits in different
        # packages) — don't guess.
        if len(iface_candidates) != 1:
            continue
        iface_id = iface_candidates[0]

        candidates = _implementing_stereotype_classes(graph, iface_id)
        if len(candidates) > 1 and site.qualifier is not None:
            candidates = [
                c for c in candidates
                if _bean_name_matches(graph, c, site.qualifier)
            ]

        if len(candidates) != 1:
            continue  # Ambiguous or no match.
        target = candidates[0]

        if owning_class == target:
            continue  # Self-injection is never meaningful here.
        if graph.out_neighbors(owning_class):
            already = any(
                d == target and e.kind == EdgeKind.DEPENDS_ON
                for d, e in graph.out_neighbors(owning_class)
            )
            if already:
                continue

        edge = Edge(EdgeKind.DEPENDS_ON, Confidence.INFERRED)
        edge.properties["resolved_from"] = "spring_di"
        graph.add_edge(owning_class, target, edge)
        added += 1

    return added


class _InjectionSite:
    __slots__ = ("field_id", "declared_type", "qualifier")

    def __init__(
        self,
        field_id: NodeId,
        declared_type: str,
        qualifier: str | None,
    ) -> None:
        self.field_id = field_id
        self.declared_type = declared_type
        self.qualifier = qualifier


def _collect_injection_sites(graph: Graph) -> list[_InjectionSite]:
    sites: list[_InjectionSite] = []
    for nid, node in graph.nodes():
        if node.kind != NodeKind.VARIABLE:
            continue
        declared_type = node.properties.get("declared_type")
        if not isinstance(declared_type, str):
            continue

        annotations = node.properties.get("annotations")
        is_autowired = False
        if isinstance(annotations, list):
            is_autowired = any(
                isinstance(a, str) and a == "Autowired" for a in annotations
            )

        is_ctor_param = "::param::" in node.qualified_name
        if not is_autowired and not is_ctor_param:
            continue

        qualifier = node.properties.get("qualifier")
        if not isinstance(qualifier, str):
            qualifier = None

        sites.append(_InjectionSite(nid, declared_type, qualifier))
    return sites


def _owning_class(graph: Graph, field_id: NodeId) -> NodeId | None:
    """Find the Class/Trait that Defines this field/parameter."""
    for parent, edge in graph.in_neighbors(field_id):
        if edge.kind == EdgeKind.DEFINES:
            parent_node = graph.node(parent)
            if parent_node is None:
                continue
            if parent_node.kind == NodeKind.CLASS:
                return parent
            if parent_node.kind == NodeKind.METHOD:
                for grandparent, edge2 in graph.in_neighbors(parent):
                    if edge2.kind == EdgeKind.DEFINES:
                        return grandparent
    return None


def _bare_type_name(declared_type: str) -> str:
    """Strip generic type arguments: ``List<PaymentGateway>`` → ``List``."""
    idx = declared_type.find("<")
    if idx >= 0:
        return declared_type[:idx].strip()
    return declared_type.strip()


def _build_traits_by_name(graph: Graph) -> dict[str, list[NodeId]]:
    """Map bare interface name → every Trait node with that name."""
    by_name: dict[str, list[NodeId]] = defaultdict(list)
    for nid, node in graph.nodes():
        if node.kind == NodeKind.TRAIT:
            by_name[node.name].append(nid)
    return by_name


def _implementing_stereotype_classes(
    graph: Graph, iface_id: NodeId
) -> list[NodeId]:
    result: list[NodeId] = []
    for class_id, edge in graph.in_neighbors(iface_id):
        if edge.kind != EdgeKind.IMPLEMENTS:
            continue
        node = graph.node(class_id)
        if node is not None and _has_stereotype_annotation(node):
            result.append(class_id)
    return result


def _has_stereotype_annotation(node: Node) -> bool:
    annotations = node.properties.get("annotations")
    if isinstance(annotations, list):
        for a in annotations:
            if isinstance(a, str) and a in STEREOTYPE_ANNOTATIONS:
                return True
    return False


def _bean_name_matches(graph: Graph, class_id: NodeId, qualifier: str) -> bool:
    node = graph.node(class_id)
    if node is None:
        return False
    return _default_bean_name(node.name) == qualifier


def _default_bean_name(class_name: str) -> str:
    """Decapitalize: ``StripeGateway`` → ``stripeGateway``."""
    if not class_name:
        return class_name
    return class_name[0].lower() + class_name[1:]
