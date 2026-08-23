"""Framework pattern database.

Detects framework-specific patterns (React hooks, Express routes,
Spring beans, etc.) by scanning the graph for signature structures:
specific function/class signatures, import patterns, and edge
configurations that indicate a framework is in use.

Patterns are matched at runtime against the extracted graph.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any

from graphician.core.edge import EdgeKind
from graphician.core.graph import Graph
from graphician.core.id import NodeId
from graphician.core.node import NodeKind


# ---------------------------------------------------------------------------
# Pattern definition
# ---------------------------------------------------------------------------

class PatternCategory(str, enum.Enum):
    """Category of framework pattern."""
    DEPENDENCY_INJECTION = "dependency_injection"
    ROUTING = "routing"
    LIFECYCLE = "lifecycle"
    STATE_MANAGEMENT = "state_management"
    VALIDATION = "validation"
    MIDDLEWARE = "middleware"
    DATA_MAPPING = "data_mapping"
    TESTING = "testing"
    COMMAND_LINE = "command_line"
    GENERIC = "generic"


@dataclass(frozen=True)
class FrameworkPattern:
    """A single framework pattern definition."""
    id: str
    display_name: str
    description: str
    framework: str
    category: PatternCategory
    min_confidence: float = 0.5
    required_node_kinds: list[NodeKind] = field(default_factory=list)
    required_edge_kinds: list[EdgeKind] = field(default_factory=list)
    signature_names: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    min_nodes: int = 1
    max_nodes: int = 500
    requires_embeddings: bool = False


# ---------------------------------------------------------------------------
# Built-in catalog
# ---------------------------------------------------------------------------

def _built_in_patterns() -> list[FrameworkPattern]:
    """Return the built-in pattern catalog (~40 patterns)."""
    return [
        # ---- React / JSX ----
        FrameworkPattern(
            id="react_hooks",
            display_name="React Hooks",
            description="Detects React functional components using hooks "
                        "(useState, useEffect, etc.)",
            framework="react",
            category=PatternCategory.LIFECYCLE,
            min_confidence=0.6,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES, EdgeKind.CALLS],
            signature_names=[
                "useState", "useEffect", "useContext", "useReducer",
                "useMemo", "useCallback", "useRef",
                "useImperativeHandle", "useLayoutEffect",
                "useDebugValue", "useDeferredValue", "useTransition",
                "useInsertionEffect", "useSyncExternalStore",
            ],
            import_patterns=["react", "@react"],
            min_nodes=2,
            max_nodes=200,
        ),
        FrameworkPattern(
            id="react_class_component",
            display_name="React Class Components",
            description="Detects legacy React class components with "
                        "lifecycle methods",
            framework="react",
            category=PatternCategory.LIFECYCLE,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "componentDidMount", "componentDidUpdate",
                "componentWillUnmount", "shouldComponentUpdate",
                "getDerivedStateFromProps", "getSnapshotBeforeUpdate",
            ],
            import_patterns=["react"],
            min_nodes=1,
            max_nodes=50,
        ),
        FrameworkPattern(
            id="react_router",
            display_name="React Router",
            description="Detects React Router route definitions",
            framework="react",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES, EdgeKind.CALLS],
            signature_names=[
                "Route", "Routes", "BrowserRouter", "Switch",
                "useNavigate", "useParams", "useLocation", "Link",
            ],
            import_patterns=["react-router"],
            min_nodes=2,
            max_nodes=100,
        ),
        # ---- Express / Node.js ----
        FrameworkPattern(
            id="express_middleware",
            display_name="Express Middleware",
            description="Detects Express middleware functions and app.use() "
                        "calls",
            framework="express",
            category=PatternCategory.MIDDLEWARE,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES, EdgeKind.CALLS],
            signature_names=["app.use", "router.use", "next"],
            import_patterns=["express"],
            min_nodes=2,
            max_nodes=200,
        ),
        FrameworkPattern(
            id="express_routes",
            display_name="Express Routes",
            description="Detects Express route handlers (app.get, app.post, "
                        "etc.)",
            framework="express",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES, EdgeKind.CALLS],
            signature_names=[
                "app.get", "app.post", "app.put", "app.delete",
                "app.patch", "router.get", "router.post",
            ],
            import_patterns=["express"],
            min_nodes=2,
            max_nodes=200,
        ),
        # ---- Spring / Java ----
        FrameworkPattern(
            id="spring_di",
            display_name="Spring Dependency Injection",
            description="Detects Spring DI annotations: @Autowired, @Inject, "
                        "@Component, @Bean",
            framework="spring",
            category=PatternCategory.DEPENDENCY_INJECTION,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Autowired", "Inject", "Component", "Service",
                "Repository", "Controller", "RestController", "Bean",
                "Configuration", "Qualifier", "Primary",
            ],
            import_patterns=["org.springframework", "javax.inject"],
            min_nodes=2,
            max_nodes=500,
        ),
        FrameworkPattern(
            id="spring_rest",
            display_name="Spring REST Controllers",
            description="Detects Spring REST endpoint mappings",
            framework="spring",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "GetMapping", "PostMapping", "PutMapping", "DeleteMapping",
                "PatchMapping", "RequestMapping", "ResponseBody",
                "RequestBody",
            ],
            import_patterns=["org.springframework"],
            min_nodes=2,
            max_nodes=200,
        ),
        FrameworkPattern(
            id="spring_jpa",
            display_name="Spring Data JPA",
            description="Detects JPA entity classes and repository interfaces",
            framework="spring",
            category=PatternCategory.DATA_MAPPING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Entity", "Table", "Id", "GeneratedValue", "OneToMany",
                "ManyToOne", "ManyToMany", "OneToOne", "Column",
                "Transient", "Repository", "CrudRepository",
                "JpaRepository",
            ],
            import_patterns=["javax.persistence", "jakarta.persistence"],
            min_nodes=2,
            max_nodes=300,
        ),
        # ---- Django / Python ----
        FrameworkPattern(
            id="django_middleware",
            display_name="Django Middleware",
            description="Detects Django middleware classes and MIDDLEWARE "
                        "setting",
            framework="django",
            category=PatternCategory.MIDDLEWARE,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "process_request", "process_response", "process_view",
                "process_exception", "process_template_response",
            ],
            import_patterns=["django"],
            min_nodes=2,
            max_nodes=100,
        ),
        FrameworkPattern(
            id="django_models",
            display_name="Django Models",
            description="Detects Django ORM model classes",
            framework="django",
            category=PatternCategory.DATA_MAPPING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Model", "CharField", "IntegerField", "ForeignKey",
                "ManyToManyField", "TextField", "DateField",
                "DateTimeField", "BooleanField", "EmailField",
                "URLField", "OneToOneField",
            ],
            import_patterns=["django.db"],
            min_nodes=2,
            max_nodes=300,
        ),
        FrameworkPattern(
            id="django_views",
            display_name="Django Views",
            description="Detects Django view functions and class-based views",
            framework="django",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "View", "TemplateView", "ListView", "DetailView",
                "CreateView", "UpdateView", "DeleteView", "FormView",
            ],
            import_patterns=["django.views"],
            min_nodes=1,
            max_nodes=200,
        ),
        # ---- FastAPI / Python ----
        FrameworkPattern(
            id="fastapi_routes",
            display_name="FastAPI Routes",
            description="Detects FastAPI route decorators and dependency "
                        "injection",
            framework="fastapi",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "APIRouter", "Depends", "Body", "Query", "Path", "Header",
            ],
            import_patterns=["fastapi", "starlette"],
            min_nodes=2,
            max_nodes=200,
        ),
        # ---- Angular / TypeScript ----
        FrameworkPattern(
            id="angular_components",
            display_name="Angular Components",
            description="Detects Angular component decorators and module "
                        "definitions",
            framework="angular",
            category=PatternCategory.GENERIC,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Component", "Module", "Directive", "Pipe", "Injectable",
                "Input", "Output", "ViewChild", "HostListener",
                "OnInit", "OnDestroy", "ngOnInit", "ngOnDestroy",
                "NgModule",
            ],
            import_patterns=["@angular/core", "@angular"],
            min_nodes=2,
            max_nodes=500,
        ),
        # ---- NestJS / TypeScript ----
        FrameworkPattern(
            id="nestjs_controllers",
            display_name="NestJS Controllers",
            description="Detects NestJS controller decorators and route "
                        "handlers",
            framework="nestjs",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Controller", "Get", "Post", "Put", "Delete", "Patch",
                "Body", "Param", "Query", "Inject", "Injectable",
                "Module", "UseGuards", "UseInterceptors",
                "HttpException",
            ],
            import_patterns=["@nestjs"],
            min_nodes=2,
            max_nodes=500,
        ),
        # ---- Vue / Composition API ----
        FrameworkPattern(
            id="vue_composition",
            display_name="Vue Composition API",
            description="Detects Vue 3 Composition API usage "
                        "(ref, reactive, computed, etc.)",
            framework="vue",
            category=PatternCategory.STATE_MANAGEMENT,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES, EdgeKind.CALLS],
            signature_names=[
                "ref", "reactive", "computed", "watch", "watchEffect",
                "onMounted", "onUnmounted", "onBeforeMount",
                "onBeforeUnmount", "provide", "inject",
                "defineComponent",
            ],
            import_patterns=["vue"],
            min_nodes=2,
            max_nodes=200,
        ),
        # ---- Flask / Python ----
        FrameworkPattern(
            id="flask_routes",
            display_name="Flask Routes",
            description="Detects Flask route decorators and application "
                        "factory pattern",
            framework="flask",
            category=PatternCategory.ROUTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Blueprint", "before_request", "after_request",
                "teardown_request", "errorhandler", "url_for",
                "render_template", "jsonify",
            ],
            import_patterns=["flask"],
            min_nodes=2,
            max_nodes=200,
        ),
        # ---- Testing frameworks ----
        FrameworkPattern(
            id="jest_tests",
            display_name="Jest Test Suite",
            description="Detects Jest test patterns (describe, it, test, "
                        "beforeEach, etc.)",
            framework="jest",
            category=PatternCategory.TESTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "describe", "it", "test", "beforeEach", "afterEach",
                "beforeAll", "afterAll", "expect", "jest.fn",
                "jest.mock", "jest.spyOn",
            ],
            import_patterns=["jest"],
            min_nodes=3,
            max_nodes=500,
        ),
        FrameworkPattern(
            id="pytest_tests",
            display_name="Pytest Test Suite",
            description="Detects pytest test functions, fixtures, and "
                        "parametrize patterns",
            framework="pytest",
            category=PatternCategory.TESTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "fixture", "parametrize", "pytestmark", "conftest",
                "tmp_path", "monkeypatch", "caplog", "capfd",
            ],
            import_patterns=["pytest"],
            min_nodes=3,
            max_nodes=500,
        ),
        FrameworkPattern(
            id="junit_tests",
            display_name="JUnit Test Suite",
            description="Detects JUnit test annotations (@Test, "
                        "@BeforeEach, @ParameterizedTest, etc.)",
            framework="junit",
            category=PatternCategory.TESTING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Test", "BeforeEach", "AfterEach", "BeforeAll",
                "AfterAll", "ParameterizedTest", "RepeatedTest",
                "Nested", "Disabled", "Ignore", "AssertThat",
                "AssertEquals",
            ],
            import_patterns=["org.junit"],
            min_nodes=3,
            max_nodes=500,
        ),
        # ---- CLI frameworks ----
        FrameworkPattern(
            id="clap_cli",
            display_name="Clap CLI Parser",
            description="Detects Clap CLI argument parsing patterns",
            framework="clap",
            category=PatternCategory.COMMAND_LINE,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Parser", "Args", "Arg", "Subcommand", "ValueEnum",
                "derive", "command", "arg", "subcommand",
            ],
            import_patterns=["clap"],
            min_nodes=2,
            max_nodes=100,
        ),
        FrameworkPattern(
            id="click_cli",
            display_name="Click CLI",
            description="Detects Click CLI command decorators and group "
                        "patterns",
            framework="click",
            category=PatternCategory.COMMAND_LINE,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "command", "group", "option", "argument",
                "pass_context", "echo", "prompt", "confirm",
            ],
            import_patterns=["click"],
            min_nodes=2,
            max_nodes=100,
        ),
        # ---- State management ----
        FrameworkPattern(
            id="redux_state",
            display_name="Redux State Management",
            description="Detects Redux store, reducer, and action patterns",
            framework="redux",
            category=PatternCategory.STATE_MANAGEMENT,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "createStore", "combineReducers", "createSlice",
                "configureStore", "createAsyncThunk", "useSelector",
                "useDispatch", "dispatch", "reducer", "action",
                "actionCreator",
            ],
            import_patterns=["redux", "@reduxjs/toolkit"],
            min_nodes=3,
            max_nodes=300,
        ),
        # ---- Validation ----
        FrameworkPattern(
            id="zod_schemas",
            display_name="Zod Schema Validation",
            description="Detects Zod schema definitions and validation "
                        "patterns",
            framework="zod",
            category=PatternCategory.VALIDATION,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "z.object", "z.string", "z.number", "z.boolean",
                "z.array", "z.enum", "z.union", "z.intersection",
                "z.nullable", "z.optional", "z.coerce", "z.transform",
                "z.preprocess", "z.lazy", "z.instanceof", "z.literal",
                "z.record", "z.tuple",
            ],
            import_patterns=["zod"],
            min_nodes=2,
            max_nodes=200,
        ),
        FrameworkPattern(
            id="pydantic_models",
            display_name="Pydantic Models",
            description="Detects Pydantic model definitions and field "
                        "validators",
            framework="pydantic",
            category=PatternCategory.VALIDATION,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "BaseModel", "Field", "validator", "model_validator",
                "root_validator", "PrivateAttr", "Config",
            ],
            import_patterns=["pydantic"],
            min_nodes=2,
            max_nodes=200,
        ),
        # ---- ORM ----
        FrameworkPattern(
            id="prisma_orm",
            display_name="Prisma ORM",
            description="Detects Prisma schema patterns and client usage",
            framework="prisma",
            category=PatternCategory.DATA_MAPPING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "PrismaClient", "model", "enum", "Prisma",
                "createPrismaClient",
            ],
            import_patterns=["@prisma/client", "prisma"],
            min_nodes=2,
            max_nodes=200,
        ),
        FrameworkPattern(
            id="sequelize_orm",
            display_name="Sequelize ORM",
            description="Detects Sequelize model definitions and associations",
            framework="sequelize",
            category=PatternCategory.DATA_MAPPING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS, NodeKind.FUNCTION],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Model", "init", "hasMany", "belongsTo", "hasOne",
                "belongsToMany", "DataTypes", "STRING", "INTEGER",
                "BOOLEAN", "TEXT",
            ],
            import_patterns=["sequelize"],
            min_nodes=3,
            max_nodes=300,
        ),
        FrameworkPattern(
            id="sqlalchemy_orm",
            display_name="SQLAlchemy ORM",
            description="Detects SQLAlchemy model declarations and "
                        "relationship patterns",
            framework="sqlalchemy",
            category=PatternCategory.DATA_MAPPING,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "declarative_base", "Column", "Integer", "String",
                "ForeignKey", "relationship", "backref",
                "OneToOne", "OneToMany", "ManyToMany", "mapped_column",
            ],
            import_patterns=["sqlalchemy"],
            min_nodes=2,
            max_nodes=300,
        ),
        # ---- GraphQL ----
        FrameworkPattern(
            id="graphql_schema",
            display_name="GraphQL Schema",
            description="Detects GraphQL schema definitions and resolvers",
            framework="graphql",
            category=PatternCategory.GENERIC,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "GraphQLSchema", "GraphQLObjectType",
                "GraphQLInterfaceType", "GraphQLUnionType",
                "GraphQLScalarType", "GraphQLInputObjectType",
                "GraphQLList", "GraphQLNonNull", "makeExecutableSchema",
                "createSchema", "defineResolvers", "Field", "Query",
                "Mutation", "Subscription",
            ],
            import_patterns=["graphql"],
            min_nodes=3,
            max_nodes=300,
        ),
        # ---- Message queues / event systems ----
        FrameworkPattern(
            id="rabbitmq_consumer",
            display_name="RabbitMQ Consumer",
            description="Detects RabbitMQ consumer/producer patterns",
            framework="rabbitmq",
            category=PatternCategory.GENERIC,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "channel", "queue_declare", "basic_publish",
                "basic_consume", "basic_ack", "exchange_declare",
                "binding_declare",
            ],
            import_patterns=["pika", "amqp"],
            min_nodes=2,
            max_nodes=50,
        ),
        FrameworkPattern(
            id="redis_cache",
            display_name="Redis Caching",
            description="Detects Redis client usage and caching patterns",
            framework="redis",
            category=PatternCategory.GENERIC,
            min_confidence=0.5,
            required_node_kinds=[NodeKind.FUNCTION, NodeKind.CLASS],
            required_edge_kinds=[EdgeKind.DEFINES],
            signature_names=[
                "Redis", "set", "get", "delete", "expire", "ttl",
                "incr", "decr", "hset", "hget", "sadd", "lpush", "rpush",
            ],
            import_patterns=["redis"],
            min_nodes=2,
            max_nodes=100,
        ),
    ]


# ---------------------------------------------------------------------------
# Pattern match result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternMatch:
    """A single matched framework pattern in the graph."""
    pattern_id: str
    display_name: str
    framework: str
    category: str
    matched_node_ids: list[str]
    matched_node_names: list[str]
    matched_edge_kinds: list[str]
    confidence: float
    source_uris: list[str]


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

def detect_patterns(graph: Graph) -> list[PatternMatch]:
    """Run all built-in patterns against the graph.

    Returns a list of matches, one per detected pattern instance.
    """
    patterns = _built_in_patterns()
    results: list[PatternMatch] = []
    for pattern in patterns:
        if pattern.requires_embeddings:
            continue
        matches = _match_single_pattern(graph, pattern)
        results.extend(matches)
    return results


def _match_single_pattern(
    graph: Graph, pattern: FrameworkPattern,
) -> list[PatternMatch]:
    """Match a single pattern against the graph."""
    # Step 1: find candidate nodes matching signature_names and
    # required_node_kinds.
    signature_nodes: list[tuple[NodeId, str, str]] = []
    for nid, node in graph.nodes():
        # Check node kind filter.
        if pattern.required_node_kinds and node.kind not in pattern.required_node_kinds:
            continue
        # Check signature name filter.
        if pattern.signature_names:
            name = node.name
            qn = node.qualified_name
            has_match = any(
                sig in name or sig in qn
                for sig in pattern.signature_names
            )
            if not has_match:
                continue
        signature_nodes.append((nid, node.name, node.source_uri or ""))

    if not signature_nodes:
        return []

    # Step 2: collect edge kinds from outgoing edges of signature nodes.
    edge_kinds_found: set[EdgeKind] = set()
    for nid, _, _ in signature_nodes:
        for _neighbor, edge in graph.out_neighbors(nid):
            edge_kinds_found.add(edge.kind)

    # Step 3: check required edge kinds.
    if pattern.required_edge_kinds:
        if not all(
            req in edge_kinds_found
            for req in pattern.required_edge_kinds
        ):
            return []

    # Step 4: check import patterns via source URIs.
    if pattern.import_patterns:
        has_import = any(
            imp in uri
            for _, _, uri in signature_nodes
            for imp in pattern.import_patterns
        )
        if not has_import:
            return []

    # Step 5: compute confidence and build result.
    confidence = _compute_match_confidence(
        pattern, signature_nodes, edge_kinds_found,
    )
    if confidence < pattern.min_confidence:
        return []

    return [PatternMatch(
        pattern_id=pattern.id,
        display_name=pattern.display_name,
        framework=pattern.framework,
        category=pattern.category.value,
        matched_node_ids=[str(nid.value) for nid, _, _ in signature_nodes],
        matched_node_names=[name for _, name, _ in signature_nodes],
        matched_edge_kinds=[ek.value for ek in edge_kinds_found],
        confidence=round(confidence, 3),
        source_uris=[uri for _, _, uri in signature_nodes if uri],
    )]


def _compute_match_confidence(
    pattern: FrameworkPattern,
    signature_nodes: list[tuple[NodeId, str, str]],
    edge_kinds_found: set[EdgeKind],
) -> float:
    """Compute a confidence score for a pattern match."""
    score = 0.0
    max_score = 0.0

    # Node count contribution (0-0.3): prefer matches within the expected
    # range.
    max_score += 0.3
    node_count = len(signature_nodes)
    expected = (pattern.min_nodes + pattern.max_nodes) / 2.0
    if expected > 0.0:
        ratio = node_count / expected
        score += 0.3 * (1.0 - min(abs(ratio - 1.0), 1.0))

    # Edge kind coverage (0-0.4): how many required edge kinds are
    # present.
    if pattern.required_edge_kinds:
        max_score += 0.4
        covered = sum(
            1 for rk in pattern.required_edge_kinds
            if rk in edge_kinds_found
        )
        ratio = covered / len(pattern.required_edge_kinds)
        score += 0.4 * ratio

    # Signature name coverage (0-0.3): how many signature names match.
    if pattern.signature_names:
        max_score += 0.3
        matched_sigs: set[str] = set()
        for _, name, _ in signature_nodes:
            for sig in pattern.signature_names:
                if sig in name:
                    matched_sigs.add(sig)
        ratio = len(matched_sigs) / len(pattern.signature_names)
        score += 0.3 * min(ratio, 1.0)

    # Normalize to [0, 1].
    if max_score > 0.0:
        return min(score / max_score, 1.0)
    return 0.5  # default if no scoring dimensions


# ---------------------------------------------------------------------------
# Pattern catalog management
# ---------------------------------------------------------------------------

def load_patterns_from_file(path: str) -> list[FrameworkPattern]:
    """Load additional patterns from a TOML file."""
    import tomllib
    with open(path, "rb") as f:
        raw = f.read()
    parsed = tomllib.loads(raw.decode())
    patterns: list[FrameworkPattern] = []
    for item in parsed.get("patterns", []):
        # Convert string category to enum.
        if "category" in item and isinstance(item["category"], str):
            item["category"] = PatternCategory(item["category"])
        patterns.append(FrameworkPattern(**item))
    return patterns


def merged_catalog(
    custom: list[FrameworkPattern],
) -> list[FrameworkPattern]:
    """Merge custom patterns with the built-in catalog."""
    catalog = list(_built_in_patterns())
    custom_ids = {p.id for p in custom}
    catalog = [p for p in catalog if p.id not in custom_ids]
    catalog.extend(custom)
    return catalog
