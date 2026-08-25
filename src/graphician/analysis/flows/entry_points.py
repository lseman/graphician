"""Entry point detection for flow tracing."""

from __future__ import annotations

from ...core.node import Node


def _is_python_framework_entry(node: Node) -> bool:
    """Python framework entry detection: Flask, FastAPI, Celery, Django, etc."""
    name = node.name
    qn = node.qualified_name

    # Name patterns
    if name.startswith("route_") or name.startswith("api_"):
        return True
    if name in ("get", "post", "put", "delete", "patch") and (
        "router" in qn or "route" in qn
    ):
        return True

    return _check_decorators(
        node,
        {
            # Flask/FastAPI
            "@route", "@api", "@endpoint", "@handler", "@command",
            "@task", "@job", "@cron", "@app.route", "@blueprint",
            "@api_view", "@login_required", "@permission",
            # Celery
            "@celery", "@shared_task", "@app.task",
            # Django
            "@csrf_exempt", "@csrf_protect", "@never_cache",
            "@require_http_methods",
            # Pytest
            "@pytest", "@fixture", "@parametrize", "@mark.",
            # SQLAlchemy
            "@declared_attr", "@validates", "@listens_for", "@hybrid_property",
            # Click
            "@click", "@command()", "@group()", "@option",
            # Starlette/FastAPI
            "@app.websocket", "@app.on_event", "@middleware",
            # Async
            "@asyncio", "@coroutine",
            # GraphQL
            "@graphql", "@resolver", "@type_()", "@field",
            # Dataclass
            "@dataclass", "@attr.s", "@attr.attrs",
            # Django/SQLAlchemy models
            "@models.register", "@db.model", "@db.entity",
            "@declarative", "@registry", "@serializable",
        },
    )


def _is_js_ts_framework_entry(node: Node) -> bool:
    """JS/TS framework entry detection: React, Angular, NestJS, Express, etc."""
    name = node.name

    # Name patterns
    if any(name.startswith(p) for p in (
        "handle_", "on_", "serve_", "middleware_", "route_",
        "endpoint_", "controller_", "action_", "callback_",
    )):
        return True

    return _check_decorators(
        node,
        {
            # React
            "@react.component", "@memo", "@observer", "@inject",
            "@react.observable",
            # Angular
            "@component", "@directive", "@pipe", "@injectable",
            "@module", "@NgModule", "@input", "@output",
            "@contentchild", "@contentchildren", "@viewchild",
            "@viewchildren", "@hostlistener", "@hostbinding",
            # NestJS
            "@controller", "@get", "@post", "@put", "@delete",
            "@patch", "@head", "@options", "@all", "@useguard",
            "@useinterceptor", "@filters", "@resolve", "@param",
            "@body", "@query", "@req", "@res", "@headers",
            "@cookies", "@session", "@sse",
            # Express
            "@express", "@route", "app.", "router.",
            # MobX
            "@action", "@action.bound", "@computed",
            "@observable", # Vue
            "@definecomponent", "@vue/component", "@prop",
            "@emit", "@model",
            # Test decorators
            "@test", "@spec", "@describe", "@beforeeach",
            "@aftereach", "@beforeall", "@afterall",
            # GraphQL
            "@graphql", "@resolver", "@field",
            # Electron
            "@electron", "@ipcmain", "@ipcrenderer",
        },
    )


def _is_java_framework_entry(node: Node) -> bool:
    """Java framework entry detection: Spring, JAX-RS, JUnit, CDI."""
    annotations = node.properties.get("annotations")
    if not isinstance(annotations, list):
        return False

    all_anns = " ".join(str(a) for a in annotations).lower()

    java_patterns = {
        # Spring
        "@controller", "@restcontroller", "@service", "@repository",
        "@component", "@bean", "@autowired", "@resource", "@qualifier",
        "@primary", "@configurable", "@configuration", "@propertysource",
        "@value", "@profile", "@postconstruct", "@predestroy",
        "@transactional", "@async", "@scheduled", "@cacheable",
        "@caching", "@cacheput", "@cacheevict", "@entity", "@table",
        "@mappedsuperclass", "@embeddable",
        # JAX-RS
        "@path", "@get", "@post", "@put", "@delete", "@patch",
        "@head", "@options", "@consumes", "@produces",
        "@requestscope", "@applicationscope", "@sessionscope",
        "@context", "@inject", "@provider", "@feature", "@connector",
        # JUnit
        "@test", "@beforeeach", "@aftereach", "@beforeall",
        "@afterall", "@disabled", "@enabledif", "@tag", "@nested",
        "@displayname",
        # CDI
        "@singleton", "@managedbean", "@named",
        "@requestscoped", "@sessionscoped", "@applicationscoped",
        "@conversationscoped",
    }

    return any(p in all_anns for p in java_patterns)


def _is_generic_event_entry(node: Node) -> bool:
    """Generic event-driven entry patterns: on_, handle_, dispatch_, emit_, trigger_."""
    name = node.name
    if name.startswith("on_") and len(name) > 3:
        return True
    if name.startswith("handle_") and len(name) > 7:
        return True
    if name.startswith("dispatch_") and len(name) > 9:
        return True
    if name.startswith("emit_") and len(name) > 5:
        return True
    return bool(name.startswith("trigger_") and len(name) > 8)


def _check_decorators(node: Node, patterns: set[str]) -> bool:
    """Check if any decorator annotation matches known patterns."""
    decorators = node.properties.get("decorators")
    if not isinstance(decorators, list):
        return False
    all_dec = " ".join(str(d).lower() for d in decorators)
    return any(p in all_dec for p in patterns)


def _is_framework_entry(node: Node) -> bool:
    """Detect framework-decorated entry points across 30+ patterns."""
    return (
        _is_python_framework_entry(node)
        or _is_js_ts_framework_entry(node)
        or _is_java_framework_entry(node)
        or _is_generic_event_entry(node)
    )
