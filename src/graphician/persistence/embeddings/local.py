"""Local feature-hash embedding model (ariadne-hash-v2).

Deterministic, no external dependencies. Complement to FTS5.
"""

from __future__ import annotations

import struct

DEFAULT_EMBEDDING_DIM: int = 384


def embedding_source_text(
    kind: str,
    name: str,
    qualified_name: str,
    source_uri: str | None = None,
    source_text: str | None = None,
) -> str:
    """Build the text from which an embedding is computed for a node.

    Includes metadata (kind, name, qualified_name) plus source_uri and
    source_text (function/class body). Source text provides the primary
    semantic signal for code-aware matching.
    """
    text = f"{kind} {name} {qualified_name.replace('::', ' ')}"
    if source_uri:
        text += f" {source_uri}"
    if source_text:
        text += f"\n{source_text}"
    return text


def semantic_embedding(text: str) -> list[float]:
    """Build a local feature-hash embedding for a text string."""
    vector = [0.0] * DEFAULT_EMBEDDING_DIM
    tokens = semantic_tokens(text)
    if not tokens:
        return vector

    unique_tokens = unique_ordered(tokens)
    for token in tokens:
        push_signed_hashed_feature(vector, f"tok:{token}", 1.25)
        push_signed_hashed_feature(vector, f"stem:{code_stem(token)}", 0.70)
        canonical = canonical_token(token)
        if canonical != token:
            push_signed_hashed_feature(vector, f"canon:{canonical}", 1.05)
        for gram in char_ngrams(token, 3, 5):
            push_signed_hashed_feature(vector, f"char:{gram}", 0.24)
        for piece in token_pieces(token):
            push_signed_hashed_feature(vector, f"piece:{piece}", 0.42)

    for i in range(len(tokens) - 1):
        push_signed_hashed_feature(vector, f"bi:{tokens[i]}:{tokens[i+1]}", 0.82)
        left = canonical_token(tokens[i])
        right = canonical_token(tokens[i+1])
        if left != tokens[i] or right != tokens[i+1]:
            push_signed_hashed_feature(vector, f"cbi:{left}:{right}", 0.58)

    for i in range(len(tokens) - 2):
        push_signed_hashed_feature(vector, f"tri:{tokens[i]}:{tokens[i+1]}:{tokens[i+2]}", 0.36)
        push_signed_hashed_feature(vector, f"skip:{tokens[i]}:{tokens[i+2]}", 0.28)

    acronym = "".join(t[0] for t in unique_tokens if t)
    if len(acronym) >= 2:
        push_signed_hashed_feature(vector, f"acro:{acronym}", 0.85)

    if is_code_like(text):
        code_features(unique_tokens, vector)

    for concept in semantic_concepts(unique_tokens):
        push_signed_hashed_feature(vector, f"concept:{concept}", 3.0)

    normalize_vector(vector)
    return vector


def semantic_tokens(raw: str) -> list[str]:
    """Tokenize text into semantic tokens."""
    normalized = []
    prev: str | None = None
    i = 0
    while i < len(raw):
        c = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        if c.isascii() and (c.isalnum() or c == '_'):
            if prev:
                camel_boundary = prev.islower() and c.isupper()
                acronym_boundary = prev.isupper() and c.isupper() and nxt and nxt.islower()
                digit_boundary = prev.isalpha() != c.isalpha()
                if camel_boundary or acronym_boundary or digit_boundary:
                    normalized.append(' ')
            normalized.append(c.lower())
            prev = c
        else:
            normalized.append(' ')
            prev = None
        i += 1

    return [singularize_token(t) for t in ''.join(normalized).split() if t]


def unique_ordered(tokens: list[str]) -> list[str]:
    """Return tokens in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def singularize_token(token: str) -> str:
    """Singularize a token (simple heuristic)."""
    if len(token) > 4 and token.endswith('s'):
        return token[:-1]
    return token


def canonical_token(token: str) -> str:
    """Map a token to its canonical form."""
    canon_map: dict[str, str] = {
        "delete": "remove", "deleted": "remove", "remove": "remove",
        "removed": "remove", "drop": "remove", "purge": "remove",
        "cleanup": "remove",
        "add": "add", "added": "add", "create": "add", "created": "add",
        "insert": "add", "new": "add",
        "change": "change", "changed": "change", "changes": "change",
        "diff": "change", "delta": "change", "modify": "change",
        "modified": "change", "update": "change", "updated": "change",
        "find": "search", "search": "search", "lookup": "search",
        "query": "search", "discover": "search",
        "auth": "auth", "authenticate": "auth", "authentication": "auth",
        "login": "auth", "signin": "auth", "signon": "auth",
        "test": "test", "tests": "test", "spec": "test", "specs": "test",
        "coverage": "test",
        "bug": "bug", "defect": "bug", "error": "bug", "failure": "bug",
        "panic": "bug", "regression": "bug",
        "cache": "cache", "cached": "cache", "memo": "cache",
        "memoize": "cache", "memoized": "cache",
        "config": "config", "configuration": "config", "setting": "config",
        "settings": "config",
        "db": "storage", "database": "storage", "sqlite": "storage",
        "store": "storage", "storage": "storage", "persist": "storage",
        "persistence": "storage",
        "doc": "doc", "docs": "doc", "document": "doc",
        "documentation": "doc", "readme": "doc",
        "embed": "embedding", "embedding": "embedding", "embeddings": "embedding",
        "semantic": "embedding", "vector": "embedding", "vectors": "embedding",
        "file": "source", "files": "source", "path": "source",
        "paths": "source", "source": "source", "sources": "source",
        "graph": "graph", "node": "graph", "nodes": "graph",
        "edge": "graph", "edges": "graph", "flow": "graph", "flows": "graph",
        "http": "server", "server": "server", "serve": "server",
        "route": "server", "routes": "server",
        "ignore": "ignore", "gitignore": "ignore", "graphicianignore": "ignore",
        "exclude": "ignore", "skip": "ignore",
        "index": "index", "indexed": "index", "indexing": "index",
        "fts": "index", "fts5": "index",
        "install": "install", "installer": "install", "setup": "install",
        "hook": "install", "hooks": "install",
        "json": "agent", "mcp": "agent", "tool": "agent", "tools": "agent",
        "agent": "agent", "agents": "agent",
        "rank": "rank", "ranking": "rank", "score": "rank",
        "scored": "rank", "scoring": "rank", "boost": "rank",
        "boosted": "rank",
        "read": "extract", "reader": "extract", "parse": "extract",
        "parser": "extract", "extract": "extract", "extraction": "extract",
        "review": "review", "risk": "review", "impact": "review",
        "blast": "review", "radius": "review",
        "symbol": "symbol", "symbols": "symbol", "function": "symbol",
        "functions": "symbol", "method": "symbol", "methods": "symbol",
        "ui": "ui", "viewer": "ui", "view": "ui",
        "watch": "watch", "daemon": "watch", "poll": "watch",
        "polling": "watch",
    }
    return canon_map.get(token, token)


def code_stem(token: str) -> str:
    """Extract a code stem from a token."""
    stem = singularize_token(token)
    for suffix in ["ing", "ed", "er", "or", "able", "ible", "tion", "ions"]:
        if len(stem) > len(suffix) + 3 and stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def is_code_like(text: str) -> bool:
    """Heuristic: is this text likely to be code?"""
    has_op = ['(', ')', '{', '}', '[', ']', '=', '=>', '->', '::', '.', '->']
    has_delim = any(c in text for c in ['(', ')', '{', '}', '[', ']'])
    has_keyword = ['def', 'fn', 'func', 'class', 'struct', 'interface',
                   'let', 'const', 'var', 'if', 'for', 'while', 'return',
                   'import', 'from', 'use', 'pub', 'private', 'public',
                   'async', 'await', 'try', 'catch', 'throw', 'raise']
    return any(kw in text for kw in has_keyword) or \
           any(op in text for op in has_op) or has_delim


def code_features(tokens: list[str], vector: list[float]) -> None:
    """Extract code-specific features for structural similarity."""
    token_set = set(tokens)

    for modifier in ["public", "private", "protected", "static", "pub", "const"]:
        if modifier in token_set:
            push_signed_hashed_feature(vector, f"access:{modifier}", 1.2)

    for type_kw in ["string", "int", "float", "bool", "boolean", "number",
                    "vec", "vector", "list", "array", "map", "dict",
                    "option", "optional", "nullable", "any", "void", "null"]:
        if type_kw in token_set:
            push_signed_hashed_feature(vector, f"type:{type_kw}", 1.1)

    for async_kw in ["async", "await", "coroutine", "promise", "future", "defer"]:
        if async_kw in token_set:
            push_signed_hashed_feature(vector, f"async:{async_kw}", 1.5)

    if "error" in token_set or "exception" in token_set or "panic" in token_set:
        push_signed_hashed_feature(vector, "error-handling", 1.3)
    if "try" in token_set or "catch" in token_set:
        push_signed_hashed_feature(vector, "try-catch", 1.3)
    if "result" in token_set or "ok" in token_set or "err" in token_set:
        push_signed_hashed_feature(vector, "result-type", 1.3)

    for coll_op in ["append", "insert", "push", "pop", "remove", "filter",
                    "map", "reduce", "sort", "reverse", "flatten", "merge",
                    "join", "collect", "iter", "iterate", "find", "contains",
                    "get", "set", "delete", "clear", "size", "length"]:
        if coll_op in token_set:
            push_signed_hashed_feature(vector, f"coll:{coll_op}", 1.0)

    for io_kw in ["read", "write", "open", "close", "stream", "buffer",
                  "file", "path", "url", "http", "tcp", "socket",
                  "connect", "disconnect"]:
        if io_kw in token_set:
            push_signed_hashed_feature(vector, f"io:{io_kw}", 1.0)

    for test_kw in ["test", "spec", "mock", "stub", "fixture", "setup",
                    "teardown", "assert", "expect", "should", "given",
                    "when", "then"]:
        if test_kw in token_set:
            push_signed_hashed_feature(vector, f"test-pat:{test_kw}", 1.2)
            break


def char_ngrams(token: str, min_n: int, max_n: int) -> list[str]:
    """Generate character n-grams of a token."""
    chars = list(token)
    out: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(chars) < n:
            continue
        for i in range(len(chars) - n + 1):
            out.append(''.join(chars[i:i+n]))
    return out


def token_pieces(token: str) -> list[str]:
    """Split a token into camelCase/snake_case pieces."""
    parts: list[str] = []
    current = []
    for c in token:
        if c == '_':
            if current:
                parts.append(''.join(current))
                current = []
        elif c.isupper():
            if current:
                parts.append(''.join(current))
            current = [c.lower()]
        else:
            current.append(c)
    if current:
        parts.append(''.join(current))
    return [p for p in parts if p]


def semantic_concepts(tokens: list[str]) -> list[str]:
    """Extract high-level semantic concepts from tokens."""
    concepts: list[str] = []
    token_set = set(tokens)
    if any(t in token_set for t in ["function", "method", "fn", "func"]):
        concepts.append("function")
    if any(t in token_set for t in ["class", "struct", "interface", "type"]):
        concepts.append("type")
    if any(t in token_set for t in ["test", "spec", "assert", "expect"]):
        concepts.append("test")
    if any(t in token_set for t in ["db", "database", "query", "sql"]):
        concepts.append("database")
    if any(t in token_set for t in ["http", "route", "endpoint", "api"]):
        concepts.append("api")
    if any(t in token_set for t in ["cache", "cached", "memo"]):
        concepts.append("cache")
    if any(t in token_set for t in ["log", "trace", "debug"]):
        concepts.append("logging")
    if any(t in token_set for t in ["config", "setting", "env"]):
        concepts.append("config")
    return concepts


def push_signed_hashed_feature(vector: list[float], feature: str, weight: float) -> None:
    """Push a signed hashed feature into the vector."""
    hash_val = stable_hash64(feature.encode(), 0xcbf29ce484222325)
    index = hash_val % len(vector)
    sign_hash = stable_hash64(feature.encode(), 0x9e3779b97f4a7c15)
    sign = 1.0 if sign_hash & 1 == 0 else -1.0
    vector[index] += sign * weight


def stable_hash64(data: bytes, seed: int) -> int:
    """FNV-1a 64-bit hash."""
    hash_val = seed
    for byte in data:
        hash_val ^= byte
        hash_val = (hash_val * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return hash_val


def normalize_vector(vector: list[float]) -> None:
    """L2-normalize a vector in place."""
    norm = sum(v * v for v in vector) ** 0.5
    if norm > 0.0:
        for i in range(len(vector)):
            vector[i] /= norm


def encode_embedding(vector: list[float]) -> bytes:
    """Encode a float vector to bytes (little-endian, double precision)."""
    return struct.pack(f'{len(vector)}d', *vector)


def decode_embedding(blob: bytes) -> list[float] | None:
    """Decode bytes to a float vector."""
    if len(blob) % 8 != 0:
        return None
    count = len(blob) // 8
    return list(struct.unpack(f'{count}d', blob))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(lv * rv for lv, rv in zip(left, right, strict=False))
    left_norm = sum(v * v for v in left) ** 0.5
    right_norm = sum(v * v for v in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
