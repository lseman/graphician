"""Ingest stable compiler and language-server edge evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.edge import Confidence, Edge, EdgeKind
from ..core.graph import Graph

COMPILER_EVIDENCE_VERSION = 1


@dataclass(frozen=True)
class CompilerEdgeEvidence:
    source: str
    target: str
    kind: EdgeKind
    detail: str | None = None


@dataclass(frozen=True)
class CompilerEvidenceFile:
    version: int
    provider: str
    edges: tuple[CompilerEdgeEvidence, ...] = ()


@dataclass
class CompilerEnrichmentReport:
    added: int = 0
    upgraded: int = 0
    unresolved: int = 0


def load_compiler_evidence(path: Path) -> CompilerEvidenceFile:
    """Load and validate ``.ariadne/compiler-evidence.json``."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid compiler evidence {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"invalid compiler evidence {path}: expected an object")
    version = raw.get("version")
    if version != COMPILER_EVIDENCE_VERSION:
        raise ValueError(
            f"unsupported compiler evidence version {version!r}; "
            f"expected {COMPILER_EVIDENCE_VERSION}"
        )
    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("compiler evidence provider must not be empty")
    raw_edges = raw.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ValueError("compiler evidence edges must be a list")
    edges: list[CompilerEdgeEvidence] = []
    for index, item in enumerate(raw_edges):
        if not isinstance(item, dict):
            raise ValueError(f"compiler evidence edge {index} must be an object")
        try:
            source = item["source"]
            target = item["target"]
            kind = EdgeKind(item["kind"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid compiler evidence edge {index}: {exc}") from exc
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"compiler evidence edge {index} endpoints must be strings")
        detail = item.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise ValueError(f"compiler evidence edge {index} detail must be a string")
        edges.append(CompilerEdgeEvidence(source, target, kind, detail))
    return CompilerEvidenceFile(version, provider.strip(), tuple(edges))


def apply_compiler_evidence(
    graph: Graph, evidence: CompilerEvidenceFile
) -> CompilerEnrichmentReport:
    """Add compiler edges and upgrade matching heuristic edges in place."""
    report = CompilerEnrichmentReport()
    for item in evidence.edges:
        source = graph.find_by_qname(item.source)
        target = graph.find_by_qname(item.target)
        if source is None or target is None:
            report.unresolved += 1
            continue
        existing = next(
            (
                edge
                for _, src, dst, edge in graph.edges()
                if src == source and dst == target and edge.kind == item.kind
            ),
            None,
        )
        if existing is None:
            existing = Edge.extracted(item.kind)
            _stamp_provenance(existing, evidence.provider, item.detail)
            graph.add_edge(source, target, existing)
            report.added += 1
        else:
            existing.confidence = Confidence.EXTRACTED
            _stamp_provenance(existing, evidence.provider, item.detail)
            report.upgraded += 1
    return report


def _stamp_provenance(edge: Edge, provider: str, detail: str | None) -> None:
    edge.properties["provenance"] = "compiler"
    edge.properties["provider"] = provider
    if detail is not None:
        edge.properties["evidence_detail"] = detail

