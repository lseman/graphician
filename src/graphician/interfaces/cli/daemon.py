"""Daemon and git hooks installation for Graphician.

Provides:
- ``cmd_daemon``: multi-repo watch daemon (add/start/status)
- ``cmd_install``: install auto-update git hooks and MCP configs
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Config paths
_DAEMON_CONFIG_DIR = Path.home() / ".graphician"
_DAEMON_REPOS_FILE = _DAEMON_CONFIG_DIR / "repos.json"


def cmd_daemon(command: str, db_path: str, path: str = "", alias: str = "",
               interval: int = 30) -> dict[str, Any]:
    """Execute a daemon command.

    Args:
        command: One of "add", "start", "status".
        db_path: Path to the Graphician SQLite database.
        path: Repository path (required for "add" and "start").
        alias: Repository alias (optional for "add").
        interval: Poll interval in seconds (for "start").

    Returns:
        Result dict.
    """
    if command == "add":
        return _daemon_add(db_path, path, alias)
    elif command == "status":
        return _daemon_status()
    elif command == "start":
        return _daemon_start(db_path, path or str(Path.cwd()), interval)
    else:
        return {"error": f"Unknown daemon command: {command}"}


def cmd_install(db_path: str, repo: str, force: bool = False,
                agents: bool = False, mcp: bool = False) -> dict[str, Any]:
    """Install auto-update git hooks and optional configs.

    Args:
        db_path: Path to the Graphician SQLite database.
        repo: Repository root path.
        force: Replace existing hooks.
        agents: Install AGENTS.md with Graphician instructions.
        mcp: Install MCP configs for Claude/Cursor/VSCode/Codex.

    Returns:
        Summary dict.
    """
    repo_path = Path(repo).resolve()
    git_dir = repo_path / ".git"

    if not git_dir.is_dir():
        return {"error": f"{repo_path} is not a git repository"}

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    db_path = Path(db_path).resolve()

    result: dict[str, Any] = {"hooks_installed": [], "configs_installed": []}

    # Install git hooks
    for hook_name in ["post-commit", "post-merge", "post-checkout"]:
        hook_path = hooks_dir / hook_name
        if hook_path.exists() and not force:
            return {
                "error": f"{hook_path} already exists; use --force to replace",
                "hooks_installed": result["hooks_installed"],
            }

        script = (
            f'#!/bin/sh\n'
            f'"{sys.executable}" -m graphician --db "{db_path}" update '
            f'"{repo_path}" >/dev/null 2>&1 || true\n'
        )
        hook_path.write_text(script, encoding="utf-8")
        hook_path.chmod(0o755)
        result["hooks_installed"].append(str(hook_path))

    # Install AGENTS.md
    if agents:
        agents_path = repo_path / "AGENTS.md"
        block = (
            f"# Graphician Agent Instructions\n\n"
            f"- Start exploration with `graphician --db {db_path} tool "
            f'minimal_context --params \'{{"target":"...","mode":"review"}}\'`.\n'
            f"- For code review, run `graphician --db {db_path} tool "
            f'detect_changes --params \'{{"base":"HEAD~1"}}\'` before reading files.\n'
            f"- Use `impact`, `traverse`, and `review_context` to gather bounded "
            f"context before broad grep/read.\n"
            f"- Use `gaps`, `bridge_nodes`, and `large_functions` to find risky "
            f"areas and review questions.\n"
            f"- Fall back to direct file reads only after Graphician identifies the "
            f"relevant files or symbols.\n"
        )
        agents_path.write_text(block, encoding="utf-8")
        result["configs_installed"].append(str(agents_path))

    # Install MCP configs
    if mcp:
        _install_mcp_configs(repo_path, db_path, result)

    return result


# ── Daemon implementation ──────────────────────────────────────────


def _daemon_add(db_path: str, path: str, alias: str) -> dict[str, Any]:
    """Register a repository with the daemon."""
    repo_path = Path(path).resolve()
    if not repo_path.exists():
        return {"error": f"Path does not exist: {repo_path}"}

    if not alias:
        alias = repo_path.name or "repo"

    repos = _load_daemon_repos()
    repos.append({"alias": alias, "path": str(repo_path)})
    _save_daemon_repos(repos)

    return {"registered": str(repo_path), "alias": alias}


def _daemon_status() -> dict[str, Any]:
    """Show registered repositories."""
    return {"repos": _load_daemon_repos()}


def _daemon_start(db_path: str, path: str, interval: int) -> dict[str, Any]:
    """Start the daemon (returns config, caller runs the loop).

    Returns the list of repos and poll interval for the caller to
    execute the watch loop.
    """
    repos = _load_daemon_repos()
    if not repos:
        return {"error": "No repositories registered; run 'daemon add <path>' first"}

    paths = [r["path"] for r in repos]
    return {
        "repos": paths,
        "db_path": db_path,
        "interval": max(interval, 1),
        "repo_count": len(repos),
    }


def _load_daemon_repos() -> list[dict[str, str]]:
    """Load registered daemon repos from config file."""
    if _DAEMON_REPOS_FILE.exists():
        try:
            return json.loads(_DAEMON_REPOS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_daemon_repos(repos: list[dict[str, str]]) -> None:
    """Save registered daemon repos to config file."""
    _DAEMON_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _DAEMON_REPOS_FILE.write_text(json.dumps(repos, indent=2), encoding="utf-8")


# ── MCP config installation ────────────────────────────────────────


def _install_mcp_configs(
    repo_path: Path,
    db_path: Path,
    result: dict[str, Any],
) -> None:
    """Install MCP configuration files for various agents."""
    # Get the Python executable
    exe = sys.executable

    mcp_config = _mcp_servers_config(exe, db_path)
    config_json = json.dumps(mcp_config, indent=2)

    # Claude Code
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(exist_ok=True)
    claude_path = claude_dir / "CLAUDE.md"
    if not claude_path.exists():
        claude_path.write_text(
            f"# Graphician MCP\n\n"
            f"Use the graphician MCP tool for codebase exploration.\n"
            f"Database: {db_path}\n",
            encoding="utf-8",
        )
        result["configs_installed"].append(str(claude_path))

    # Cursor
    cursor_dir = repo_path / ".cursor"
    cursor_dir.mkdir(exist_ok=True)
    cursor_path = cursor_dir / "mcp.json"
    cursor_path.write_text(config_json, encoding="utf-8")
    result["configs_installed"].append(str(cursor_path))

    # VS Code
    vscode_dir = repo_path / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    vscode_path = vscode_dir / "mcp.json"
    vscode_path.write_text(config_json, encoding="utf-8")
    result["configs_installed"].append(str(vscode_path))

    # Codex
    codex_dir = repo_path / ".codex"
    codex_dir.mkdir(exist_ok=True)
    codex_path = codex_dir / "graphician-mcp.toml"
    toml_content = (
        f'[command.graphician-mcp]\n'
        f'command = "{exe}"\n'
        f'args = ["-m", "graphician", "--db", "{db_path}", "mcp-server"]\n'
        f'description = "Graphician code graph MCP server"\n'
    )
    codex_path.write_text(toml_content, encoding="utf-8")
    result["configs_installed"].append(str(codex_path))


def _mcp_servers_config(exe: str, db_path: Path) -> dict[str, Any]:
    """Generate MCP server configuration for Cursor/VSCode."""
    return {
        "mcpServers": {
            "graphician": {
                "command": exe,
                "args": [
                    "-m", "graphician",
                    "--db", str(db_path),
                    "mcp-server",
                ],
                "env": {},
            }
        }
    }
