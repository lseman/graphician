#!/usr/bin/env bash
set -euo pipefail

# install.sh — Install graphician and register it as an agent skill
#
# Usage:
#   ./install.sh              # install in current venv
#   ./install.sh --system     # install system-wide (requires sudo)
#   ./install.sh --dev        # install with dev dependencies
#   ./install.sh --uninstall  # remove the skill (keeps package)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Options ──────────────────────────────────────────────────────────────────
UNINSTALL=false
WITH_DEV=false
WITH_SYSTEM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --uninstall) UNINSTALL=true; shift ;;
        --dev)       WITH_DEV=true; shift ;;
        --system)    WITH_SYSTEM=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--uninstall] [--dev] [--system]"
            echo ""
            echo "  --uninstall   Remove agent skill (keeps Python package)"
            echo "  --dev         Install dev dependencies too"
            echo "  --system      Install system-wide (requires sudo)"
            exit 0
            ;;
        *) error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Uninstall ────────────────────────────────────────────────────────────────
if $UNINSTALL; then
    SKILL_DIR="${HOME}/.agents/skills/graphician"
    if [[ -d "$SKILL_DIR" ]]; then
        rm -rf "$SKILL_DIR"
        info "Removed agent skill at $SKILL_DIR"
    else
        warn "Skill directory not found: $SKILL_DIR"
    fi
    # Also clean up the local symlink if it exists
    if [[ -L "${SCRIPT_DIR}/.agents" ]]; then
        rm "${SCRIPT_DIR}/.agents"
        info "Removed local .agents symlink"
    fi
    exit 0
fi

# ── Install Python package ───────────────────────────────────────────────────
info "Installing graphician Python package..."

if $WITH_DEV; then
    PIP_ARGS="-e .[dev]"
elif $WITH_SYSTEM; then
    PIP_ARGS="-e . --break-system-packages"
    info "Installing system-wide (requires sudo for some steps)"
    if [[ $EUID -ne 0 ]]; then
        warn "Not running as root. Use 'sudo ./install.sh --system' for true system-wide install."
    fi
else
    PIP_ARGS="-e ."
fi

python -m pip install $PIP_ARGS 2>&1 | tail -3 || {
    # Handle PEP 668: externally managed environment
    if [[ $EUID -eq 0 ]] || [[ "${VIRTUAL_ENV:-}" != "" ]]; then
        # Running as root or in venv - try with --break-system-packages
        warn "Retrying with --break-system-packages..."
        python -m pip install $PIP_ARGS --break-system-packages 2>&1 | tail -3
    else
        error "PEP 668: externally managed environment. Use --system or activate a venv first."
        exit 1
    fi
}

if ! python -c "import graphician" 2>/dev/null; then
    error "Failed to install graphician. Check the output above."
    exit 1
fi
info "graphician installed successfully"

# ── Create agent skill ───────────────────────────────────────────────────────
SKILL_DIR="${HOME}/.agents/skills/graphician"
mkdir -p "$SKILL_DIR"

# Clean up any old skill files from previous installs
rm -f "$SKILL_DIR/skill.md" "$SKILL_DIR/skill-graphician.md"

info "Writing agent skill to $SKILL_DIR"

cat > "$SKILL_DIR/SKILL.md" << 'SKILL_EOF'
---
name: graphician
description: "Use for codebase navigation, impact analysis, and reasoning about code structure. Builds a persistent knowledge graph from source code with community detection, fuzzy search, and path traversal."
---

# /graphician

Transform any codebase into a navigable knowledge graph with community detection, fuzzy search, and dependency analysis.

## Usage

```bash
# Build graph from current directory
graphician build .

# Build with specific languages
graphician build . --python --typescript

# Query the graph
graphician query "How does authentication work?"

# Find impact of changes
graphician impact src/auth.py

# Show dependencies
graphician deps src/main.py

# Build and serve for interactive exploration
graphician serve .

# Build and show graph visualization
graphician build . --viz

# Incremental update
graphician build . --update
```

## What graphician is for

Drop any codebase and get a queryable knowledge graph. Persistent across sessions, with community detection that surfaces cross-file connections you wouldn't think to ask about.

## What You Must Do When Invoked

If the user invoked `/graphician --help` or `/graphician -h` (with no other arguments), print the contents of the `## Usage` section above verbatim and stop. Do not run any commands, do not detect files, do not default the path to `.`. Just print the Usage block and return.

If no path was given, use `.` (current directory). Do not ask the user for a path.

Follow these steps in order. Do not skip steps.

### Step 1 - Ensure graphician is installed

```bash
# Detect the correct Python interpreter
PYTHON=""
GRAPHICIAN_BIN=$(which graphician 2>/dev/null)
if [ -n "$GRAPHICIAN_BIN" ]; then
    _SHEBANG=$(head -1 "$GRAPHICIAN_BIN" | tr -d '#!')
    case "$_SHEBANG" in
        *[!a-zA-Z0-9/_.@-]*) ;;
        *) "$_SHEBANG" -c "import graphician" 2>/dev/null && PYTHON="$_SHEBANG" ;;
    esac
fi
if [ -z "$PYTHON" ]; then PYTHON="python3"; fi
if ! "$PYTHON" -c "import graphician" 2>/dev/null; then
    if command -v uv >/dev/null 2>&1; then
        uv tool install --upgrade graphician -q 2>&1 | tail -3
    else
        "$PYTHON" -m pip install graphician -q 2>/dev/null \
          || "$PYTHON" -m pip install graphician -q --break-system-packages 2>&1 | tail -3
    fi
fi
```

If the import succeeds, print nothing and move to Step 2.

### Step 2 - Build the graph

```bash
$PYTHON -m graphician build "$(pwd)"
```

Wait for the build to complete. The graph is stored in `graphician.db` in the current directory.

### Step 3 - Query the graph

Run queries against the graph:

```bash
$PYTHON -m graphician query "<your question>"
```

For specific analysis tasks:
```bash
# Impact analysis
$PYTHON -m graphician impact src/auth.py

# Dependency analysis
$PYTHON -m graphician deps src/main.py

# Show call graph
$PYTHON -m graphician call src/auth.py --depth 2
```

## Tips

- For large codebases, use `--update` for incremental builds
- Use `--python --typescript --rust` to limit which languages to extract
- The graph persists in `graphician.db` — no need to rebuild for queries
- Use `graphician serve .` for an interactive web interface
SKILL_EOF

# ── Create local symlink (optional) ──────────────────────────────────────────
# Create a symlink in the project root for easy access
if [[ ! -L "${SCRIPT_DIR}/.agents" ]]; then
    ln -sf "${HOME}/.agents" "${SCRIPT_DIR}/.agents"
    info "Created local .agents symlink in project root"
fi

info "Done! graphician is installed and registered as an agent skill."
info "Skill location: $SKILL_DIR/SKILL.md"
