"""Python script generation for Jedi-based call resolution.

Builds a standalone Python script that the host process executes as a
subprocess. The script uses Jedi to resolve dropped method calls and
prints JSON results back to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_jedi_script(
    pending: list[tuple[str, int, int, str, str]],
    repo_root: Path,
) -> str:
    """Build a Python script that uses Jedi to resolve pending method calls.

    Args:
        pending: List of (file_path, jedi_line, col, method_name, enclosing_qname)
            tuples from the scan pass.
        repo_root: Project root directory.

    Returns:
        A Python script string that can be executed via ``python3 -c``.
    """
    json_calls = json.dumps(pending)
    repo_root_str = str(repo_root)

    return f'''import jedi, json, sys
from pathlib import Path

repo_root = Path(r"{repo_root_str}")

py_dirs = sorted(set(Path(p).parent for p, _, _, _, _ in json.loads(r"{json_calls}")))
if py_dirs:
    common_root = Path.commonpath(str(p) for p in py_dirs)
else:
    common_root = repo_root

project = jedi.Project(
    path=str(common_root),
    added_sys_path=[str(repo_root)],
    smart_sys_path=False,
)

def resolve_via_goto(script, line, col):
    try:
        names = script.goto(line, col)
        return names[0] if names else None
    except Exception:
        return None

def resolve_via_infer(script, line, col):
    try:
        names = script.infer(line, col)
        return names[0] if names else None
    except Exception:
        return None

results = []
for file_path, jedi_line, col, method_name, enclosing in json.loads(r"{json_calls}"):
    try:
        with open(file_path, "r", errors="replace") as f:
            source = f.read()
        script = jedi.Script(source, path=file_path, project=project)

        # For getattr-style calls, try infer first to resolve the object type.
        is_getattr = method_name.startswith("getattr_variable(")
        if is_getattr:
            names = resolve_via_infer(script, jedi_line, col)
            if names and names.module_path:
                module_path = Path(names.module_path).resolve()
                try:
                    module_path.relative_to(repo_root)
                except ValueError:
                    continue
                parent = names.parent()
                if parent and parent.type == "class":
                    target = f"{{module_path}}::{{parent.name}}.{{method_name}}"
                else:
                    target = f"{{module_path}}::{{name}}"
                results.append([file_path, jedi_line, enclosing, target])
            continue

        # Normal attribute call or getattr with string method name.
        name = resolve_via_goto(script, jedi_line, col)
        if not name or not name.module_path:
            name = resolve_via_infer(script, jedi_line, col)

        if not name or not name.module_path:
            continue

        module_path = Path(name.module_path).resolve()
        try:
            module_path.relative_to(repo_root)
        except ValueError:
            continue

        parent = name.parent()
        if parent and parent.type == "class":
            target = f"{{module_path}}::{{parent.name}}.{{name.name}}"
        else:
            target = f"{{module_path}}::{{name.name}}"
        results.append([file_path, jedi_line, enclosing, target])
    except Exception:
        continue

print(json.dumps(results))
'''
