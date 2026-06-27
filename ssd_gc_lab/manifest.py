from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional


def _json_safe(value: Any) -> Any:
    """Convert common Python values to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _run_git(args: Iterable[str], cwd: str) -> Optional[str]:
    safe_cwd = cwd.replace("\\", "/")
    try:
        out = subprocess.check_output(
            ["git", "-c", f"safe.directory={safe_cwd}", *args],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        )
        return out.strip()
    except Exception:
        return None


def collect_git_state(cwd: str) -> Dict[str, Any]:
    """Collect lightweight git provenance without failing outside a git repo."""
    cwd = os.path.abspath(cwd)
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    branch = _run_git(["branch", "--show-current"], cwd)
    status = _run_git(["status", "--short"], cwd)

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def build_run_manifest(
    *,
    args: Any,
    metrics_row: Mapping[str, Any],
    out_dir: str,
    artifacts: Optional[Mapping[str, Any]] = None,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a reproducibility manifest for one simulation run."""
    cwd = os.path.abspath(cwd or os.getcwd())
    out_dir = os.path.abspath(out_dir)

    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    command = [os.path.basename(sys.executable), "run_sim.py", *sys.argv[1:]]

    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "working_directory": cwd,
        "output_directory": out_dir,
        "command": command,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
        },
        "git": collect_git_state(cwd),
        "parameters": _json_safe(args_dict),
        "metrics": _json_safe(dict(metrics_row)),
        "artifacts": _json_safe(dict(artifacts or {})),
    }


def write_manifest(path: str, manifest: Mapping[str, Any]) -> None:
    """Write a manifest JSON file with stable formatting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(dict(manifest)), f, indent=2, sort_keys=True)
        f.write("\n")
