from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


KEY_METRICS = ["waf", "gc_count", "wear_avg", "wear_std", "wear_max", "free_blocks"]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def discover_run_manifests(base_dir: str) -> List[str]:
    """Find per-run manifest.json files, excluding the matrix-level manifest."""
    found: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        if "manifest.json" in files:
            found.append(os.path.join(root, "manifest.json"))
    return sorted(found)


def load_run_rows(base_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_dir = os.path.abspath(base_dir)

    for path in discover_run_manifests(base_dir):
        manifest = _read_json(path)
        metrics = dict(manifest.get("metrics", {}))
        params = dict(manifest.get("parameters", {}))

        rel_dir = os.path.relpath(os.path.dirname(path), base_dir)
        parts = rel_dir.split(os.sep)
        scenario = parts[0] if parts else rel_dir

        row = {
            "scenario": scenario,
            "run_dir": rel_dir,
            "policy": metrics.get("policy") or params.get("gc_policy"),
            "seed": metrics.get("seed") or params.get("seed"),
            "ops": metrics.get("ops") or params.get("ops"),
            "update_ratio": metrics.get("update_ratio") or params.get("update_ratio"),
            "trim_ratio": metrics.get("trim_ratio") or params.get("trim_ratio"),
            "manifest": os.path.relpath(path, base_dir),
        }
        for key in KEY_METRICS:
            row[key] = metrics.get(key)
        rows.append(row)

    return rows


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("scenario")), str(row.get("policy")))
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for (scenario, policy), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "scenario": scenario,
            "policy": policy,
            "runs": len(group),
        }
        for metric in KEY_METRICS:
            vals = [_to_float(row.get(metric)) for row in group]
            vals = [v for v in vals if v is not None]
            item[f"{metric}_mean"] = round(mean(vals), 6) if vals else ""
            item[f"{metric}_min"] = round(min(vals), 6) if vals else ""
            item[f"{metric}_max"] = round(max(vals), 6) if vals else ""
        summary.append(item)
    return summary


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_report(path: str, *, base_dir: str, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    matrix_manifest_path = os.path.join(base_dir, "matrix_manifest.json")
    matrix = _read_json(matrix_manifest_path) if os.path.exists(matrix_manifest_path) else {}

    scenarios = sorted({str(row.get("scenario")) for row in rows})
    policies = sorted({str(row.get("policy")) for row in rows})

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SSD GC Validation Report\n\n")
        f.write("## Scope\n\n")
        f.write(f"- Base directory: `{base_dir}`\n")
        f.write(f"- Matrix profile: `{matrix.get('profile', 'unknown')}`\n")
        f.write(f"- Scenarios: {', '.join(scenarios) if scenarios else 'none'}\n")
        f.write(f"- Policies: {', '.join(policies) if policies else 'none'}\n")
        f.write(f"- Runs discovered: {len(rows)}\n\n")

        f.write("## Summary By Scenario And Policy\n\n")
        cols = ["scenario", "policy", "runs", "waf_mean", "gc_count_mean", "wear_std_mean", "wear_max_mean"]
        f.write(_markdown_table(summary, cols))
        f.write("\n")

        f.write("## Run Inventory\n\n")
        inv_cols = ["scenario", "policy", "seed", "ops", "waf", "gc_count", "wear_std", "manifest"]
        f.write(_markdown_table(rows, inv_cols))
        f.write("\n")

        f.write("## Interpretation Notes\n\n")
        f.write("- WAF should remain at or above 1.0 for host-write workloads.\n")
        f.write("- Lower `wear_std` indicates more even erase distribution in this simplified model.\n")
        f.write("- Each run manifest records command, parameters, git state, Python version, and final metrics.\n")
        f.write("- These numbers are simulator-relative validation signals, not real SSD performance claims.\n")


def generate_report(base_dir: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    out_dir = out_dir or base_dir
    rows = load_run_rows(base_dir)
    summary = summarize_rows(rows)

    run_csv = os.path.join(out_dir, "validation_runs.csv")
    summary_csv = os.path.join(out_dir, "validation_summary.csv")
    report_md = os.path.join(out_dir, "validation_report.md")

    write_csv(run_csv, rows)
    write_csv(summary_csv, summary)
    write_markdown_report(report_md, base_dir=base_dir, rows=rows, summary=summary)

    return {
        "run_csv": run_csv,
        "summary_csv": summary_csv,
        "report_md": report_md,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a markdown validation report from matrix manifests.")
    parser.add_argument("--base_dir", default="results/final_clean")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    outputs = generate_report(args.base_dir, args.out_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
