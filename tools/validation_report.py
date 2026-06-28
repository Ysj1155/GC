from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import os
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


KEY_METRICS = ["waf", "gc_count", "wear_avg", "wear_std", "wear_max", "free_blocks", "trim_ops", "trim_hits", "trim_misses", "retrim_count", "trim_invalidated_pages", "trimmed_pages", "trim_gc_lag_eligible_count", "trim_gc_lag_reclaimed_count", "trim_gc_lag_pending_count", "trim_gc_reclaim_rate", "trim_gc_lag_avg", "trim_gc_lag_p95", "trim_gc_lag_max", "trim_window_count", "trim_window_avg_trim_ops", "trim_window_avg_invalid_pages_delta", "trim_window_avg_free_pages_delta", "trim_window_avg_free_blocks_delta", "trim_window_avg_gc_count_delta", "trim_window_avg_waf_delta", "trim_window_gc_window_count"]


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
            "trim_locality": params.get("trim_locality", ""),
            "phase_pattern": params.get("phase_pattern", ""),
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




def _is_locality_row(row: Dict[str, Any]) -> bool:
    locality = str(row.get("trim_locality") or "")
    scenario = str(row.get("scenario") or "")
    return locality in {"hot", "cold", "mixed"} and scenario.startswith("trim_locality_")


def summarize_locality_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if not _is_locality_row(row):
            continue
        key = (str(row.get("policy")), str(row.get("trim_locality")))
        groups.setdefault(key, []).append(row)

    metrics = [
        "waf",
        "gc_count",
        "wear_std",
        "trim_gc_lag_avg",
        "trim_gc_reclaim_rate",
        "trim_window_avg_invalid_pages_delta",
        "trim_window_avg_free_blocks_delta",
        "trim_window_avg_gc_count_delta",
        "trim_window_avg_waf_delta",
    ]
    out: List[Dict[str, Any]] = []
    for (policy, locality), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "policy": policy,
            "trim_locality": locality,
            "runs": len(group),
        }
        for metric in metrics:
            vals = [_to_float(row.get(metric)) for row in group]
            vals = [v for v in vals if v is not None]
            item[f"{metric}_mean"] = round(mean(vals), 6) if vals else ""
        out.append(item)
    return out

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
    locality_summary = summarize_locality_rows(rows)

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

        f.write("## TRIM Summary\n\n")
        trim_cols = ["scenario", "policy", "runs", "trim_ops_mean", "trim_hits_mean", "trim_misses_mean", "retrim_count_mean", "trim_invalidated_pages_mean"]
        f.write(_markdown_table(summary, trim_cols))
        f.write("\n")

        f.write("## TRIM-to-GC Lag Summary\n\n")
        lag_cols = ["scenario", "policy", "runs", "trim_gc_lag_eligible_count_mean", "trim_gc_lag_reclaimed_count_mean", "trim_gc_lag_pending_count_mean", "trim_gc_reclaim_rate_mean", "trim_gc_lag_avg_mean", "trim_gc_lag_p95_mean"]
        f.write(_markdown_table(summary, lag_cols))
        f.write("\n")

        f.write("## TRIM Window Summary\n\n")
        window_cols = ["scenario", "policy", "runs", "trim_window_count_mean", "trim_window_avg_trim_ops_mean", "trim_window_avg_invalid_pages_delta_mean", "trim_window_avg_free_blocks_delta_mean", "trim_window_avg_gc_count_delta_mean", "trim_window_avg_waf_delta_mean"]
        f.write(_markdown_table(summary, window_cols))
        f.write("\n")

        f.write("## TRIM Locality Sensitivity\n\n")
        locality_cols = ["policy", "trim_locality", "runs", "waf_mean", "gc_count_mean", "wear_std_mean", "trim_gc_lag_avg_mean", "trim_gc_reclaim_rate_mean", "trim_window_avg_invalid_pages_delta_mean", "trim_window_avg_gc_count_delta_mean"]
        f.write(_markdown_table(locality_summary, locality_cols))
        f.write("\n")

        f.write("## Run Inventory\n\n")
        inv_cols = ["scenario", "policy", "seed", "ops", "trim_locality", "waf", "gc_count", "wear_std", "trim_ops", "trim_hits", "trim_misses", "trim_gc_lag_avg", "trim_gc_lag_pending_count", "trim_window_count", "trim_window_avg_invalid_pages_delta", "trim_window_avg_gc_count_delta", "manifest"]
        f.write(_markdown_table(rows, inv_cols))
        f.write("\n")

        f.write("## Interpretation Notes\n\n")
        f.write("- WAF should remain at or above 1.0 for host-write workloads.\n")
        f.write("- Lower `wear_std` indicates more even erase distribution in this simplified model.\n")
        f.write("- Each run manifest records command, parameters, git state, Python version, and final metrics.\n")
        f.write("- TRIM invalidates logical mappings but should not increment host/device write counters.\n")
        f.write("- TRIM-heavy rows should be interpreted together with GC and wear metrics.\n")
        f.write("- TRIM-to-GC lag measures simulator steps from TRIM invalidation to later victim block erase.\n")
        f.write("- TRIM window deltas compare trace snapshots before and after grouped TRIM bursts.\n")
        f.write("- TRIM locality sensitivity compares hot, cold, and mixed delete targets under matched workload settings.\n")
        f.write("- These numbers are simulator-relative validation signals, not real SSD performance claims.\n")


def generate_report(base_dir: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    out_dir = out_dir or base_dir
    rows = load_run_rows(base_dir)
    summary = summarize_rows(rows)
    locality_summary = summarize_locality_rows(rows)

    run_csv = os.path.join(out_dir, "validation_runs.csv")
    summary_csv = os.path.join(out_dir, "validation_summary.csv")
    report_md = os.path.join(out_dir, "validation_report.md")
    locality_csv = os.path.join(out_dir, "trim_locality_sensitivity.csv")

    write_csv(run_csv, rows)
    write_csv(summary_csv, summary)
    write_csv(locality_csv, locality_summary)
    write_markdown_report(report_md, base_dir=base_dir, rows=rows, summary=summary)

    return {
        "run_csv": run_csv,
        "summary_csv": summary_csv,
        "locality_csv": locality_csv,
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
