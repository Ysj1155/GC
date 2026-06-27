from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import os
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.validation_report import load_run_rows, write_csv


LOWER_IS_BETTER = {"waf", "gc_count", "wear_std", "wear_max"}
HIGHER_IS_BETTER = {"free_blocks"}
KEY_METRICS = ["waf", "gc_count", "wear_std", "wear_max", "free_blocks"]


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_mean(values: Sequence[float]) -> str:
    return f"{mean(values):.6f}" if values else ""


def _safe_min(values: Sequence[float]) -> str:
    return f"{min(values):.6f}" if values else ""


def _safe_max(values: Sequence[float]) -> str:
    return f"{max(values):.6f}" if values else ""


def _value(row: Dict[str, Any], metric: str) -> Optional[float]:
    return _to_float(row.get(metric))


def _row_label(row: Dict[str, Any]) -> str:
    parts = [
        str(row.get("scenario", "unknown")),
        str(row.get("policy", row.get("gc_policy", "unknown"))),
        f"seed{row.get('seed', '')}",
    ]
    return "/".join(parts)


def load_experiment_rows(base_dir: Optional[str] = None, input_csv: Optional[str] = None) -> List[Dict[str, Any]]:
    if input_csv:
        return _read_csv(input_csv)
    if base_dir:
        validation_csv = os.path.join(base_dir, "validation_runs.csv")
        if os.path.exists(validation_csv):
            return _read_csv(validation_csv)
        return load_run_rows(base_dir)
    raise ValueError("Either base_dir or input_csv is required.")


def build_policy_scorecard(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        scenario = str(row.get("scenario", "all"))
        policy = str(row.get("policy", row.get("gc_policy", "unknown")))
        groups.setdefault((scenario, policy), []).append(row)

    out: List[Dict[str, Any]] = []
    for (scenario, policy), group in sorted(groups.items()):
        item: Dict[str, Any] = {
            "scenario": scenario,
            "policy": policy,
            "runs": len(group),
        }
        for metric in KEY_METRICS:
            values = [_value(row, metric) for row in group]
            values = [v for v in values if v is not None]
            item[f"{metric}_mean"] = _safe_mean(values)
            item[f"{metric}_min"] = _safe_min(values)
            item[f"{metric}_max"] = _safe_max(values)
        out.append(item)

    _attach_metric_ranks(out)
    return out


def _attach_metric_ranks(rows: List[Dict[str, Any]]) -> None:
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row.get("scenario", "all")), []).append(row)

    for group in by_scenario.values():
        for metric in KEY_METRICS:
            key = f"{metric}_mean"
            values = []
            for row in group:
                val = _to_float(row.get(key))
                if val is not None:
                    values.append((val, row))
            reverse = metric in HIGHER_IS_BETTER
            values.sort(key=lambda item: item[0], reverse=reverse)
            for rank, (_val, row) in enumerate(values, start=1):
                row[f"{metric}_rank"] = rank


def find_anomaly_runs(rows: Sequence[Dict[str, Any]], z_threshold: float = 1.0) -> List[Dict[str, Any]]:
    baselines: Dict[str, tuple[float, float]] = {}
    for metric in KEY_METRICS:
        values = [_value(row, metric) for row in rows]
        values = [v for v in values if v is not None]
        if len(values) < 2:
            continue
        baselines[metric] = (mean(values), pstdev(values) or 0.0)

    out: List[Dict[str, Any]] = []
    for row in rows:
        reasons: List[str] = []
        for metric, (mu, sigma) in baselines.items():
            val = _value(row, metric)
            if val is None or sigma == 0.0:
                continue
            z = (val - mu) / sigma
            if metric in LOWER_IS_BETTER and z >= z_threshold:
                reasons.append(f"high_{metric}:z={z:.2f}")
            if metric in HIGHER_IS_BETTER and z <= -z_threshold:
                reasons.append(f"low_{metric}:z={z:.2f}")
        if reasons:
            item = dict(row)
            item["run_label"] = _row_label(row)
            item["anomaly_reasons"] = "; ".join(reasons)
            out.append(item)
    return out


def pareto_front(rows: Sequence[Dict[str, Any]], metrics: Sequence[str] = ("waf", "wear_std", "gc_count")) -> List[Dict[str, Any]]:
    candidates = [row for row in rows if all(_value(row, metric) is not None for metric in metrics)]
    out: List[Dict[str, Any]] = []
    for row in candidates:
        dominated = False
        row_vals = [_value(row, metric) for metric in metrics]
        assert all(v is not None for v in row_vals)
        for other in candidates:
            if other is row:
                continue
            other_vals = [_value(other, metric) for metric in metrics]
            assert all(v is not None for v in other_vals)
            no_worse = all(float(o) <= float(r) for o, r in zip(other_vals, row_vals))
            strictly_better = any(float(o) < float(r) for o, r in zip(other_vals, row_vals))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            item = dict(row)
            item["pareto_metrics"] = ",".join(metrics)
            out.append(item)
    return out


def recommend_next_sweeps(scorecard: Sequence[Dict[str, Any]], anomalies: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in scorecard:
        policy = str(row.get("policy", "unknown"))
        scenario = str(row.get("scenario", "all"))
        waf_rank = _to_float(row.get("waf_rank"))
        wear_rank = _to_float(row.get("wear_std_rank"))
        runs = int(float(row.get("runs") or 0))
        if runs < 3:
            out.append({
                "scenario": scenario,
                "policy": policy,
                "priority": "medium",
                "recommendation": "Repeat with at least three seeds before trusting the ranking.",
                "reason": f"Only {runs} run(s) found.",
            })
        if waf_rank is not None and wear_rank is not None and abs(waf_rank - wear_rank) >= 2:
            out.append({
                "scenario": scenario,
                "policy": policy,
                "priority": "high",
                "recommendation": "Run a trade-off sweep that changes wear pressure and reclaim pressure separately.",
                "reason": f"WAF rank {int(waf_rank)} and wear_std rank {int(wear_rank)} disagree.",
            })

    for row in anomalies[:10]:
        out.append({
            "scenario": row.get("scenario", "unknown"),
            "policy": row.get("policy", row.get("gc_policy", "unknown")),
            "priority": "high",
            "recommendation": "Re-run the anomalous condition with trace and GC event logging enabled.",
            "reason": row.get("anomaly_reasons", ""),
        })
    return out


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_insights_markdown(
    path: str,
    rows: Sequence[Dict[str, Any]],
    scorecard: Sequence[Dict[str, Any]],
    anomalies: Sequence[Dict[str, Any]],
    pareto: Sequence[Dict[str, Any]],
    recommendations: Sequence[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    best_waf = sorted(
        [row for row in scorecard if _to_float(row.get("waf_mean")) is not None],
        key=lambda row: float(row["waf_mean"]),
    )[:5]
    best_wear = sorted(
        [row for row in scorecard if _to_float(row.get("wear_std_mean")) is not None],
        key=lambda row: float(row["wear_std_mean"]),
    )[:5]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# SSD GC Experiment Insights\n\n")
        f.write("## Scope\n\n")
        f.write(f"- Runs analyzed: {len(rows)}\n")
        f.write(f"- Scorecard rows: {len(scorecard)}\n")
        f.write(f"- Anomalies flagged: {len(anomalies)}\n")
        f.write(f"- Pareto-front runs: {len(pareto)}\n\n")
        f.write("## Best Mean WAF\n\n")
        f.write(_markdown_table(best_waf, ["scenario", "policy", "runs", "waf_mean", "wear_std_mean", "gc_count_mean"]))
        f.write("\n## Best Mean Wear Balance\n\n")
        f.write(_markdown_table(best_wear, ["scenario", "policy", "runs", "wear_std_mean", "waf_mean", "gc_count_mean"]))
        f.write("\n## Top Anomaly Runs\n\n")
        f.write(_markdown_table(anomalies[:10], ["scenario", "policy", "seed", "waf", "wear_std", "gc_count", "anomaly_reasons"]))
        f.write("\n## Recommended Next Sweeps\n\n")
        f.write(_markdown_table(recommendations[:12], ["priority", "scenario", "policy", "recommendation", "reason"]))
        f.write("\n## Interpretation Guardrails\n\n")
        f.write("- Lower WAF means less internal write amplification in this simulator.\n")
        f.write("- Lower wear_std means erase counts are more evenly distributed.\n")
        f.write("- A Pareto-front run is not globally best; it is a trade-off point not dominated on WAF, wear_std, and gc_count.\n")
        f.write("- Any AI narrative should cite these CSV outputs instead of inventing causal claims.\n")


def run_insight_miner(base_dir: Optional[str], input_csv: Optional[str], out_dir: str, z_threshold: float = 1.0) -> Dict[str, str]:
    rows = load_experiment_rows(base_dir=base_dir, input_csv=input_csv)
    os.makedirs(out_dir, exist_ok=True)

    scorecard = build_policy_scorecard(rows)
    anomalies = find_anomaly_runs(rows, z_threshold=z_threshold)
    front = pareto_front(rows)
    recommendations = recommend_next_sweeps(scorecard, anomalies)

    outputs = {
        "policy_scorecard": os.path.join(out_dir, "policy_scorecard.csv"),
        "anomaly_runs": os.path.join(out_dir, "anomaly_runs.csv"),
        "pareto_front": os.path.join(out_dir, "pareto_front.csv"),
        "recommended_next_sweeps": os.path.join(out_dir, "recommended_next_sweeps.csv"),
        "insights": os.path.join(out_dir, "insights.md"),
    }
    write_csv(outputs["policy_scorecard"], scorecard)
    write_csv(outputs["anomaly_runs"], anomalies)
    write_csv(outputs["pareto_front"], front)
    write_csv(outputs["recommended_next_sweeps"], recommendations)
    write_insights_markdown(outputs["insights"], rows, scorecard, anomalies, front, recommendations)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine SSD GC experiment results for policy trade-offs and anomalies.")
    parser.add_argument("--base_dir", default=None, help="Matrix/report directory containing manifests or validation_runs.csv.")
    parser.add_argument("--input_csv", default=None, help="Existing summary or validation_runs CSV.")
    parser.add_argument("--out_dir", default="results/insights")
    parser.add_argument("--z_threshold", type=float, default=1.0)
    args = parser.parse_args()

    outputs = run_insight_miner(args.base_dir, args.input_csv, args.out_dir, args.z_threshold)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
