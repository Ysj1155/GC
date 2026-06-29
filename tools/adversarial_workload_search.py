from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import os
from typing import Any, Dict, List, Optional, Sequence

from ssd_gc_lab.experiment_runner import run_single_experiment
from tools.validation_report import write_csv


SEARCH_FIELDS = [
    "update_ratio",
    "hot_ratio",
    "hot_weight",
    "user_capacity_ratio",
    "warmup_fill",
    "trim_ratio",
    "bg_gc_every",
    "gc_free_block_threshold",
    "burst_length",
    "burst_ratio",
    "phase_pattern",
    "trim_locality",
    "trim_burst_length",
    "trim_burst_interval",
]


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else 0.0


def build_workload_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    fields: Dict[str, List[Any]] = {
        "update_ratio": _parse_float_list(args.update_ratios),
        "hot_ratio": _parse_float_list(args.hot_ratios),
        "hot_weight": _parse_float_list(args.hot_weights),
        "user_capacity_ratio": _parse_float_list(args.user_capacity_ratios),
        "warmup_fill": _parse_float_list(args.warmup_fills),
        "trim_ratio": _parse_float_list(args.trim_ratios),
        "bg_gc_every": _parse_int_list(args.bg_gc_everys),
        "gc_free_block_threshold": _parse_float_list(args.gc_free_block_thresholds),
        "burst_length": _parse_int_list(args.burst_lengths),
        "burst_ratio": _parse_float_list(args.burst_ratios),
        "phase_pattern": _parse_str_list(args.phase_patterns),
        "trim_locality": _parse_str_list(args.trim_localities),
        "trim_burst_length": _parse_int_list(args.trim_burst_lengths),
        "trim_burst_interval": _parse_int_list(args.trim_burst_intervals),
    }
    keys = list(fields)
    return [dict(zip(keys, values)) for values in itertools.product(*(fields[key] for key in keys))]


def _namespace(args: argparse.Namespace, policy: str, seed: int, workload: Dict[str, Any]) -> argparse.Namespace:
    trim_ratio = float(workload["trim_ratio"])
    trim_burst_length = int(workload["trim_burst_length"])
    trim_burst_interval = int(workload["trim_burst_interval"])
    return argparse.Namespace(
        ops=args.ops,
        update_ratio=float(workload["update_ratio"]),
        hot_ratio=float(workload["hot_ratio"]),
        hot_weight=float(workload["hot_weight"]),
        blocks=args.blocks,
        pages_per_block=args.pages_per_block,
        gc_free_block_threshold=float(workload["gc_free_block_threshold"]),
        user_capacity_ratio=float(workload["user_capacity_ratio"]),
        seed=seed,
        bg_gc_every=int(workload["bg_gc_every"]),
        enable_trim=(trim_ratio > 0.0 or trim_burst_length > 0 or trim_burst_interval > 0),
        trim_ratio=trim_ratio,
        warmup_fill=float(workload["warmup_fill"]),
        burst_length=int(workload["burst_length"]),
        burst_ratio=float(workload["burst_ratio"]),
        phase_pattern=str(workload["phase_pattern"]),
        trim_locality=str(workload["trim_locality"]),
        trim_burst_length=trim_burst_length,
        trim_burst_interval=trim_burst_interval,
        gc_policy=policy,
        cota_alpha=None,
        cota_beta=None,
        cota_gamma=None,
        cota_delta=None,
        cold_victim_bias=1.0,
        trim_age_bonus=0.0,
        victim_prefetch_k=1,
        age_stale_K=50.0,
        atcb_alpha=0.5,
        atcb_beta=0.3,
        atcb_gamma=0.1,
        atcb_eta=0.1,
        re50315_K=1.0,
        note=f"{policy}_adv_seed{seed}",
    )


def _objective_score(row: Dict[str, Any], objective: str, baseline: Optional[Dict[str, Any]] = None) -> float:
    if objective == "waf":
        return _metric(row, "waf")
    if objective == "wear_std":
        return _metric(row, "wear_std")
    if objective == "gc_count":
        return _metric(row, "gc_count")
    if objective == "waf_gap" and baseline is not None:
        return _metric(row, "waf") - _metric(baseline, "waf")
    if objective == "wear_gap" and baseline is not None:
        return _metric(row, "wear_std") - _metric(baseline, "wear_std")
    raise ValueError(f"Unsupported objective: {objective}")


def _workload_id(workload: Dict[str, Any]) -> str:
    return "_".join(f"{key}{workload[key]}" for key in SEARCH_FIELDS)


def run_search(args: argparse.Namespace) -> Dict[str, str]:
    os.makedirs(args.out_dir, exist_ok=True)
    workloads = build_workload_grid(args)
    seeds = _parse_int_list(args.seeds)
    rows: List[Dict[str, Any]] = []

    for workload in workloads:
        for seed in seeds:
            target_args = _namespace(args, args.policy, seed, workload)
            _sim, _meta, target_row = run_single_experiment(target_args, enable_trace=False)
            baseline_row = None
            if args.baseline_policy:
                base_args = _namespace(args, args.baseline_policy, seed, workload)
                _base_sim, _base_meta, baseline_row = run_single_experiment(base_args, enable_trace=False)

            item = dict(target_row)
            item.update(workload)
            item["workload_id"] = _workload_id(workload)
            item["target_policy"] = args.policy
            item["baseline_policy"] = args.baseline_policy or ""
            if baseline_row is not None:
                item["baseline_waf"] = baseline_row.get("waf")
                item["baseline_wear_std"] = baseline_row.get("wear_std")
                item["baseline_gc_count"] = baseline_row.get("gc_count")
                item["waf_gap"] = _metric(target_row, "waf") - _metric(baseline_row, "waf")
                item["wear_gap"] = _metric(target_row, "wear_std") - _metric(baseline_row, "wear_std")
            item["objective"] = args.objective
            item["objective_score"] = round(_objective_score(target_row, args.objective, baseline_row), 6)
            rows.append(item)

    ranked = sorted(rows, key=lambda row: float(row.get("objective_score") or 0.0), reverse=True)
    top = ranked[: args.top_k]

    outputs = {
        "adversarial_runs": os.path.join(args.out_dir, "adversarial_runs.csv"),
        "adversarial_top": os.path.join(args.out_dir, "adversarial_top.csv"),
        "report": os.path.join(args.out_dir, "adversarial_report.md"),
    }
    write_csv(outputs["adversarial_runs"], ranked)
    write_csv(outputs["adversarial_top"], top)
    _write_report(outputs["report"], args, ranked, top)
    return outputs


def _write_report(path: str, args: argparse.Namespace, rows: Sequence[Dict[str, Any]], top: Sequence[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Adversarial Workload Search Report\n\n")
        f.write(f"- Target policy: `{args.policy}`\n")
        f.write(f"- Baseline policy: `{args.baseline_policy or 'none'}`\n")
        f.write(f"- Objective: `{args.objective}`\n")
        f.write(f"- Runs evaluated: {len(rows)}\n\n")
        columns = [
            "rank", "objective_score", "seed", "update_ratio", "hot_ratio", "hot_weight",
            "user_capacity_ratio", "warmup_fill", "trim_ratio", "bg_gc_every",
            "gc_free_block_threshold", "burst_length", "burst_ratio", "phase_pattern",
            "trim_locality", "trim_burst_length", "trim_burst_interval", "waf", "wear_std", "gc_count",
        ]
        f.write("## Highest-Stress Conditions\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for rank, row in enumerate(top, start=1):
            display = dict(row)
            display["rank"] = rank
            f.write("| " + " | ".join(str(display.get(col, "")) for col in columns) + " |\n")
        f.write("\n## How To Use This\n\n")
        f.write("- Treat the top rows as stress candidates, not as final claims.\n")
        f.write("- Re-run the top conditions with more seeds and per-GC event logs.\n")
        f.write("- If a baseline was used, inspect gap columns before calling a policy weak or strong.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Search for workload conditions that stress an SSD GC policy.")
    parser.add_argument("--policy", default="cota", choices=["greedy", "cb", "cost_benefit", "age_stale", "bsgc", "cota", "atcb", "re50315"])
    parser.add_argument("--baseline_policy", default="greedy", choices=["", "greedy", "cb", "cost_benefit", "age_stale", "bsgc", "cota", "atcb", "re50315"])
    parser.add_argument("--objective", default="waf", choices=["waf", "wear_std", "gc_count", "waf_gap", "wear_gap"])
    parser.add_argument("--seeds", default="41")
    parser.add_argument("--ops", type=int, default=12000)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--pages_per_block", type=int, default=32)
    parser.add_argument("--update_ratios", default="0.7,0.9")
    parser.add_argument("--hot_ratios", default="0.1,0.2")
    parser.add_argument("--hot_weights", default="0.6,0.85")
    parser.add_argument("--user_capacity_ratios", default="0.9,0.95")
    parser.add_argument("--warmup_fills", default="0.5,0.75")
    parser.add_argument("--trim_ratios", default="0.0,0.1")
    parser.add_argument("--bg_gc_everys", default="0,64")
    parser.add_argument("--gc_free_block_thresholds", default="0.08,0.12")
    parser.add_argument("--burst_lengths", default="0,16")
    parser.add_argument("--burst_ratios", default="0.0,0.02")
    parser.add_argument("--phase_patterns", default="steady,bulk_update_trim")
    parser.add_argument("--trim_localities", default="mixed,hot,cold")
    parser.add_argument("--trim_burst_lengths", default="0,8")
    parser.add_argument("--trim_burst_intervals", default="0,128")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--out_dir", default="results/adversarial_workloads")
    args = parser.parse_args()

    outputs = run_search(args)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
