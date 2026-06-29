from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import itertools
import os
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from ssd_gc_lab.experiment_runner import run_single_experiment
from tools.insight_miner import pareto_front
from tools.validation_report import write_csv


def _parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_param_grid(grid: str) -> List[Dict[str, float]]:
    if not grid:
        return [{}]
    items: List[tuple[str, List[float]]] = []
    for chunk in grid.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, raw_values = chunk.split("=", 1)
        items.append((key.strip(), _parse_float_list(raw_values)))

    keys = [item[0] for item in items]
    value_lists = [item[1] for item in items]
    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def _base_namespace(args: argparse.Namespace, seed: int, params: Dict[str, float]) -> argparse.Namespace:
    data = {
        "ops": args.ops,
        "update_ratio": args.update_ratio,
        "hot_ratio": args.hot_ratio,
        "hot_weight": args.hot_weight,
        "blocks": args.blocks,
        "pages_per_block": args.pages_per_block,
        "gc_free_block_threshold": args.gc_free_block_threshold,
        "user_capacity_ratio": args.user_capacity_ratio,
        "seed": seed,
        "bg_gc_every": args.bg_gc_every,
        "enable_trim": args.trim_ratio > 0.0,
        "trim_ratio": args.trim_ratio,
        "warmup_fill": args.warmup_fill,
        "burst_length": args.burst_length,
        "burst_ratio": args.burst_ratio,
        "phase_pattern": args.phase_pattern,
        "trim_locality": args.trim_locality,
        "trim_burst_length": args.trim_burst_length,
        "trim_burst_interval": args.trim_burst_interval,
        "gc_policy": args.policy,
        "cota_alpha": None,
        "cota_beta": None,
        "cota_gamma": None,
        "cota_delta": None,
        "cold_victim_bias": 1.0,
        "trim_age_bonus": 0.0,
        "victim_prefetch_k": 1,
        "age_stale_K": 50.0,
        "atcb_alpha": 0.5,
        "atcb_beta": 0.3,
        "atcb_gamma": 0.1,
        "atcb_eta": 0.1,
        "re50315_K": 1.0,
        "note": "",
    }
    data.update(params)
    param_note = "_".join(f"{key}{value:g}" for key, value in sorted(params.items())) or "default"
    data["note"] = f"{args.policy}_{param_note}_seed{seed}"
    return argparse.Namespace(**data)


def _metric(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, "") else 0.0


def summarize_params(rows: Sequence[Dict[str, Any]], param_names: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in param_names)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        item: Dict[str, Any] = {name: value for name, value in zip(param_names, key)}
        item["runs"] = len(group)
        for metric in ("waf", "gc_count", "wear_std", "wear_max", "free_blocks"):
            values = [_metric(row, metric) for row in group]
            item[f"{metric}_mean"] = round(mean(values), 6) if values else ""
            item[f"{metric}_min"] = round(min(values), 6) if values else ""
            item[f"{metric}_max"] = round(max(values), 6) if values else ""
        item["score"] = round(
            float(item.get("waf_mean") or 0.0) * 0.50
            + float(item.get("wear_std_mean") or 0.0) * 0.35
            + float(item.get("gc_count_mean") or 0.0) * 0.0005,
            6,
        )
        out.append(item)
    return sorted(out, key=lambda row: float(row.get("score") or 0.0))


def _write_report(path: str, policy: str, param_names: Sequence[str], summary: Sequence[Dict[str, Any]], pareto: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Policy Parameter Tuning Report\n\n")
        f.write(f"- Policy: `{policy}`\n")
        f.write(f"- Tuned parameters: {', '.join(param_names) if param_names else 'none'}\n")
        f.write(f"- Parameter combinations: {len(summary)}\n")
        f.write(f"- Pareto candidates: {len(pareto)}\n\n")
        f.write("## Best Composite Scores\n\n")
        f.write("| rank | parameters | score | waf_mean | wear_std_mean | gc_count_mean |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for rank, row in enumerate(summary[:10], start=1):
            params = ", ".join(f"{name}={row.get(name)}" for name in param_names) or "default"
            f.write(
                f"| {rank} | {params} | {row.get('score')} | {row.get('waf_mean')} | "
                f"{row.get('wear_std_mean')} | {row.get('gc_count_mean')} |\n"
            )
        f.write("\n## Reading Notes\n\n")
        f.write("- The composite score is intentionally simple and should not replace engineering judgement.\n")
        f.write("- Pareto rows preserve trade-offs that may be useful even when their composite score is not first.\n")
        f.write("- Use this as a candidate generator, then re-run promising regions with more seeds and stronger workloads.\n")


def run_tuner(args: argparse.Namespace) -> Dict[str, str]:
    params = parse_param_grid(args.param_grid)
    param_names = sorted({key for item in params for key in item})
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for param_set in params:
        for seed in seeds:
            run_args = _base_namespace(args, seed, param_set)
            _sim, _meta, row = run_single_experiment(run_args, enable_trace=False)
            for key, value in param_set.items():
                row[key] = value
            rows.append(row)

    summary = summarize_params(rows, param_names)
    front = pareto_front(summary, metrics=("waf_mean", "wear_std_mean", "gc_count_mean"))

    outputs = {
        "tuning_runs": os.path.join(args.out_dir, "tuning_runs.csv"),
        "tuning_summary": os.path.join(args.out_dir, "tuning_summary.csv"),
        "pareto_params": os.path.join(args.out_dir, "pareto_params.csv"),
        "report": os.path.join(args.out_dir, "tuning_report.md"),
    }
    write_csv(outputs["tuning_runs"], rows)
    write_csv(outputs["tuning_summary"], summary)
    write_csv(outputs["pareto_params"], front)
    _write_report(outputs["report"], args.policy, param_names, summary, front)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small policy parameter tuner over the SSD GC simulator.")
    parser.add_argument("--policy", default="cota", choices=["greedy", "cb", "cost_benefit", "age_stale", "bsgc", "cota", "atcb", "re50315"])
    parser.add_argument("--param_grid", default="cota_alpha=0.45,0.55;cota_delta=0.05,0.15")
    parser.add_argument("--seeds", default="41,42")
    parser.add_argument("--ops", type=int, default=20000)
    parser.add_argument("--update_ratio", type=float, default=0.9)
    parser.add_argument("--hot_ratio", type=float, default=0.2)
    parser.add_argument("--hot_weight", type=float, default=0.8)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--pages_per_block", type=int, default=32)
    parser.add_argument("--gc_free_block_threshold", type=float, default=0.12)
    parser.add_argument("--user_capacity_ratio", type=float, default=0.95)
    parser.add_argument("--bg_gc_every", type=int, default=0)
    parser.add_argument("--trim_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_fill", type=float, default=0.7)
    parser.add_argument("--burst_length", type=int, default=0)
    parser.add_argument("--burst_ratio", type=float, default=0.0)
    parser.add_argument("--phase_pattern", default="steady", choices=["steady", "bulk_update_trim", "phased", "rocksdb_like"])
    parser.add_argument("--trim_locality", default="mixed", choices=["mixed", "hot", "cold"])
    parser.add_argument("--trim_burst_length", type=int, default=0)
    parser.add_argument("--trim_burst_interval", type=int, default=0)
    parser.add_argument("--age_stale_K", type=float, default=50.0)
    parser.add_argument("--out_dir", default="results/policy_tuning")
    args = parser.parse_args()

    outputs = run_tuner(args)
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
