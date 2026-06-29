from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MatrixScenario:
    name: str
    description: str
    params: Dict[str, Any]

POLICY_TIERS: Dict[str, List[str]] = {
    "core_baseline": ["greedy", "cb"],
    "balanced_baseline": ["age_stale", "bsgc", "atcb", "re50315"],
    "custom": ["cota"],
}

DEFAULT_POLICY_SET: List[str] = [
    policy
    for tier in ("core_baseline", "balanced_baseline", "custom")
    for policy in POLICY_TIERS[tier]
]

def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _profile_seeds(profile: str) -> List[int]:
    if profile == "full":
        return [41, 42, 43, 44, 45]
    return [41]


def get_scenarios(profile: str) -> List[MatrixScenario]:
    """Return portfolio-oriented validation scenarios."""
    if profile not in {"quick", "full"}:
        raise ValueError(f"unknown profile: {profile}")

    scale = 1 if profile == "quick" else 8

    return [
        MatrixScenario(
            name="random_update_stress",
            description="High-update random write pressure with hot/cold skew.",
            params={
                "ops": 20_000 * scale,
                "update_ratio": 0.85,
                "hot_ratio": 0.2,
                "hot_weight": 0.75,
                "user_capacity_ratio": 0.9,
                "bg_gc_every": 256,
                "warmup_fill": 0.6,
            },
        ),
        MatrixScenario(
            name="trim_burst",
            description="Mixed random updates with TRIM pressure.",
            params={
                "ops": 20_000 * scale,
                "update_ratio": 0.75,
                "hot_ratio": 0.2,
                "hot_weight": 0.7,
                "enable_trim": True,
                "trim_ratio": 0.1,
                "user_capacity_ratio": 0.9,
                "bg_gc_every": 256,
                "warmup_fill": 0.6,
            },
        ),
        MatrixScenario(
            name="delete_after_bulk_load",
            description="Bulk-load followed by TRIM-heavy cleanup pressure.",
            params={
                "ops": 20_000 * scale,
                "update_ratio": 0.65,
                "hot_ratio": 0.25,
                "hot_weight": 0.8,
                "enable_trim": True,
                "trim_ratio": 0.05,
                "trim_locality": "cold",
                "trim_burst_length": 16,
                "trim_burst_interval": 256,
                "phase_pattern": "bulk_update_trim",
                "user_capacity_ratio": 0.92,
                "bg_gc_every": 192,
                "warmup_fill": 0.65,
            },
        ),
        *[
            MatrixScenario(
                name=f"trim_locality_{locality}",
                description=f"Controlled TRIM locality sensitivity run targeting {locality} LPNs.",
                params={
                    "ops": 16_000 * scale,
                    "update_ratio": 0.72,
                    "hot_ratio": 0.2,
                    "hot_weight": 0.8,
                    "enable_trim": True,
                    "trim_ratio": 0.08,
                    "trim_locality": locality,
                    "trim_burst_length": 12,
                    "trim_burst_interval": 192,
                    "phase_pattern": "bulk_update_trim",
                    "user_capacity_ratio": 0.9,
                    "bg_gc_every": 192,
                    "warmup_fill": 0.65,
                },
            )
            for locality in ("hot", "cold", "mixed")
        ],
        MatrixScenario(
            name="low_op_pressure",
            description="Low over-provisioning pressure to force GC decisions.",
            params={
                "ops": 20_000 * scale,
                "update_ratio": 0.9,
                "hot_ratio": 0.15,
                "hot_weight": 0.8,
                "user_capacity_ratio": 0.95,
                "gc_free_block_threshold": 0.08,
                "bg_gc_every": 128,
            },
        ),
        MatrixScenario(
            name="wear_leveling_off",
            description="Wear-leveling comparison baseline with high update pressure.",
            params={
                "ops": 6_000 * scale,
                "update_ratio": 0.9,
                "hot_ratio": 0.2,
                "hot_weight": 0.75,
                "user_capacity_ratio": 0.88,
                "gc_free_block_threshold": 0.1,
                "bg_gc_every": 96,
                "warmup_fill": 0.55,
            },
        ),
        MatrixScenario(
            name="wear_leveling_on",
            description="Same workload with static wear-leveling enabled.",
            params={
                "ops": 6_000 * scale,
                "update_ratio": 0.9,
                "hot_ratio": 0.2,
                "hot_weight": 0.75,
                "user_capacity_ratio": 0.88,
                "gc_free_block_threshold": 0.1,
                "bg_gc_every": 96,
                "warmup_fill": 0.55,
                "enable_wear_leveling": True,
                "wear_leveling_every": 96,
                "wear_leveling_threshold": 1,
                "wear_leveling_min_valid_ratio": 0.5,
            },
        ),
        MatrixScenario(
            name="endurance_short",
            description="Longer high-update run for wear distribution checks.",
            params={
                "ops": 40_000 * scale,
                "update_ratio": 0.9,
                "hot_ratio": 0.2,
                "hot_weight": 0.75,
                "user_capacity_ratio": 0.9,
                "bg_gc_every": 128,
                "warmup_fill": 0.5,
            },
        ),
    ]




def filter_scenarios(scenarios: Iterable[MatrixScenario], selected_names: Iterable[str]) -> List[MatrixScenario]:
    selected = list(selected_names)
    if not selected:
        return list(scenarios)

    by_name = {scenario.name: scenario for scenario in scenarios}
    missing = [name for name in selected if name not in by_name]
    if missing:
        known = ",".join(sorted(by_name))
        raise ValueError(f"unknown scenario(s): {','.join(missing)}. known={known}")
    return [by_name[name] for name in selected]

def _append_option(cmd: List[str], key: str, value: Any) -> None:
    if value is None or value is False:
        return
    opt = f"--{key}"
    if value is True:
        cmd.append(opt)
        return
    cmd.extend([opt, str(value)])


def build_run_command(
    *,
    scenario: MatrixScenario,
    policy: str,
    seed: int,
    out_root: str,
    qc: str = "strict",
    python_executable: Optional[str] = None,
) -> List[str]:
    """Build one run_sim.py command for a matrix entry."""
    run_dir = os.path.join(out_root, scenario.name, f"{policy}_seed{seed}")
    cmd = [
        python_executable or sys.executable,
        os.path.join(os.path.dirname(__file__), "run_sim.py"),
        "--gc_policy",
        policy,
        "--seed",
        str(seed),
        "--out_dir",
        run_dir,
        "--out_csv",
        "summary.csv",
        "--manifest_json",
        "manifest.json",
        "--qc",
        qc,
        "--note",
        f"{scenario.name}_{policy}_{seed}",
    ]

    for key, value in scenario.params.items():
        _append_option(cmd, key, value)

    if scenario.params.get("enable_trim"):
        cmd.extend([
            "--gc_events_csv",
            "gc_events.csv",
            "--trim_events_csv",
            "trim_events.csv",
            "--trim_gc_lag_csv",
            "trim_gc_lag.csv",
            "--trim_windows_csv",
            "trim_windows.csv",
        ])

    return cmd


def iter_run_commands(
    *,
    scenarios: Iterable[MatrixScenario],
    policies: Iterable[str],
    seeds: Iterable[int],
    out_root: str,
    qc: str,
    python_executable: Optional[str] = None,
) -> List[List[str]]:
    commands: List[List[str]] = []
    for scenario in scenarios:
        for policy in policies:
            for seed in seeds:
                commands.append(
                    build_run_command(
                        scenario=scenario,
                        policy=policy,
                        seed=int(seed),
                        out_root=out_root,
                        qc=qc,
                        python_executable=python_executable,
                    )
                )
    return commands


def write_matrix_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")




def _option_value(cmd: List[str], option: str, default: Optional[str] = None) -> Optional[str]:
    try:
        idx = cmd.index(option)
    except ValueError:
        return default
    if idx + 1 >= len(cmd):
        return default
    return cmd[idx + 1]


def _manifest_path_for_command(cmd: List[str]) -> Optional[str]:
    out_dir = _option_value(cmd, "--out_dir")
    if not out_dir:
        return None
    manifest_name = _option_value(cmd, "--manifest_json", "manifest.json") or "manifest.json"
    return os.path.join(out_dir, manifest_name)
def _run_subprocess(index: int, cmd: List[str]) -> Dict[str, Any]:
    print(f"[MATRIX] started {index + 1}: " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "index": index,
        "command": cmd,
        "returncode": proc.returncode,
        "dry_run": False,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _emit_completed_run(item: Dict[str, Any], total: int) -> None:
    index = int(item["index"])
    returncode = item["returncode"]
    print(f"[MATRIX] completed {index + 1}/{total} returncode={returncode}", flush=True)
    stdout = item.get("stdout") or ""
    stderr = item.get("stderr") or ""
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)


def run_matrix_commands(commands: List[List[str]], *, dry_run: bool = False, jobs: int = 1, skip_existing: bool = False) -> List[Dict[str, Any]]:
    if jobs < 1:
        raise ValueError("jobs must be >= 1")

    skipped: List[Dict[str, Any]] = []
    active_commands: List[tuple[int, List[str]]] = []
    for index, cmd in enumerate(commands):
        manifest_path = _manifest_path_for_command(cmd)
        if skip_existing and manifest_path and os.path.exists(manifest_path):
            print(f"[MATRIX] skip existing {index + 1}/{len(commands)}: {manifest_path}", flush=True)
            skipped.append({"index": index, "command": cmd, "returncode": 0, "dry_run": dry_run, "skipped": True})
        else:
            active_commands.append((index, cmd))

    if dry_run:
        results: List[Dict[str, Any]] = list(skipped)
        for index, cmd in active_commands:
            print(" ".join(cmd), flush=True)
            results.append({"index": index, "command": cmd, "returncode": None, "dry_run": True, "skipped": False})
        return sorted(results, key=lambda item: int(item["index"]))

    if not active_commands:
        return sorted(skipped, key=lambda item: int(item["index"]))

    if jobs == 1:
        results = []
        for index, cmd in active_commands:
            print(" ".join(cmd), flush=True)
            proc = subprocess.run(cmd)
            results.append({"index": index, "command": cmd, "returncode": proc.returncode, "dry_run": False})
            if proc.returncode != 0:
                break
        return sorted(skipped + results, key=lambda item: int(item["index"]))

    print(f"[MATRIX] running {len(active_commands)} active runs with jobs={jobs} ({len(skipped)} skipped, {len(commands)} total)", flush=True)
    results_by_index: Dict[int, Dict[str, Any]] = {}
    failed = False
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_subprocess, index, cmd): index
            for index, cmd in active_commands
        }
        for future in as_completed(futures):
            if failed:
                continue
            item = future.result()
            result = {k: v for k, v in item.items() if k not in {"stdout", "stderr"}}
            results_by_index[int(item["index"])] = result
            _emit_completed_run(item, len(commands))
            if item["returncode"] != 0:
                failed = True
                for pending in futures:
                    pending.cancel()

    return sorted(skipped + [results_by_index[index] for index in sorted(results_by_index)], key=lambda item: int(item["index"]))

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run portfolio-oriented SSD GC validation matrix scenarios."
    )
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICY_SET),
        help="Comma-separated policies to run.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Defaults to profile-specific seeds.",
    )
    parser.add_argument("--out_dir", "--outdir", default="results/final_clean", help="Output directory for matrix results.")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario names to run. Defaults to all scenarios.")
    parser.add_argument("--qc", choices=["off", "warn", "strict"], default="strict")
    parser.add_argument("--jobs", type=int, default=1, help="Number of matrix runs to execute in parallel.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs whose manifest.json already exists.")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    policies = _split_csv(args.policies)
    seeds = [int(x) for x in _split_csv(args.seeds)] if args.seeds else _profile_seeds(args.profile)
    scenarios = get_scenarios(args.profile)
    selected_scenarios = _split_csv(args.scenarios) if args.scenarios else []
    scenarios = filter_scenarios(scenarios, selected_scenarios)
    commands = iter_run_commands(
        scenarios=scenarios,
        policies=policies,
        seeds=seeds,
        out_root=args.out_dir,
        qc=args.qc,
    )

    results = run_matrix_commands(commands, dry_run=args.dry_run, jobs=args.jobs, skip_existing=args.skip_existing)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "policies": policies,
        "seeds": seeds,
        "scenario_filter": selected_scenarios,
        "jobs": args.jobs,
        "skip_existing": args.skip_existing,
        "scenarios": [
            {"name": s.name, "description": s.description, "params": s.params}
            for s in scenarios
        ],
        "runs": results,
    }
    write_matrix_manifest(os.path.join(args.out_dir, "matrix_manifest.json"), payload)

    failed = [r for r in results if r["returncode"] not in (0, None)]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
