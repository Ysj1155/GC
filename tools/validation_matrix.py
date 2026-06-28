from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run portfolio-oriented SSD GC validation matrix scenarios."
    )
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--policies",
        default="greedy,cb,bsgc,cota",
        help="Comma-separated policies to run.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Defaults to profile-specific seeds.",
    )
    parser.add_argument("--out_dir", default="results/final_clean")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario names to run. Defaults to all scenarios.")
    parser.add_argument("--qc", choices=["off", "warn", "strict"], default="strict")
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

    results = []
    for cmd in commands:
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            results.append({"command": cmd, "returncode": None, "dry_run": True})
            continue
        proc = subprocess.run(cmd)
        results.append({"command": cmd, "returncode": proc.returncode, "dry_run": False})
        if proc.returncode != 0:
            break

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "policies": policies,
        "seeds": seeds,
        "scenario_filter": selected_scenarios,
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
