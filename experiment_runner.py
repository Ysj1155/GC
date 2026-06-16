from __future__ import annotations

"""
experiment_runner.py

Shared experiment execution helpers.

run_sim.py and experiments.py are both CLI entry points.  They should describe
"how the user wants to run things", not duplicate the lower-level recipe for
building SimConfig, creating workloads, injecting GC policies, warming up the
SSD, running the simulator, and checking result sanity.
"""

import csv
import os
from datetime import datetime
from typing import Any, Dict, Optional

from config import SimConfig
from metrics import append_summary_csv, summary_row
from policy_factory import inject_policy
from simulator import Simulator
from workload import make_workload


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def resolve_output_path(path: str | None, out_dir: str) -> str | None:
    """
    Resolve an output path relative to out_dir unless it is already absolute.

    Example:
    - out_dir=results/run, path=summary.csv -> results/run/summary.csv
    - path=C:/tmp/summary.csv -> C:/tmp/summary.csv
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(out_dir, path)


def infer_user_total_pages(cfg: Any) -> int:
    """
    Infer the logical user-page count from a SimConfig-like object.

    This compatibility helper lets older scripts and newer config shapes keep
    working while the simulator gradually becomes cleaner.
    """
    for attr in ("user_total_pages", "total_user_pages", "ssd_total_pages"):
        value = getattr(cfg, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    blocks = (
        getattr(cfg, "num_blocks", None)
        or getattr(cfg, "blocks", None)
        or getattr(cfg, "total_blocks", None)
    )
    ppb = getattr(cfg, "pages_per_block", None) or getattr(cfg, "ppb", None)
    ratio = (
        getattr(cfg, "user_capacity_ratio", None)
        or getattr(cfg, "capacity_ratio", None)
        or 1.0
    )

    try:
        if blocks and ppb:
            return int(int(blocks) * int(ppb) * float(ratio))
    except Exception:
        pass

    total_pages = getattr(cfg, "total_pages", None)
    try:
        if total_pages and isinstance(total_pages, int) and total_pages > 0:
            return int(total_pages * float(ratio))
    except Exception:
        pass

    raise RuntimeError(
        "user_total_pages 추정 실패: SimConfig 구조가 예상과 다릅니다. "
        "필요 필드(user_total_pages / (blocks*ppb*ratio) / total_pages)를 확인하세요."
    )


def build_config(args: Any) -> SimConfig:
    """Build the simulator configuration from argparse-style arguments."""
    cfg = SimConfig(
        num_blocks=args.blocks,
        pages_per_block=args.pages_per_block,
        gc_free_block_threshold=args.gc_free_block_threshold,
        rng_seed=args.seed,
        user_capacity_ratio=args.user_capacity_ratio,
    )

    user_total_pages = infer_user_total_pages(cfg)
    try:
        setattr(cfg, "user_total_pages", user_total_pages)
    except Exception:
        pass

    return cfg


def build_workload(args: Any, user_total_pages: int):
    """Create the workload sequence for one simulation run."""
    return make_workload(
        n_ops=args.ops,
        update_ratio=args.update_ratio,
        ssd_total_pages=user_total_pages,
        rng_seed=args.seed,
        hot_ratio=args.hot_ratio,
        hot_weight=args.hot_weight,
        enable_trim=args.enable_trim,
        trim_ratio=args.trim_ratio,
    )


def apply_warmup(sim: Simulator, cfg: SimConfig, user_total_pages: int, warmup_fill: float) -> None:
    """
    Pre-fill the SSD before the measured workload.

    Warmup is useful for avoiding unrealistically empty-device behavior.  The
    helper keeps at least a small reserve so GC still has room to migrate data.
    """
    if warmup_fill <= 0.0:
        return

    reserve_free_blocks = 2
    ppb = getattr(cfg, "pages_per_block", getattr(cfg, "ppb", 64))
    max_warm_pages = max(0, user_total_pages - reserve_free_blocks * ppb)
    target_pages = min(
        int(user_total_pages * min(max(warmup_fill, 0.0), 0.99)),
        max_warm_pages,
    )

    wrote = 0
    lpn = 0
    while wrote < target_pages and lpn < user_total_pages:
        if getattr(sim.ssd, "free_pages", 1) <= ppb:
            sim.ssd.collect_garbage(sim.gc_policy, cause="warmup")
        sim.ssd.write_lpn(lpn)
        wrote += 1
        lpn += 1


def build_run_meta(args: Any) -> Dict[str, Any]:
    """Build the metadata columns that make a result row reproducible."""
    note = getattr(args, "note", "")
    return {
        "run_id": note or f"{args.gc_policy}_{args.seed}",
        "policy": args.gc_policy,
        "ops": args.ops,
        "update_ratio": args.update_ratio,
        "hot_ratio": args.hot_ratio,
        "hot_weight": args.hot_weight,
        "seed": args.seed,
        "trim_enabled": 1 if getattr(args, "enable_trim", False) else 0,
        "trim_ratio": getattr(args, "trim_ratio", 0.0),
        "warmup_fill": args.warmup_fill,
        "bg_gc_every": args.bg_gc_every,
        "note": note,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "cota_alpha": getattr(args, "cota_alpha", None),
        "cota_beta": getattr(args, "cota_beta", None),
        "cota_gamma": getattr(args, "cota_gamma", None),
        "cota_delta": getattr(args, "cota_delta", None),
        "cold_victim_bias": getattr(args, "cold_victim_bias", 1.0),
        "trim_age_bonus": getattr(args, "trim_age_bonus", 0.0),
        "victim_prefetch_k": getattr(args, "victim_prefetch_k", 1),
    }


def run_single_experiment(args: Any, *, enable_trace: bool = False) -> tuple[Simulator, Dict[str, Any], Dict[str, Any]]:
    """
    Run one simulation and return (simulator, metadata, summary_row).

    This is the shared execution path used by both single-run and batch CLIs.
    """
    cfg = build_config(args)
    user_total_pages = infer_user_total_pages(cfg)

    sim = Simulator(cfg, enable_trace=enable_trace, bg_gc_every=args.bg_gc_every)
    inject_policy(args, sim)

    workload = build_workload(args, user_total_pages)
    apply_warmup(sim, cfg, user_total_pages, args.warmup_fill)
    sim.run(workload)

    meta = build_run_meta(args)
    row = summary_row(sim, meta)
    return sim, meta, row


def quick_qc(row: Dict[str, Any]) -> bool:
    """
    Run basic sanity checks against a result row.

    This is not a correctness proof; it catches obviously broken metrics early.
    """
    warn = []
    g = row.get

    waf = g("waf")
    host = g("host_writes")
    dev = g("device_writes")
    gc_n = g("gc_count")
    gc_t = g("gc_avg_s")
    fb = g("free_blocks")
    fp = g("free_pages")
    vp = g("valid_pages")
    ip = g("invalid_pages")
    tp = g("total_pages")
    ws = g("wear_std")
    wmin = g("wear_min")
    wmax = g("wear_max")
    tr = g("transition_rate")
    rr = g("reheat_rate")
    trim = g("trimmed_pages")

    if waf is None or waf < 1.0:
        warn.append(f"WAF={waf} (기본적으로 >= 1.0 이어야 정상)")
    if host is not None and dev is not None and dev < host:
        warn.append(f"device_writes({dev}) < host_writes({host})")
    if gc_n is not None and gc_n < 0:
        warn.append(f"gc_count={gc_n} (<0)")
    if gc_t is not None and gc_t < 0:
        warn.append(f"gc_avg_s={gc_t} (<0)")
    if fb is not None and fb <= 0:
        warn.append(f"free_blocks={fb} (<=0)")

    if all(x is not None for x in (vp, ip, fp, tp)):
        if (vp + ip + fp) != tp:
            warn.append(f"valid+invalid+free != total_pages ({vp}+{ip}+{fp} != {tp})")

    if ws is not None and ws < 0:
        warn.append(f"wear_std={ws} (<0)")
    if wmin is not None and wmax is not None and wmax < wmin:
        warn.append(f"wear_max({wmax}) < wear_min({wmin})")

    for name, value in [("transition_rate", tr), ("reheat_rate", rr)]:
        if value is not None and not (0.0 <= value <= 1.0):
            warn.append(f"{name}={value} (0~1 범위 밖)")

    if trim is not None and trim < 0:
        warn.append(f"trimmed_pages={trim} (<0)")
    if trim is not None and tp is not None and trim > tp:
        warn.append(f"trimmed_pages({trim}) > total_pages({tp})")

    if warn:
        print("[QC] WARN:", " | ".join(warn))
        return False

    print("[QC] OK  :", f"policy={g('policy')} seed={g('seed')} waf={waf}")
    return True


def run_once(args: Any, out_dir: str, out_csv: Optional[str]) -> tuple[Dict[str, Any], bool]:
    """
    Compatibility wrapper for batch runners.

    The out_dir parameter is retained for the older experiments.py contract.
    """
    _ = out_dir
    sim, meta, row = run_single_experiment(args, enable_trace=False)
    if out_csv:
        append_summary_csv(out_csv, sim, meta)
    ok = quick_qc(row)
    return row, ok


def write_trace_csv(path: str, sim: Simulator) -> None:
    """Write per-operation trace data when tracing was enabled."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "free_pages", "device_writes", "gc_count", "gc_event"])
        for i in range(len(sim.trace["step"])):
            writer.writerow([
                sim.trace["step"][i],
                sim.trace["free_pages"][i],
                sim.trace["device_writes"][i],
                sim.trace["gc_count"][i],
                sim.trace["gc_event"][i],
            ])


def write_gc_events_csv(path: str, sim: Simulator) -> None:
    """Write per-GC event log rows if the SSD produced them."""
    events = getattr(sim.ssd, "gc_event_log", None)
    if not events:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(events[0].keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(event)
