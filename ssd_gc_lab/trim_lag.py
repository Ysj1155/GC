from __future__ import annotations

"""
TRIM-to-GC lag analysis.

This module links two existing event streams:
- TRIM events say which physical block/page was invalidated.
- GC events say which victim block was later erased.

The resulting lag is measured in simulator steps, not wall-clock time.
"""

from bisect import bisect_left
from typing import Any, Dict, Iterable, List, Optional, Tuple


LAG_ROW_FIELDS = [
    "trim_step",
    "lpn",
    "old_block",
    "old_page",
    "gc_step",
    "lag_steps",
    "reclaimed",
    "gc_cause",
]


def _to_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _percentile(values: List[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _gc_index(gc_events: Iterable[Dict[str, Any]]) -> Dict[int, List[Tuple[int, Dict[str, Any]]]]:
    by_block: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for event in gc_events:
        victim = _to_int(event.get("victim"))
        step = _to_int(event.get("step"))
        if victim is None or step is None:
            continue
        by_block.setdefault(victim, []).append((step, event))

    for events in by_block.values():
        events.sort(key=lambda item: item[0])
    return by_block


def analyze_trim_to_gc_lag(
    trim_events: Iterable[Dict[str, Any]],
    gc_events: Iterable[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Match each TRIM-invalidated physical page to the next GC erase of its block.

    A TRIM event is eligible only when it invalidated a mapped physical page.
    Misses/retrims are intentionally excluded because there is no physical page
    to track toward a later erase.
    """
    gc_by_block = _gc_index(gc_events)
    rows: List[Dict[str, Any]] = []
    lags: List[int] = []

    for event in trim_events:
        if _to_int(event.get("trim_hit")) != 1:
            continue
        if _to_int(event.get("invalidated_pages")) != 1:
            continue

        trim_step = _to_int(event.get("step"))
        old_block = _to_int(event.get("old_block"))
        old_page = _to_int(event.get("old_page"))
        if trim_step is None or old_block is None:
            continue

        gc_step = None
        gc_cause = ""
        candidates = gc_by_block.get(old_block, [])
        candidate_steps = [step for step, _gc in candidates]
        pos = bisect_left(candidate_steps, trim_step)
        if pos < len(candidates):
            gc_step = candidates[pos][0]
            gc_cause = str(candidates[pos][1].get("cause", ""))

        reclaimed = 1 if gc_step is not None else 0
        lag_steps = (gc_step - trim_step) if gc_step is not None else ""
        if isinstance(lag_steps, int):
            lags.append(lag_steps)

        rows.append({
            "trim_step": trim_step,
            "lpn": event.get("lpn", ""),
            "old_block": old_block,
            "old_page": old_page if old_page is not None else "",
            "gc_step": gc_step if gc_step is not None else "",
            "lag_steps": lag_steps,
            "reclaimed": reclaimed,
            "gc_cause": gc_cause,
        })

    eligible = len(rows)
    reclaimed_count = len(lags)
    pending_count = eligible - reclaimed_count
    reclaim_rate = (reclaimed_count / eligible) if eligible else 0.0

    summary = {
        "trim_gc_lag_eligible_count": eligible,
        "trim_gc_lag_reclaimed_count": reclaimed_count,
        "trim_gc_lag_pending_count": pending_count,
        "trim_gc_reclaim_rate": round(reclaim_rate, 6),
        "trim_gc_lag_min": min(lags) if lags else 0,
        "trim_gc_lag_avg": round(sum(lags) / len(lags), 6) if lags else 0.0,
        "trim_gc_lag_p50": round(_percentile(lags, 0.50), 6) if lags else 0.0,
        "trim_gc_lag_p95": round(_percentile(lags, 0.95), 6) if lags else 0.0,
        "trim_gc_lag_max": max(lags) if lags else 0,
    }
    return rows, summary
