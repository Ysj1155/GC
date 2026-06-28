from __future__ import annotations

"""
Before/after TRIM window analysis.

The analyzer groups nearby TRIM operations into windows, then compares trace
snapshots before and after each window. The step unit is the simulator workload
operation step (`op_step`), not wall-clock time.
"""

from bisect import bisect_left, bisect_right
from typing import Any, Dict, Iterable, List, Optional, Tuple


WINDOW_ROW_FIELDS = [
    "window_id",
    "start_step",
    "end_step",
    "trim_ops",
    "trim_hits",
    "trim_invalidated_pages",
    "before_step",
    "after_step",
    "before_free_pages",
    "after_free_pages",
    "free_pages_delta",
    "before_free_blocks",
    "after_free_blocks",
    "free_blocks_delta",
    "before_invalid_pages",
    "after_invalid_pages",
    "invalid_pages_delta",
    "before_gc_count",
    "after_gc_count",
    "gc_count_delta",
    "before_waf",
    "after_waf",
    "waf_delta",
]


def _to_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _trace_rows(trace: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    steps = trace.get("step", []) or []
    rows: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        host = _to_float(_pick(trace, "host_writes", idx, 0)) or 0.0
        device = _to_float(_pick(trace, "device_writes", idx, 0)) or 0.0
        waf = (device / host) if host > 0 else 0.0
        rows.append({
            "step": _to_int(step),
            "free_pages": _pick(trace, "free_pages", idx, ""),
            "free_blocks": _pick(trace, "free_blocks", idx, ""),
            "invalid_pages": _pick(trace, "invalid_pages", idx, ""),
            "gc_count": _pick(trace, "gc_count", idx, ""),
            "host_writes": host,
            "device_writes": device,
            "waf": round(waf, 6),
        })
    return [row for row in rows if row["step"] is not None]


def _pick(trace: Dict[str, List[Any]], key: str, idx: int, default: Any = "") -> Any:
    values = trace.get(key, []) or []
    return values[idx] if idx < len(values) else default


def _eligible_trim_events(trim_events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for event in trim_events:
        step = _to_int(event.get("op_step"))
        if step is None:
            step = _to_int(event.get("step"))
        if step is None:
            continue
        item = dict(event)
        item["_window_step"] = step
        events.append(item)
    return sorted(events, key=lambda item: int(item["_window_step"]))


def _group_trim_windows(trim_events: List[Dict[str, Any]], merge_gap: int) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for event in trim_events:
        step = int(event["_window_step"])
        if not groups:
            groups.append([event])
            continue
        prev_step = int(groups[-1][-1]["_window_step"])
        if step - prev_step <= merge_gap:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _snapshot(rows: List[Dict[str, Any]], steps: List[int], target: int, *, before: bool) -> Dict[str, Any]:
    if not rows:
        return {}
    if before:
        pos = bisect_right(steps, target) - 1
        pos = max(0, pos)
    else:
        pos = bisect_left(steps, target)
        pos = min(pos, len(rows) - 1)
    return rows[pos]


def _delta(after: Dict[str, Any], before: Dict[str, Any], key: str) -> Any:
    a = _to_float(after.get(key))
    b = _to_float(before.get(key))
    if a is None or b is None:
        return ""
    diff = a - b
    return int(diff) if diff.is_integer() else round(diff, 6)


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    values = [_to_float(row.get(key)) for row in rows]
    nums = [value for value in values if value is not None]
    return round(sum(nums) / len(nums), 6) if nums else 0.0


def analyze_trim_windows(
    trace: Dict[str, List[Any]],
    trim_events: Iterable[Dict[str, Any]],
    *,
    before_ops: int = 32,
    after_ops: int = 32,
    merge_gap: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compare trace snapshots around grouped TRIM windows."""
    trace_rows = _trace_rows(trace or {})
    trim_rows = _eligible_trim_events(trim_events)
    if not trace_rows or not trim_rows:
        return [], _empty_summary()

    groups = _group_trim_windows(trim_rows, max(0, int(merge_gap)))
    steps = [int(row["step"]) for row in trace_rows]
    rows: List[Dict[str, Any]] = []

    for idx, group in enumerate(groups, start=1):
        start_step = int(group[0]["_window_step"])
        end_step = int(group[-1]["_window_step"])
        before_row = _snapshot(trace_rows, steps, start_step - max(0, int(before_ops)), before=True)
        after_row = _snapshot(trace_rows, steps, end_step + max(0, int(after_ops)), before=False)

        before_waf = before_row.get("waf", "")
        after_waf = after_row.get("waf", "")
        row = {
            "window_id": idx,
            "start_step": start_step,
            "end_step": end_step,
            "trim_ops": len(group),
            "trim_hits": sum(1 for event in group if _to_int(event.get("trim_hit")) == 1),
            "trim_invalidated_pages": sum(_to_int(event.get("invalidated_pages")) or 0 for event in group),
            "before_step": before_row.get("step", ""),
            "after_step": after_row.get("step", ""),
            "before_free_pages": before_row.get("free_pages", ""),
            "after_free_pages": after_row.get("free_pages", ""),
            "free_pages_delta": _delta(after_row, before_row, "free_pages"),
            "before_free_blocks": before_row.get("free_blocks", ""),
            "after_free_blocks": after_row.get("free_blocks", ""),
            "free_blocks_delta": _delta(after_row, before_row, "free_blocks"),
            "before_invalid_pages": before_row.get("invalid_pages", ""),
            "after_invalid_pages": after_row.get("invalid_pages", ""),
            "invalid_pages_delta": _delta(after_row, before_row, "invalid_pages"),
            "before_gc_count": before_row.get("gc_count", ""),
            "after_gc_count": after_row.get("gc_count", ""),
            "gc_count_delta": _delta(after_row, before_row, "gc_count"),
            "before_waf": before_waf,
            "after_waf": after_waf,
            "waf_delta": _delta(after_row, before_row, "waf"),
        }
        rows.append(row)

    summary = {
        "trim_window_count": len(rows),
        "trim_window_avg_trim_ops": _mean(rows, "trim_ops"),
        "trim_window_avg_invalid_pages_delta": _mean(rows, "invalid_pages_delta"),
        "trim_window_avg_free_pages_delta": _mean(rows, "free_pages_delta"),
        "trim_window_avg_free_blocks_delta": _mean(rows, "free_blocks_delta"),
        "trim_window_avg_gc_count_delta": _mean(rows, "gc_count_delta"),
        "trim_window_avg_waf_delta": _mean(rows, "waf_delta"),
        "trim_window_gc_window_count": sum(1 for row in rows if (_to_float(row.get("gc_count_delta")) or 0.0) > 0.0),
    }
    return rows, summary


def _empty_summary() -> Dict[str, Any]:
    return {
        "trim_window_count": 0,
        "trim_window_avg_trim_ops": 0.0,
        "trim_window_avg_invalid_pages_delta": 0.0,
        "trim_window_avg_free_pages_delta": 0.0,
        "trim_window_avg_free_blocks_delta": 0.0,
        "trim_window_avg_gc_count_delta": 0.0,
        "trim_window_avg_waf_delta": 0.0,
        "trim_window_gc_window_count": 0,
    }
