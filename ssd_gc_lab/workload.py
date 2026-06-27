from __future__ import annotations

"""
workload.py

Workload generation for the SSD GC simulator.

The generator is deterministic for the same seed and supports both the older
plain-write format:

    [lpn, lpn, ...]

and the event format used when TRIM or phased behavior is enabled:

    [("write", lpn), ("trim", lpn), ...]

The default arguments preserve the original steady workload behavior. Newer
arguments add validation-oriented stress dimensions: bursty update pressure,
phase changes, TRIM locality, and TRIM bursts.
"""

from typing import List, Tuple, Union
import random


class _IndexList:
    """O(1)-ish add/remove/random-choice helper for live LPN sets."""

    def __init__(self) -> None:
        self._arr: List[int] = []
        self._pos: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._arr)

    def add(self, x: int) -> None:
        if x in self._pos:
            return
        self._pos[x] = len(self._arr)
        self._arr.append(x)

    def remove(self, x: int) -> None:
        idx = self._pos.pop(x, None)
        if idx is None:
            return
        last = self._arr.pop()
        if idx < len(self._arr):
            self._arr[idx] = last
            self._pos[last] = idx

    def choice(self, rng: random.Random) -> int:
        if not self._arr:
            raise IndexError("empty _IndexList")
        return self._arr[rng.randrange(len(self._arr))]

    def to_list(self) -> List[int]:
        return list(self._arr)


Workload = Union[List[int], List[Tuple[str, int]]]


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def make_workload(
    n_ops: int,
    update_ratio: float,
    ssd_total_pages: int,
    rng_seed: int = 42,
    hot_ratio: float = 0.2,
    hot_weight: float = 0.7,
    enable_trim: bool = False,
    trim_ratio: float = 0.0,
    burst_length: int = 0,
    burst_ratio: float = 0.0,
    phase_pattern: str = "steady",
    trim_locality: str = "mixed",
    trim_burst_length: int = 0,
    trim_burst_interval: int = 0,
) -> Workload:
    """
    Build a deterministic page-level workload.

    Existing dimensions:
    - update_ratio: probability of rewriting a live LPN instead of allocating a new LPN.
    - hot_ratio: fraction of the LPN space treated as hot.
    - hot_weight: probability of selecting from the hot live set for updates.
    - trim_ratio: probability of issuing a TRIM event when live LPNs exist.

    Additional stress dimensions:
    - burst_length / burst_ratio: creates short update-heavy bursts.
    - phase_pattern: steady, bulk_update_trim/phased, or rocksdb_like.
    - trim_locality: mixed, hot, or cold trim target selection.
    - trim_burst_length / trim_burst_interval: periodic TRIM-heavy windows.
    """
    rng = random.Random(rng_seed)
    n_ops = max(0, int(n_ops))
    update_ratio = _clamp01(update_ratio)
    hot_ratio = _clamp01(hot_ratio)
    hot_weight = _clamp01(hot_weight)
    trim_ratio = _clamp01(trim_ratio)
    burst_length = max(0, int(burst_length or 0))
    burst_ratio = _clamp01(burst_ratio or 0.0)
    phase_pattern = (phase_pattern or "steady").lower()
    trim_locality = (trim_locality or "mixed").lower()
    trim_burst_length = max(0, int(trim_burst_length or 0))
    trim_burst_interval = max(0, int(trim_burst_interval or 0))

    if ssd_total_pages <= 0:
        if enable_trim or trim_ratio > 0.0:
            return [("write", 0)] * n_ops
        return [0] * n_ops

    hot_cut = max(1, int(ssd_total_pages * hot_ratio))
    live_hot = _IndexList()
    live_cold = _IndexList()
    next_lpn = 0
    burst_remaining = 0

    def is_hot(lpn: int) -> bool:
        return lpn < hot_cut

    def add_live(lpn: int) -> None:
        (live_hot if is_hot(lpn) else live_cold).add(lpn)

    def remove_live(lpn: int) -> None:
        (live_hot if is_hot(lpn) else live_cold).remove(lpn)

    def have_live() -> bool:
        return (len(live_hot) + len(live_cold)) > 0

    def pick_update_lpn(effective_hot_weight: float) -> int:
        if rng.random() < effective_hot_weight:
            if len(live_hot) > 0:
                return live_hot.choice(rng)
            if len(live_cold) > 0:
                return live_cold.choice(rng)
        else:
            if len(live_cold) > 0:
                return live_cold.choice(rng)
            if len(live_hot) > 0:
                return live_hot.choice(rng)
        return 0

    def pick_trim_lpn(effective_hot_weight: float) -> int:
        if trim_locality == "hot":
            if len(live_hot) > 0:
                return live_hot.choice(rng)
            if len(live_cold) > 0:
                return live_cold.choice(rng)
        elif trim_locality == "cold":
            if len(live_cold) > 0:
                return live_cold.choice(rng)
            if len(live_hot) > 0:
                return live_hot.choice(rng)
        else:
            return pick_update_lpn(effective_hot_weight)
        return 0

    def phase_params(op_idx: int) -> tuple[float, float, float]:
        if phase_pattern in ("steady", "none", ""):
            return update_ratio, hot_weight, trim_ratio

        progress = op_idx / max(1, n_ops)
        if phase_pattern in ("bulk_update_trim", "phased"):
            if progress < 0.30:
                return min(update_ratio, 0.20), hot_weight, 0.0
            if progress < 0.80:
                return max(update_ratio, 0.90), max(hot_weight, 0.85), trim_ratio
            return max(update_ratio, 0.70), hot_weight, max(trim_ratio, 0.20)

        if phase_pattern == "rocksdb_like":
            if progress < 0.40:
                return min(update_ratio, 0.20), max(hot_weight, 0.85), 0.0
            phase_slot = int((progress - 0.40) / 0.10)
            if phase_slot % 2 == 0:
                return max(update_ratio, 0.92), max(hot_weight, 0.90), trim_ratio
            return max(update_ratio, 0.70), max(hot_weight, 0.80), max(trim_ratio, 0.05)

        return update_ratio, hot_weight, trim_ratio

    def emit_write(effective_update_ratio: float, effective_hot_weight: float) -> int:
        nonlocal next_lpn
        new_write = (not have_live()) or (rng.random() >= effective_update_ratio)
        if new_write and next_lpn < ssd_total_pages:
            lpn = next_lpn
            next_lpn += 1
            add_live(lpn)
            return lpn
        if have_live():
            return pick_update_lpn(effective_hot_weight)
        lpn = min(next_lpn, ssd_total_pages - 1)
        if next_lpn < ssd_total_pages:
            next_lpn += 1
            add_live(lpn)
        return lpn

    force_events = (
        enable_trim
        or trim_ratio > 0.0
        or trim_burst_length > 0
        or trim_burst_interval > 0
        or phase_pattern not in ("steady", "none", "")
    )

    if not force_events:
        ops: List[int] = []
        for op_idx in range(n_ops):
            effective_update_ratio, effective_hot_weight, _effective_trim_ratio = phase_params(op_idx)
            if burst_remaining <= 0 and burst_length > 0 and rng.random() < burst_ratio:
                burst_remaining = burst_length
            if burst_remaining > 0:
                effective_update_ratio = max(effective_update_ratio, 0.95)
                effective_hot_weight = max(effective_hot_weight, 0.90)
                burst_remaining -= 1
            ops.append(emit_write(effective_update_ratio, effective_hot_weight))
        return ops

    ops2: List[Tuple[str, int]] = []
    for op_idx in range(n_ops):
        effective_update_ratio, effective_hot_weight, effective_trim_ratio = phase_params(op_idx)
        if burst_remaining <= 0 and burst_length > 0 and rng.random() < burst_ratio:
            burst_remaining = burst_length
        if burst_remaining > 0:
            effective_update_ratio = max(effective_update_ratio, 0.95)
            effective_hot_weight = max(effective_hot_weight, 0.90)
            burst_remaining -= 1

        in_trim_burst = (
            trim_burst_interval > 0
            and trim_burst_length > 0
            and (op_idx % trim_burst_interval) < trim_burst_length
        )
        if in_trim_burst:
            effective_trim_ratio = max(effective_trim_ratio, 0.80)

        if effective_trim_ratio > 0.0 and have_live() and rng.random() < effective_trim_ratio:
            lpn = pick_trim_lpn(effective_hot_weight)
            ops2.append(("trim", lpn))
            remove_live(lpn)
            continue

        ops2.append(("write", emit_write(effective_update_ratio, effective_hot_weight)))

    return ops2


def make_phased_workload(phases, ssd_total_pages: int, base_seed: int = 42) -> Workload:
    """Build a workload by concatenating explicit phase dictionaries."""
    out: list = []
    made_tuple = False

    for i, phase in enumerate(phases):
        chunk = make_workload(
            n_ops=phase["n_ops"],
            update_ratio=phase.get("update_ratio", 0.8),
            ssd_total_pages=ssd_total_pages,
            rng_seed=phase.get("seed", base_seed + i),
            hot_ratio=phase.get("hot_ratio", 0.2),
            hot_weight=phase.get("hot_weight", 0.85),
            enable_trim=phase.get("enable_trim", False),
            trim_ratio=phase.get("trim_ratio", 0.0),
            burst_length=phase.get("burst_length", 0),
            burst_ratio=phase.get("burst_ratio", 0.0),
            phase_pattern=phase.get("phase_pattern", "steady"),
            trim_locality=phase.get("trim_locality", "mixed"),
            trim_burst_length=phase.get("trim_burst_length", 0),
            trim_burst_interval=phase.get("trim_burst_interval", 0),
        )
        if chunk and isinstance(chunk[0], tuple):
            made_tuple = True
        out.extend(chunk)

    if made_tuple:
        norm: List[Tuple[str, int]] = []
        for item in out:
            if isinstance(item, tuple):
                norm.append(item)
            else:
                norm.append(("write", int(item)))
        return norm
    return out


def only_writes(seq: Workload) -> List[int]:
    out: List[int] = []
    for item in seq:
        if isinstance(item, tuple):
            op, lpn = item
            if op == "write":
                out.append(int(lpn))
        else:
            out.append(int(item))
    return out


def trim_count(seq: Workload) -> int:
    return sum(1 for item in seq if isinstance(item, tuple) and item[0] == "trim")


def make_rocksdb_like_phases(user_pages: int, base_seed: int = 500) -> list:
    bulk = int(user_pages * 0.8)
    burst = int(user_pages * 0.2)
    phases = [{
        "n_ops": bulk,
        "update_ratio": 0.2,
        "hot_ratio": 0.2,
        "hot_weight": 0.85,
        "seed": base_seed,
    }]
    for i in range(3):
        phases.append({
            "n_ops": burst,
            "update_ratio": 0.9,
            "hot_ratio": 0.2,
            "hot_weight": 0.9,
            "seed": base_seed + i + 1,
        })
        phases.append({
            "n_ops": burst,
            "update_ratio": 0.7,
            "hot_ratio": 0.2,
            "hot_weight": 0.85,
            "seed": base_seed + i + 10,
        })
    return phases
