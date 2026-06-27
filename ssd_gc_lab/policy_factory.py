from __future__ import annotations

"""
policy_factory.py

GC policy wiring lives here.

The GC algorithms themselves stay in gc_algos.py.  This module only translates
CLI/experiment arguments into the callable shape expected by Simulator:

    policy(blocks) -> victim_index | None

Keeping this logic in one place prevents run_sim.py, experiments.py, and future
validation tools from each carrying their own slightly different policy wrapper.
"""

from typing import Any, Callable, List, Optional

from ssd_gc_lab import gc_algos


def _arg(args: Any, name: str, default: Any = None) -> Any:
    """Read an argparse-style attribute with a safe default."""
    return getattr(args, name, default)


def configure_policy_globals(args: Any) -> None:
    """
    Apply optional gc_algos global knobs if the current algorithm module exposes
    them.  Older versions of this repo had these hooks; keeping the calls here
    preserves compatibility without making every runner know about them.
    """
    if hasattr(gc_algos, "config_cold_bias"):
        gc_algos.config_cold_bias(_arg(args, "cold_victim_bias", 1.0))
    if hasattr(gc_algos, "config_trim_age_bonus"):
        gc_algos.config_trim_age_bonus(_arg(args, "trim_age_bonus", 0.0))
    if hasattr(gc_algos, "config_victim_prefetch_k"):
        gc_algos.config_victim_prefetch_k(_arg(args, "victim_prefetch_k", 1))


def build_gc_policy(args: Any, sim: Any) -> Callable[[List[Any]], Optional[int]]:
    """
    Build a GC victim-selection policy from CLI/experiment arguments.

    Some policies are plain functions.  Others need runtime context:
    - COTA may receive alpha/beta/gamma/delta weights.
    - ATCB and RE50315 need the simulator's current SSD step.
    """
    configure_policy_globals(args)

    name = (_arg(args, "gc_policy", "") or "").lower()

    if name == "greedy":
        return gc_algos.greedy_policy

    if name in ("cb", "cost_benefit"):
        return gc_algos.cb_policy

    if name == "bsgc":
        return gc_algos.bsgc_policy

    if name == "cota":
        alpha = _arg(args, "cota_alpha")
        beta = _arg(args, "cota_beta")
        gamma = _arg(args, "cota_gamma")
        delta = _arg(args, "cota_delta")

        if not any(v is not None for v in (alpha, beta, gamma, delta)):
            return gc_algos.cota_policy

        def cota_with_weights(blocks, _a=alpha, _b=beta, _g=gamma, _d=delta):
            kwargs = {}
            if _a is not None:
                kwargs["alpha"] = _a
            if _b is not None:
                kwargs["beta"] = _b
            if _g is not None:
                kwargs["gamma"] = _g
            if _d is not None:
                kwargs["delta"] = _d
            return gc_algos.cota_policy(blocks, **kwargs)

        return cota_with_weights

    if name in ("atcb", "atcb_policy"):
        atcb_policy = getattr(gc_algos, "atcb_policy", None)
        if atcb_policy is None:
            raise RuntimeError("gc_algos.atcb_policy 가 없습니다. gc_algos.py 를 업데이트하세요.")

        def atcb_with_now(blocks, _sim=sim):
            return atcb_policy(
                blocks,
                alpha=_arg(args, "atcb_alpha", 0.5),
                beta=_arg(args, "atcb_beta", 0.3),
                gamma=_arg(args, "atcb_gamma", 0.1),
                eta=_arg(args, "atcb_eta", 0.1),
                now_step=_sim.ssd._step,
            )

        return atcb_with_now

    if name in ("re50315", "re50315_policy"):
        re50315_policy = getattr(gc_algos, "re50315_policy", None)
        if re50315_policy is None:
            raise RuntimeError("gc_algos.re50315_policy 가 없습니다. gc_algos.py 를 업데이트하세요.")

        def re50315_with_now(blocks, _sim=sim):
            return re50315_policy(
                blocks,
                K=_arg(args, "re50315_K", 1.0),
                now_step=_sim.ssd._step,
            )

        return re50315_with_now

    raise ValueError(f"지원하지 않는 GC 정책: {_arg(args, 'gc_policy')}")


def inject_policy(args: Any, sim: Any) -> None:
    """Assign the selected policy callable to sim.gc_policy."""
    sim.gc_policy = build_gc_policy(args, sim)
