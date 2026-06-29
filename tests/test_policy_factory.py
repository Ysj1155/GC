from argparse import Namespace

from ssd_gc_lab.config import SimConfig
from ssd_gc_lab.policy_factory import build_gc_policy
from ssd_gc_lab.simulator import Simulator
from tools.validation_matrix import DEFAULT_POLICY_SET, POLICY_TIERS


def test_default_policy_set_keeps_baselines_before_custom_policy() -> None:
    assert POLICY_TIERS["core_baseline"] == ["greedy", "cb"]
    assert "age_stale" in POLICY_TIERS["balanced_baseline"]
    assert POLICY_TIERS["custom"] == ["cota"]
    assert DEFAULT_POLICY_SET[-1] == "cota"


def test_age_stale_policy_is_available_through_policy_factory() -> None:
    sim = Simulator(SimConfig(num_blocks=8, pages_per_block=4), policy_name="greedy")
    args = Namespace(gc_policy="age_stale", age_stale_K=25.0)

    policy = build_gc_policy(args, sim)

    assert callable(policy)