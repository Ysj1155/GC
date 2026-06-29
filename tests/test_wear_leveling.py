from ssd_gc_lab.metrics import collect_run_metrics
from ssd_gc_lab.models import PageState, SSD
from ssd_gc_lab.config import SimConfig
from ssd_gc_lab.simulator import Simulator


def assert_mapping_integrity(ssd: SSD) -> None:
    snapshot = ssd.mapping_integrity_snapshot()
    assert snapshot["dangling_mapping_entries"] == 0
    assert snapshot["dangling_reverse_entries"] == 0
    assert snapshot["valid_pages_without_reverse"] == 0


def make_static_wear_leveling_candidate(ssd: SSD) -> tuple[int, list[int]]:
    for lpn in range(4):
        ssd.write_lpn(lpn)

    source_idx = ssd.mapping[0][0]
    source_lpns = [lpn for lpn, ppn in ssd.mapping.items() if ppn[0] == source_idx]
    ssd.active_block_idx = None

    for idx, block in enumerate(ssd.blocks):
        block.erase_count = 5 if idx != source_idx else 0

    return source_idx, source_lpns


def test_static_wear_leveling_moves_valid_pages_and_erases_low_wear_source() -> None:
    ssd = SSD(num_blocks=10, pages_per_block=4, rng_seed=31)
    source_idx, source_lpns = make_static_wear_leveling_candidate(ssd)
    device_before = ssd.device_write_pages

    did_wl = ssd.perform_static_wear_leveling(threshold=2, min_valid_ratio=0.8, cause="unit_test")
    row = collect_run_metrics(ssd)

    assert did_wl is True
    assert ssd.wear_leveling_count == 1
    assert ssd.wear_leveling_moved_pages == len(source_lpns)
    assert ssd.device_write_pages == device_before + len(source_lpns)
    assert ssd.blocks[source_idx].erase_count == 1
    assert all(state == PageState.FREE for state in ssd.blocks[source_idx].pages)
    assert all(ssd.mapping[lpn][0] != source_idx for lpn in source_lpns)
    assert row["wear_leveling_count"] == 1
    assert row["wear_leveling_moved_pages"] == len(source_lpns)
    assert row["wear_leveling_allocations"] == len(source_lpns)
    assert any(event["kind"] == "wear_leveling_remap" for event in ssd.mapping_event_log)
    assert_mapping_integrity(ssd)


def test_static_wear_leveling_skips_when_wear_spread_is_low() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=32)
    for lpn in range(4):
        ssd.write_lpn(lpn)

    did_wl = ssd.perform_static_wear_leveling(threshold=2, min_valid_ratio=0.8)
    row = collect_run_metrics(ssd)

    assert did_wl is False
    assert row["wear_leveling_count"] == 0
    assert row["wear_leveling_skipped_low_spread"] == 1
    assert_mapping_integrity(ssd)


def test_simulator_runs_wear_leveling_on_configured_cadence() -> None:
    cfg = SimConfig(num_blocks=10, pages_per_block=4, user_capacity_ratio=0.75, rng_seed=33)
    sim = Simulator(
        cfg,
        policy_name="greedy",
        enable_trace=True,
        enable_wear_leveling=True,
        wear_leveling_every=1,
        wear_leveling_threshold=2,
        wear_leveling_min_valid_ratio=0.8,
    )
    source_idx, _source_lpns = make_static_wear_leveling_candidate(sim.ssd)
    assert sim.ssd.blocks[source_idx].erase_count == 0

    sim.run([99])
    row = collect_run_metrics(sim)

    assert row["wear_leveling_count"] >= 1
    assert "wear_leveling_event" in sim.trace
    assert "wear_spread" in sim.trace
    assert sim.trace["wear_leveling_event"][-1] == 1
    assert_mapping_integrity(sim.ssd)