from ssd_gc_lab.gc_algos import greedy_policy
from ssd_gc_lab.metrics import collect_run_metrics
from ssd_gc_lab.models import SSD
from ssd_gc_lab.config import SimConfig
from ssd_gc_lab.simulator import Simulator


def test_mapping_lifecycle_tracks_create_and_overwrite_update() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=21)

    ssd.write_lpn(7)
    first_ppn = ssd.mapping[7]
    ssd.write_lpn(7)

    row = collect_run_metrics(ssd)
    kinds = [event["kind"] for event in ssd.mapping_event_log]

    assert first_ppn not in ssd.reverse_map
    assert "map_create" in kinds
    assert "overwrite_invalidate" in kinds
    assert "map_update" in kinds
    assert row["mapping_creates"] == 1
    assert row["mapping_updates"] == 1
    assert row["mapping_overwrite_invalidations"] == 1
    assert row["dangling_mapping_entries"] == 0
    assert row["dangling_reverse_entries"] == 0
    assert row["valid_pages_without_reverse"] == 0


def test_mapping_lifecycle_tracks_trim_unmap() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=22)

    ssd.write_lpn(3)
    old_ppn = ssd.mapping[3]
    ssd.trim_lpn(3)

    row = collect_run_metrics(ssd)
    last_event = ssd.mapping_event_log[-1]

    assert 3 not in ssd.mapping
    assert old_ppn not in ssd.reverse_map
    assert last_event["kind"] == "trim_unmap"
    assert last_event["old_block"] == old_ppn[0]
    assert row["mapping_trim_unmaps"] == 1
    assert row["mapping_unmaps"] == 1
    assert row["mapping_entries"] == 0
    assert row["reverse_mapping_entries"] == 0


def test_mapping_lifecycle_tracks_gc_remap() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=17)

    for lpn in range(8):
        ssd.write_lpn(lpn)

    victim_idx = ssd.mapping[0][0]
    victim_lpns = [
        lpn for lpn, ppn in list(ssd.mapping.items())
        if ppn[0] == victim_idx
    ]
    ssd.write_lpn(victim_lpns[0])

    ssd.collect_garbage(greedy_policy, cause="unit_test")
    row = collect_run_metrics(ssd)
    moved_valid = ssd.gc_event_log[-1]["moved_valid"]

    assert moved_valid > 0
    assert row["mapping_gc_remaps"] == moved_valid
    assert row["dangling_mapping_entries"] == 0
    assert row["dangling_reverse_entries"] == 0
    assert any(event["kind"] == "gc_remap" for event in ssd.mapping_event_log)


def test_mapping_event_count_reaches_simulator_trace() -> None:
    cfg = SimConfig(num_blocks=8, pages_per_block=4, user_capacity_ratio=0.75, rng_seed=5)
    sim = Simulator(cfg, policy_name="greedy", enable_trace=True)
    sim.run([0, 1, 2, 3, 0])

    row = collect_run_metrics(sim)

    assert "mapping_events" in sim.trace
    assert len(sim.trace["mapping_events"]) == 5
    assert sim.trace["mapping_events"][-1] == row["mapping_event_count"]
    assert row["mapping_event_count"] >= sim.ssd.host_write_pages