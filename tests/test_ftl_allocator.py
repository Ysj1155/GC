from ssd_gc_lab.gc_algos import greedy_policy
from ssd_gc_lab.metrics import collect_run_metrics
from ssd_gc_lab.models import SSD


def test_allocator_metrics_track_host_write_allocations() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=101)

    for lpn in range(6):
        ssd.write_lpn(lpn)

    row = collect_run_metrics(ssd)

    assert row["allocator_host_allocations"] == ssd.host_write_pages
    assert row["allocator_migration_allocations"] == 0
    assert row["allocator_event_count"] >= ssd.host_write_pages
    assert row["allocator_active_block_switches"] >= 1
    assert any(
        event["kind"] == "page_alloc" and event["purpose"] == "host_write"
        for event in ssd.allocator_event_log
    )


def test_allocator_metrics_track_gc_migration_allocations() -> None:
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
    assert row["allocator_host_allocations"] == ssd.host_write_pages
    assert row["allocator_migration_allocations"] == moved_valid
    assert any(
        event["kind"] == "page_alloc" and event["purpose"] == "gc_migration"
        for event in ssd.allocator_event_log
    )

def test_allocator_trace_reaches_simulator_trace() -> None:
    from ssd_gc_lab.config import SimConfig
    from ssd_gc_lab.simulator import Simulator

    cfg = SimConfig(num_blocks=8, pages_per_block=4, user_capacity_ratio=0.75, rng_seed=5)
    sim = Simulator(cfg, policy_name="greedy", enable_trace=True)
    sim.run([0, 1, 2, 3, 0])

    assert "active_block" in sim.trace
    assert "allocator_events" in sim.trace
    assert len(sim.trace["active_block"]) == 5
    assert len(sim.trace["allocator_events"]) == 5
    assert sim.trace["allocator_events"][-1] >= sim.ssd.host_write_pages