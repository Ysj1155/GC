from ssd_gc_lab.config import SimConfig
from ssd_gc_lab.metrics import collect_run_metrics
from ssd_gc_lab.simulator import Simulator
from ssd_gc_lab.trim_window import analyze_trim_windows


def test_trim_window_analysis_compares_before_after_trace() -> None:
    trace = {
        "step": [1, 2, 3, 4, 5, 6],
        "free_pages": [10, 10, 9, 8, 12, 12],
        "free_blocks": [2, 2, 1, 1, 3, 3],
        "invalid_pages": [0, 1, 2, 4, 1, 1],
        "host_writes": [1, 2, 3, 4, 5, 6],
        "device_writes": [1, 2, 3, 5, 8, 9],
        "gc_count": [0, 0, 0, 0, 1, 1],
    }
    trim_events = [
        {"op_step": 3, "trim_hit": 1, "invalidated_pages": 1},
        {"op_step": 4, "trim_hit": 1, "invalidated_pages": 1},
    ]

    rows, summary = analyze_trim_windows(
        trace,
        trim_events,
        before_ops=1,
        after_ops=1,
        merge_gap=1,
    )

    assert len(rows) == 1
    assert rows[0]["start_step"] == 3
    assert rows[0]["end_step"] == 4
    assert rows[0]["trim_ops"] == 2
    assert rows[0]["before_step"] == 2
    assert rows[0]["after_step"] == 5
    assert rows[0]["invalid_pages_delta"] == 0
    assert rows[0]["free_blocks_delta"] == 1
    assert rows[0]["gc_count_delta"] == 1

    assert summary["trim_window_count"] == 1
    assert summary["trim_window_avg_trim_ops"] == 2.0
    assert summary["trim_window_avg_gc_count_delta"] == 1.0
    assert summary["trim_window_gc_window_count"] == 1


def test_trim_window_metrics_reach_trace_enabled_run_summary() -> None:
    cfg = SimConfig(
        num_blocks=16,
        pages_per_block=8,
        user_capacity_ratio=0.85,
        gc_free_block_threshold=0.15,
        rng_seed=31,
    )
    sim = Simulator(cfg, policy_name="greedy", enable_trace=True, bg_gc_every=32)
    workload = []
    for lpn in range(80):
        workload.append(("write", lpn % cfg.user_total_pages))
        if lpn % 10 == 0:
            workload.append(("trim", lpn % cfg.user_total_pages))

    sim.run(workload)
    row = collect_run_metrics(sim)

    assert row["trim_window_count"] > 0
    assert row["trim_window_avg_trim_ops"] >= 1.0
    assert "trim_window_avg_invalid_pages_delta" in row
    assert all("op_step" in event for event in sim.ssd.trim_event_log)
