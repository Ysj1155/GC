from ssd_gc_lab.gc_algos import greedy_policy
from ssd_gc_lab.metrics import collect_run_metrics
from ssd_gc_lab.models import SSD
from ssd_gc_lab.trim_lag import analyze_trim_to_gc_lag


def test_trim_to_gc_lag_matches_next_victim_erase() -> None:
    trim_events = [
        {"step": 10, "lpn": 1, "trim_hit": 1, "invalidated_pages": 1, "old_block": 2, "old_page": 0},
        {"step": 20, "lpn": 2, "trim_hit": 1, "invalidated_pages": 1, "old_block": 2, "old_page": 1},
        {"step": 25, "lpn": 3, "trim_hit": 1, "invalidated_pages": 1, "old_block": 4, "old_page": 0},
        {"step": 30, "lpn": 4, "trim_hit": 0, "invalidated_pages": 0, "old_block": "", "old_page": ""},
    ]
    gc_events = [
        {"step": 18, "victim": 2, "cause": "low_free"},
        {"step": 35, "victim": 2, "cause": "low_free"},
        {"step": 40, "victim": 5, "cause": "low_free"},
    ]

    rows, summary = analyze_trim_to_gc_lag(trim_events, gc_events)

    assert len(rows) == 3
    assert rows[0]["gc_step"] == 18
    assert rows[0]["lag_steps"] == 8
    assert rows[1]["gc_step"] == 35
    assert rows[1]["lag_steps"] == 15
    assert rows[2]["reclaimed"] == 0
    assert rows[2]["lag_steps"] == ""

    assert summary["trim_gc_lag_eligible_count"] == 3
    assert summary["trim_gc_lag_reclaimed_count"] == 2
    assert summary["trim_gc_lag_pending_count"] == 1
    assert summary["trim_gc_reclaim_rate"] == 0.666667
    assert summary["trim_gc_lag_min"] == 8
    assert summary["trim_gc_lag_avg"] == 11.5
    assert summary["trim_gc_lag_max"] == 15


def test_trim_to_gc_lag_reaches_run_metrics() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=17)

    for lpn in range(4):
        ssd.write_lpn(lpn)

    victim_idx = ssd.mapping[0][0]
    lpns_in_victim = [
        lpn for lpn, ppn in list(ssd.mapping.items())
        if ppn[0] == victim_idx
    ]
    for lpn in lpns_in_victim:
        ssd.trim_lpn(lpn)

    ssd.collect_garbage(greedy_policy, cause="unit_test")
    row = collect_run_metrics(ssd)

    assert row["trim_gc_lag_eligible_count"] == len(lpns_in_victim)
    assert row["trim_gc_lag_reclaimed_count"] == len(lpns_in_victim)
    assert row["trim_gc_lag_pending_count"] == 0
    assert row["trim_gc_reclaim_rate"] == 1.0
    assert row["trim_gc_lag_avg"] >= 0.0
