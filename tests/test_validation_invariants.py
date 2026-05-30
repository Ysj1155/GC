from config import SimConfig
from gc_algos import cota_policy, greedy_policy
from metrics import collect_run_metrics
from models import PageState, SSD
from simulator import Simulator
from workload import make_workload


def assert_ssd_invariants(ssd: SSD) -> None:
    total_valid = 0
    total_invalid = 0
    total_free = 0

    for block in ssd.blocks:
        page_valid = sum(1 for state in block.pages if state == PageState.VALID)
        page_invalid = sum(1 for state in block.pages if state == PageState.INVALID)
        page_free = sum(1 for state in block.pages if state == PageState.FREE)

        assert block.valid_count == page_valid
        assert block.invalid_count == page_invalid
        assert block.free_count == page_free

        total_valid += page_valid
        total_invalid += page_invalid
        total_free += page_free

    assert total_valid + total_invalid + total_free == ssd.total_pages
    assert len(ssd.mapping) == total_valid
    assert len(ssd.reverse_map) == total_valid

    for lpn, ppn in ssd.mapping.items():
        block_idx, page_idx = ppn
        assert ssd.blocks[block_idx].pages[page_idx] == PageState.VALID
        assert ssd.reverse_map[ppn] == lpn

    for ppn, lpn in ssd.reverse_map.items():
        assert ssd.mapping[lpn] == ppn


def test_overwrite_invalidates_old_physical_page() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=7)

    ssd.write_lpn(0)
    old_ppn = ssd.mapping[0]
    ssd.write_lpn(0)

    assert ssd.mapping[0] != old_ppn
    assert old_ppn not in ssd.reverse_map
    assert ssd.blocks[old_ppn[0]].pages[old_ppn[1]] == PageState.INVALID
    assert ssd.host_write_pages == 2
    assert ssd.device_write_pages == 2
    assert_ssd_invariants(ssd)


def test_trim_removes_mapping_without_extra_writes() -> None:
    ssd = SSD(num_blocks=8, pages_per_block=4, rng_seed=11)

    ssd.write_lpn(3)
    host_before = ssd.host_write_pages
    device_before = ssd.device_write_pages
    old_ppn = ssd.mapping[3]

    ssd.trim_lpn(3)
    ssd.trim_lpn(3)

    assert 3 not in ssd.mapping
    assert old_ppn not in ssd.reverse_map
    assert ssd.blocks[old_ppn[0]].pages[old_ppn[1]] == PageState.INVALID
    assert ssd.host_write_pages == host_before
    assert ssd.device_write_pages == device_before
    assert_ssd_invariants(ssd)


def test_all_invalid_gc_is_counted_as_gc_event() -> None:
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

    assert ssd.blocks[victim_idx].valid_count == 0
    assert ssd.blocks[victim_idx].invalid_count > 0

    ssd.collect_garbage(greedy_policy, cause="unit_test")

    assert ssd.gc_count == 1
    assert ssd.blocks[victim_idx].erase_count == 1
    assert ssd.blocks[victim_idx].free_count == ssd.pages_per_block
    assert_ssd_invariants(ssd)


def test_simulator_metrics_remain_sane_after_gc_pressure() -> None:
    cfg = SimConfig(
        num_blocks=16,
        pages_per_block=8,
        user_capacity_ratio=0.75,
        gc_free_block_threshold=0.2,
        rng_seed=23,
    )
    sim = Simulator(cfg, policy_name="greedy", bg_gc_every=16)
    sim.gc_policy = cota_policy

    workload = make_workload(
        n_ops=300,
        update_ratio=0.85,
        ssd_total_pages=cfg.user_total_pages,
        rng_seed=23,
        hot_ratio=0.2,
        hot_weight=0.75,
    )

    sim.run(workload)
    row = collect_run_metrics(sim)

    assert row["host_writes"] == 300
    assert row["device_writes"] >= row["host_writes"]
    assert row["waf"] >= 1.0
    assert row["valid_pages"] + row["invalid_pages"] + row["free_pages"] == row["total_pages"]
    assert_ssd_invariants(sim.ssd)


def test_workload_generation_is_seed_reproducible() -> None:
    first = make_workload(
        n_ops=100,
        update_ratio=0.8,
        ssd_total_pages=256,
        rng_seed=99,
        hot_ratio=0.2,
        hot_weight=0.7,
        enable_trim=True,
        trim_ratio=0.1,
    )
    second = make_workload(
        n_ops=100,
        update_ratio=0.8,
        ssd_total_pages=256,
        rng_seed=99,
        hot_ratio=0.2,
        hot_weight=0.7,
        enable_trim=True,
        trim_ratio=0.1,
    )

    assert first == second
