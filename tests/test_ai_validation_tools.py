from tools.insight_miner import build_policy_scorecard, find_anomaly_runs, pareto_front
from tools.policy_parameter_tuner import parse_param_grid


def test_insight_miner_builds_scorecard_and_pareto_front() -> None:
    rows = [
        {"scenario": "stress", "policy": "greedy", "seed": 1, "waf": 1.1, "wear_std": 5.0, "gc_count": 10, "wear_max": 7, "free_blocks": 4},
        {"scenario": "stress", "policy": "bsgc", "seed": 1, "waf": 1.3, "wear_std": 1.0, "gc_count": 12, "wear_max": 3, "free_blocks": 4},
        {"scenario": "stress", "policy": "cota", "seed": 1, "waf": 1.2, "wear_std": 2.0, "gc_count": 11, "wear_max": 4, "free_blocks": 4},
    ]

    scorecard = build_policy_scorecard(rows)
    front = pareto_front(rows)

    assert len(scorecard) == 3
    assert any(row["policy"] == "greedy" and row["waf_rank"] == 1 for row in scorecard)
    assert any(row["policy"] == "bsgc" and row["wear_std_rank"] == 1 for row in scorecard)
    assert len(front) == 3


def test_insight_miner_flags_obvious_anomaly() -> None:
    rows = [
        {"scenario": "stress", "policy": "greedy", "seed": 1, "waf": 1.0, "wear_std": 1.0, "gc_count": 10, "wear_max": 2, "free_blocks": 4},
        {"scenario": "stress", "policy": "greedy", "seed": 2, "waf": 1.1, "wear_std": 1.2, "gc_count": 11, "wear_max": 2, "free_blocks": 4},
        {"scenario": "stress", "policy": "greedy", "seed": 3, "waf": 5.0, "wear_std": 9.0, "gc_count": 99, "wear_max": 10, "free_blocks": 1},
    ]

    anomalies = find_anomaly_runs(rows, z_threshold=0.5)

    assert anomalies
    assert any("high_waf" in row["anomaly_reasons"] for row in anomalies)


def test_policy_parameter_grid_parser() -> None:
    grid = parse_param_grid("cota_alpha=0.45,0.55;cota_delta=0.05,0.15")

    assert len(grid) == 4
    assert {"cota_alpha": 0.45, "cota_delta": 0.05} in grid
    assert {"cota_alpha": 0.55, "cota_delta": 0.15} in grid
