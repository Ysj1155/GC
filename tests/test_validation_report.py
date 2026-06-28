import json

from tools.validation_report import generate_report, load_run_rows, summarize_locality_rows, summarize_rows


def _write_manifest(path, scenario, policy, seed, waf, wear_std, params=None, metrics=None):
    run_dir = path / scenario / f"{policy}_seed{seed}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        base_metrics = {
            "policy": policy,
            "seed": seed,
            "ops": 100,
            "waf": waf,
            "gc_count": 3,
            "wear_avg": 1.0,
            "wear_std": wear_std,
            "wear_max": 2.0,
            "free_blocks": 4,
        }
        base_metrics.update(metrics or {})
        base_params = {"gc_policy": policy, "seed": seed}
        base_params.update(params or {})
        json.dump(
            {
                "metrics": base_metrics,
                "parameters": base_params,
            },
            f,
        )


def test_validation_report_summarizes_manifest_runs(tmp_path) -> None:
    _write_manifest(tmp_path, "stress", "greedy", 41, 1.2, 3.0)
    _write_manifest(tmp_path, "stress", "greedy", 42, 1.4, 5.0)

    rows = load_run_rows(str(tmp_path))
    summary = summarize_rows(rows)

    assert len(rows) == 2
    assert summary[0]["scenario"] == "stress"
    assert summary[0]["policy"] == "greedy"
    assert summary[0]["waf_mean"] == 1.3
    assert summary[0]["wear_std_mean"] == 4.0


def test_generate_report_writes_markdown_and_csv(tmp_path) -> None:
    _write_manifest(tmp_path, "stress", "cota", 41, 1.1, 2.0)

    outputs = generate_report(str(tmp_path))

    for path in outputs.values():
        assert path
    assert (tmp_path / "validation_report.md").exists()
    assert (tmp_path / "validation_summary.csv").exists()
    assert (tmp_path / "validation_runs.csv").exists()

def test_locality_summary_groups_by_policy_and_trim_target(tmp_path) -> None:
    _write_manifest(
        tmp_path,
        "trim_locality_hot",
        "greedy",
        41,
        1.2,
        3.0,
        params={"trim_locality": "hot"},
        metrics={"trim_gc_lag_avg": 10, "trim_gc_reclaim_rate": 0.5},
    )
    _write_manifest(
        tmp_path,
        "trim_locality_cold",
        "greedy",
        41,
        1.4,
        4.0,
        params={"trim_locality": "cold"},
        metrics={"trim_gc_lag_avg": 20, "trim_gc_reclaim_rate": 0.75},
    )

    rows = load_run_rows(str(tmp_path))
    locality = summarize_locality_rows(rows)

    assert len(locality) == 2
    labels = {(row["policy"], row["trim_locality"]) for row in locality}
    assert ("greedy", "hot") in labels
    assert ("greedy", "cold") in labels
    hot = next(row for row in locality if row["trim_locality"] == "hot")
    assert hot["waf_mean"] == 1.2
    assert hot["trim_gc_lag_avg_mean"] == 10
