import json

from validation_report import generate_report, load_run_rows, summarize_rows


def _write_manifest(path, scenario, policy, seed, waf, wear_std):
    run_dir = path / scenario / f"{policy}_seed{seed}"
    run_dir.mkdir(parents=True)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {
                    "policy": policy,
                    "seed": seed,
                    "ops": 100,
                    "waf": waf,
                    "gc_count": 3,
                    "wear_avg": 1.0,
                    "wear_std": wear_std,
                    "wear_max": 2.0,
                    "free_blocks": 4,
                },
                "parameters": {"gc_policy": policy, "seed": seed},
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
