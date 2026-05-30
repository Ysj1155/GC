from argparse import Namespace

from manifest import build_run_manifest


def test_build_run_manifest_contains_reproducibility_fields(tmp_path) -> None:
    args = Namespace(
        gc_policy="greedy",
        ops=100,
        seed=7,
        out_dir=str(tmp_path),
        out_csv="summary.csv",
        manifest_json="manifest.json",
    )
    metrics = {
        "policy": "greedy",
        "host_writes": 100,
        "device_writes": 125,
        "waf": 1.25,
    }

    manifest = build_run_manifest(
        args=args,
        metrics_row=metrics,
        out_dir=str(tmp_path),
        artifacts={"summary_csv": str(tmp_path / "summary.csv")},
        cwd=".",
    )

    assert manifest["schema_version"] == 1
    assert manifest["parameters"]["gc_policy"] == "greedy"
    assert manifest["parameters"]["seed"] == 7
    assert manifest["metrics"]["waf"] == 1.25
    assert manifest["artifacts"]["summary_csv"].endswith("summary.csv")
    assert "commit" in manifest["git"]
    assert "version" in manifest["python"]
