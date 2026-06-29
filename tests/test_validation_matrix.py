from tools.validation_matrix import build_run_command, filter_scenarios, get_scenarios, iter_run_commands, run_matrix_commands


def test_get_scenarios_has_portfolio_stress_cases() -> None:
    names = {scenario.name for scenario in get_scenarios("quick")}

    assert "random_update_stress" in names
    assert "trim_burst" in names
    assert "delete_after_bulk_load" in names
    assert "low_op_pressure" in names
    assert "endurance_short" in names
    assert "wear_leveling_off" in names
    assert "wear_leveling_on" in names


def test_build_run_command_includes_manifest_and_qc() -> None:
    scenario = get_scenarios("quick")[0]
    cmd = build_run_command(
        scenario=scenario,
        policy="greedy",
        seed=41,
        out_root="results/final_clean",
        qc="strict",
        python_executable="python",
    )

    assert cmd[0] == "python"
    assert cmd[1].endswith("run_sim.py")
    assert cmd[2:4] == ["--gc_policy", "greedy"]
    assert "--manifest_json" in cmd
    assert "manifest.json" in cmd
    assert "--qc" in cmd
    assert "strict" in cmd
    assert "results/final_clean" in " ".join(cmd)


def test_iter_run_commands_crosses_scenarios_policies_and_seeds() -> None:
    scenarios = get_scenarios("quick")[:2]
    commands = iter_run_commands(
        scenarios=scenarios,
        policies=["greedy", "cota"],
        seeds=[41, 42],
        out_root="results/final_clean",
        qc="strict",
        python_executable="python",
    )

    assert len(commands) == 8
    joined = [" ".join(cmd) for cmd in commands]
    assert any("trim_burst" in cmd for cmd in joined)
    assert any("--gc_policy cota" in cmd for cmd in joined)

def test_filter_scenarios_preserves_requested_order() -> None:
    scenarios = get_scenarios("quick")
    selected = filter_scenarios(scenarios, ["trim_locality_mixed", "trim_burst"])

    assert [scenario.name for scenario in selected] == ["trim_locality_mixed", "trim_burst"]


def test_wear_leveling_comparison_scenarios_match_except_wl_knobs() -> None:
    scenarios = {scenario.name: scenario for scenario in get_scenarios("quick")}
    off = dict(scenarios["wear_leveling_off"].params)
    on = dict(scenarios["wear_leveling_on"].params)

    assert on.pop("enable_wear_leveling") is True
    assert on.pop("wear_leveling_every") > 0
    assert on.pop("wear_leveling_threshold") >= 1
    assert on.pop("wear_leveling_min_valid_ratio") > 0
    assert off == on


def test_run_matrix_commands_can_skip_existing_manifest(tmp_path) -> None:
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    cmd = [
        "python",
        "tools/run_sim.py",
        "--out_dir",
        str(run_dir),
        "--manifest_json",
        "manifest.json",
    ]

    rows = run_matrix_commands([cmd], dry_run=False, jobs=1, skip_existing=True)

    assert rows[0]["skipped"] is True
    assert rows[0]["returncode"] == 0