from validation_matrix import build_run_command, get_scenarios, iter_run_commands


def test_get_scenarios_has_portfolio_stress_cases() -> None:
    names = {scenario.name for scenario in get_scenarios("quick")}

    assert "random_update_stress" in names
    assert "trim_burst" in names
    assert "low_op_pressure" in names
    assert "endurance_short" in names


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

    assert cmd[:4] == ["python", "run_sim.py", "--gc_policy", "greedy"]
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
