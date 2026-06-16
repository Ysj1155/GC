# SSD GC Validation Lab

Python-based SSD FTL/GC validation mini-lab for experimenting with garbage collection policies, workload stress patterns, endurance metrics, and reproducible test reports.

This repository started as an undergraduate SSD garbage-collection research project. It is being reshaped into a portfolio project for SSD validation / test engineering roles: the emphasis is not only on proposing a GC heuristic, but on showing how SSD behavior can be modeled, tested, stressed, measured, and reported with automation.

## What This Demonstrates

- Page-mapped out-of-place write behavior with VALID / INVALID / FREE page states
- GC victim selection policies including Greedy, CB, BSGC, ATCB, RE50315-style, and COTA
- Workload generation for random update, hot/cold skew, TRIM, and repeated seed runs
- Core validation metrics such as WAF, GC count, free-space state, wear_avg, wear_std, and wear_max
- CLI-based experiment automation with CSV summaries and analysis scripts
- A growing pytest-based validation suite for FTL/GC invariants
- A compact portfolio evidence note showing condition -> execution -> WAF / wear results -> interpretation

## Portfolio Positioning

For SSD controller / NAND / storage validation roles, this project is intended to show:

- Understanding of SSD internal concepts: FTL mapping, garbage collection, write amplification, TRIM, and wear leveling
- Python automation skills for test execution, metric collection, and result analysis
- Test-engineering thinking: invariants, sanity checks, reproducibility, stress workloads, and failure-oriented debugging
- Clear separation between model assumptions and real-device behavior

Portfolio evidence example:

- [GC Policy Validation Evidence](docs/portfolio_gc_evidence.md)

## Repository Layout

```text
.
├── config.py                 # Simulation geometry, capacity ratio, latency assumptions
├── models.py                 # Page, block, and SSD state model
├── simulator.py              # Workload execution and GC trigger orchestration
├── gc_algos.py               # GC victim-selection policies
├── policy_factory.py          # CLI/experiment policy wiring and parameter binding
├── experiment_runner.py       # Shared one-run execution, warmup, output, and QC helpers
├── workload.py               # Random update / hot-cold / TRIM workload generation
├── metrics.py                # Run metrics and summary CSV writing
├── run_sim.py                # Single-run CLI entry point
├── experiments.py            # Grid / scenario experiment runner
├── validation_matrix.py       # Portfolio-oriented stress/endurance matrix runner
├── validation_report.py       # Markdown/CSV validation report generator
├── analyze_results.py        # CSV merge and plotting helper
├── summarize.py              # Policy-level summary helper
├── docs/
│   ├── portfolio_gc_evidence.md # Condition -> execution -> result -> interpretation example
│   └── test_plan.md          # Validation-oriented test plan
└── tests/                    # Pytest validation suite
```

## Quickstart

Install the runtime dependencies:

```bash
pip install pandas matplotlib pytest
```

Run a smoke simulation:

```bash
python run_sim.py ^
  --gc_policy cota ^
  --ops 20000 ^
  --seed 42 ^
  --out_dir results/smoke ^
  --out_csv summary.csv ^
  --manifest_json manifest.json ^
  --qc strict
```

Run the validation tests:

```bash
pytest -q
```

Run a small policy matrix:

```bash
python validation_matrix.py ^
  --profile quick ^
  --policies greedy,cb,bsgc,cota ^
  --seeds 41 ^
  --out_dir results/final_clean
```

Use `--dry_run` first to preview the commands without running simulations.

## Validation Metrics

The core output row is produced by `metrics.py` and includes:

- `host_writes`, `device_writes`, `waf`
- `gc_count`, `gc_avg_s`
- `free_pages`, `free_blocks`
- `valid_pages`, `invalid_pages`, `trimmed_pages`
- `wear_min`, `wear_max`, `wear_avg`, `wear_std`
- experiment metadata such as policy, seed, workload ratios, and COTA weights

The project treats these metrics as validation signals. For example, WAF should not fall below 1, page-state totals should match physical capacity, and mapping / reverse-map consistency must survive overwrite, TRIM, and GC operations.

## Reproducibility Manifests

`run_sim.py` can write a manifest JSON next to each experiment result:

```bash
python run_sim.py ^
  --gc_policy greedy ^
  --ops 20000 ^
  --seed 41 ^
  --out_dir results/final_clean/greedy_41 ^
  --out_csv summary.csv ^
  --manifest_json manifest.json ^
  --qc strict
```

The manifest captures the command, parameters, output artifact paths, Python version, git branch/commit, dirty working-tree flag, and the final metric row. This is the main provenance artifact for portfolio-grade validation runs.

## Validation Matrix

`validation_matrix.py` provides portfolio-oriented scenario presets:

- `random_update_stress`: high-update random write pressure with hot/cold skew
- `trim_burst`: update workload mixed with TRIM pressure
- `low_op_pressure`: low over-provisioning pressure to force GC decisions
- `endurance_short`: longer high-update run for wear distribution checks

Example:

```bash
python validation_matrix.py ^
  --profile quick ^
  --policies greedy,cota ^
  --seeds 41,42 ^
  --out_dir results/final_clean
```

Each matrix entry gets its own output directory with `summary.csv` and `manifest.json`. The matrix root also gets `matrix_manifest.json`, which records the scenario list and every generated command.

Generate a portfolio report from the matrix output:

```bash
python validation_report.py ^
  --base_dir results/final_clean
```

This creates `validation_report.md`, `validation_summary.csv`, and `validation_runs.csv`.

## GC Policies Under Test

- `greedy`: selects the block with the most invalid pages
- `cb`: simplified cost-benefit style policy using invalid ratio and a wear-derived proxy
- `bsgc`: balances invalid ratio and relative wear
- `atcb`: age / temperature / cost / balance scoring baseline
- `re50315`: lightweight age-staleness inspired baseline
- `cota`: custom heuristic combining invalid ratio, coldness, age, and wear

COTA is kept as a policy under test, not as a claim of production-ready SSD firmware. The value of the project is the validation framework around these policies.

## Current Model Scope

This is a controlled simulator, not a real SSD performance predictor.

- Single-device, simplified page-mapped FTL model
- Constant-cost latency assumptions
- No PCIe/NVMe queueing model
- No channel / die / plane parallelism
- No SLC cache or firmware scheduling model
- Wear is represented with erase counts, not physical cell degradation

The results should be interpreted as relative behavior inside this model. Real-device validation would require hardware traces, NVMe command-level tests, firmware instrumentation, and vendor-specific telemetry.

## Roadmap

- Add invariant-focused pytest coverage for mapping, TRIM, GC, WAF, and reproducibility
- Expand validation manifests into matrix-level reports
- Add threshold-based pass/fail reports on top of stress workload presets
- Add plots for final validation reports
- Add SMART-like summary metrics and threshold-based pass/fail reports
- Produce a clean `results/final_clean/` validation report for portfolio review

## One-Line Summary

This is a small SSD validation lab: it models simplified FTL/GC behavior, runs policy and workload tests, checks correctness signals, and turns the results into reproducible validation artifacts.
