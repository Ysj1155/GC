# SSD GC Validation Lab

Python-based SSD FTL/GC validation mini-lab for experimenting with garbage collection policies, workload stress patterns, endurance metrics, reproducible reports, and AI-ready experiment analysis.

This repository started as an undergraduate SSD garbage-collection research project. It is being reshaped into a portfolio project for SSD validation / test engineering roles: the emphasis is not only on proposing a GC heuristic, but on showing how SSD behavior can be modeled, tested, stressed, measured, and reported with automation.

## What This Demonstrates

- Page-mapped out-of-place write behavior with VALID / INVALID / FREE page states
- GC victim selection policies including Greedy, CB, BSGC, ATCB, RE50315-style, and COTA
- Workload generation for random update, hot/cold skew, TRIM, burst, phase, and GC-trigger stress
- Core validation metrics such as WAF, GC count, free-space state, wear_avg, wear_std, and wear_max
- CLI-based experiment automation with CSV summaries, manifests, reports, and analysis scripts
- Pytest-based validation coverage for FTL/GC invariants and analysis helpers
- AI-ready analysis outputs for scorecards, anomalies, Pareto fronts, next sweeps, and narrative packets

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
├── ssd_gc_lab/              # Simulator package: model, workload, policies, metrics, manifests
│   ├── config.py            # Simulation geometry, capacity ratio, latency assumptions
│   ├── models.py            # Page, block, and SSD state model
│   ├── simulator.py         # Workload execution and GC trigger orchestration
│   ├── workload.py          # Random update / hot-cold / TRIM / burst / phase workload generation
│   ├── gc_algos.py          # GC victim-selection policies
│   ├── policy_factory.py    # Policy wiring and parameter binding
│   ├── experiment_runner.py # Shared one-run execution, warmup, output, and QC helpers
│   ├── metrics.py           # Run metrics and summary CSV writing
│   └── manifest.py          # Reproducibility manifest generation
├── tools/                   # CLI entry points and analysis utilities
│   ├── run_sim.py
│   ├── experiments.py
│   ├── validation_matrix.py
│   ├── validation_report.py
│   ├── insight_miner.py
│   ├── policy_parameter_tuner.py
│   ├── adversarial_workload_search.py
│   └── llm_report_narrator.py
├── configs/                 # Scenario/config files
├── docs/                    # Test plan and portfolio evidence
├── tests/                   # Pytest validation suite
└── results/                 # Generated local outputs, ignored for portfolio cleanliness
```

## Quickstart

Install the runtime dependencies:

```bash
pip install pandas matplotlib pytest
```

Run the validation tests:

```bash
pytest -q
```

Run a smoke simulation:

```bash
python tools/run_sim.py ^
  --gc_policy cota ^
  --ops 20000 ^
  --seed 42 ^
  --out_dir results/smoke ^
  --out_csv summary.csv ^
  --manifest_json manifest.json ^
  --qc strict
```

Run a small policy matrix:

```bash
python tools/validation_matrix.py ^
  --profile quick ^
  --policies greedy,cb,bsgc,cota ^
  --seeds 41 ^
  --out_dir results/final_clean
```

Generate a validation report from matrix output:

```bash
python tools/validation_report.py ^
  --base_dir results/final_clean
```

## AI-Ready Analysis Flow

After producing experiment results, mine the outputs for trade-offs and follow-up candidates:

```bash
python tools/insight_miner.py ^
  --base_dir results/final_clean ^
  --out_dir results/insights
```

Run a small policy parameter sweep:

```bash
python tools/policy_parameter_tuner.py ^
  --policy cota ^
  --param_grid "cota_alpha=0.45,0.55;cota_delta=0.05,0.15" ^
  --out_dir results/policy_tuning
```

Search for workload conditions that stress a policy:

```bash
python tools/adversarial_workload_search.py ^
  --policy cota ^
  --baseline_policy greedy ^
  --objective waf_gap ^
  --out_dir results/adversarial_workloads
```

Create an LLM-ready narrative packet from insight outputs:

```bash
python tools/llm_report_narrator.py ^
  --insight_dir results/insights ^
  --out_dir results/llm_narrative
```

The AI-facing tools deliberately keep the calculation deterministic. A future LLM layer should explain these CSV/Markdown outputs instead of inventing new measurements.

## Validation Metrics

The core output row is produced by `ssd_gc_lab/metrics.py` and includes:

- `host_writes`, `device_writes`, `waf`
- `gc_count`, `gc_avg_s`
- `free_pages`, `free_blocks`
- `valid_pages`, `invalid_pages`, `trimmed_pages`
- `wear_min`, `wear_max`, `wear_avg`, `wear_std`
- experiment metadata such as policy, seed, workload ratios, GC trigger settings, burst/phase settings, and COTA weights

The project treats these metrics as validation signals. For example, WAF should not fall below 1, page-state totals should match physical capacity, and mapping / reverse-map consistency must survive overwrite, TRIM, and GC operations.

## Workload Dimensions

The workload layer currently supports:

- `update_ratio`: overwrite/update pressure
- `hot_ratio` and `hot_weight`: hot/cold skew
- `trim_ratio`: delete/deallocate pressure
- `user_capacity_ratio`: over-provisioning pressure
- `warmup_fill`: preconditioned fill level before measurement
- `bg_gc_every`: background GC cadence
- `gc_free_block_threshold`: low-free-space GC trigger threshold
- `burst_length` and `burst_ratio`: short update-heavy bursts
- `phase_pattern`: steady, bulk/update/TRIM phase, or rocksdb-like phase shifts
- `trim_locality`, `trim_burst_length`, and `trim_burst_interval`: TRIM target locality and periodic TRIM bursts

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

## One-Line Summary

This is a small SSD validation lab: it models simplified FTL/GC behavior, runs policy and workload tests, checks correctness signals, and turns the results into reproducible validation and analysis artifacts.