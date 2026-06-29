# SSD FTL/GC White-box Validation Lab

SSD controller firmware 내부의 FTL/Garbage Collection 동작을 Python으로 단순화해서 모델링하고, 여러 정책과 workload를 같은 조건에서 실행/검증/분석하는 작은 white-box validation lab입니다.

이 저장소는 원래 학부 SSD GC 연구 프로젝트에서 출발했지만, 지금은 **SSD validation / test engineering 포트폴리오**에 맞게 정리하고 있습니다. 핵심은 "새 GC 알고리즘 하나를 주장하는 것"이 아니라, firmware/FTL 내부 동작을 white-box model로 만들고, stress workload를 만들고, metric과 invariant로 결과를 검증하고, 보고서로 해석하는 흐름을 보여주는 것입니다.

## 보여주는 것

- Page-mapped out-of-place write 동작
- `VALID` / `INVALID` / `FREE` page state 관리
- `LPN -> PPN` mapping과 reverse mapping
- overwrite, TRIM, GC, valid-page migration, block erase 흐름
- Greedy, CB, BSGC, ATCB, RE50315-style, COTA GC policy 비교
- WAF, GC count, free block/page, wear_avg, wear_std, wear_max metric 수집
- random update, hot/cold skew, TRIM, burst, phase workload 생성
- manifest, validation matrix, report, insight miner 기반 재현 가능한 실험 흐름
- AI-ready 분석 도구: scorecard, anomaly, Pareto front, next sweep, LLM narrative packet

## 포트폴리오 포지셔닝

이 프로젝트는 실제 SSD firmware가 아닙니다. 더 정확히는:

> SSD controller firmware 안에서 일어나는 FTL/GC/TRIM 핵심 lifecycle을 Python simulator로 단순화하고, validation workflow로 검증하는 white-box modeling 프로젝트입니다.

SSD controller / NAND / storage validation 직무 관점에서 보여주고 싶은 역량은 다음입니다.

- FTL, GC, TRIM, WAF, wear leveling 개념 이해
- Python 기반 실험 자동화와 결과 분석
- invariant, sanity check, reproducibility, stress scenario 설계
- 실험 결과를 과장하지 않고 model scope 안에서 해석하는 태도

포트폴리오 evidence 예시:

- [GC Policy Validation Evidence](docs/portfolio_gc_evidence.md)
- [TRIM + GC Lifecycle Validation Evidence](docs/trim_gc_validation_evidence.md)
- [TRIM Evidence Matrix](docs/trim_evidence_matrix.md)
- [GC Policy Tier](docs/policy_tiers.md)
- [Results Folder Inventory](docs/results_inventory.md)

## 단기 방향

단기 목표는 "GC 하나"보다 넓은 **FTL 내부 white-box model**을 구축하는 것입니다. 지금은 GC와 TRIM을 중심으로 잡았지만, 다음 확장은 SSD controller 내부에서 상태가 어떻게 변하고, 그 상태를 어떤 metric과 invariant로 검증할지에 맞춥니다.

추천 작업 순서:

1. FTL 내부 core 확장: allocator, free-space manager, mapping lifecycle, write path trace
2. wear leveling 확장: erase count metric을 넘어 wear-aware policy와 static/dynamic wear movement 검토
3. data temperature 모델 강화: workload가 준 hot/cold가 아니라 FTL이 관찰한 hot/cold metadata 구축
4. 확장된 FTL 위에서 GC/TRIM 업그레이드: TRIM 이후 reclaim, background GC, temperature/wear-aware GC 비교

현재 진행 상태:

- allocator/free-space manager 관측성 추가: host write와 GC migration의 page allocation, active block switch, reserve guard, all-invalid reclaim을 metric/trace로 기록
- mapping lifecycle 관측성 추가: map create/update, overwrite invalidation, TRIM unmap, GC remap, mapping/reverse-map integrity snapshot을 metric/trace로 기록
- static wear-leveling 1차 모델 추가: low-wear/valid-heavy block의 valid page를 이동한 뒤 source block을 erase하고, moved pages / wear spread / skip reason을 metric/trace로 기록
- wear-leveling on/off 비교 matrix 추가: 같은 workload에서 WAF 비용, wear spread, moved pages, skip reason을 비교

우선 챙길 만한 내부 모델링 축:

- address translation: `LPN -> PPN` mapping, reverse mapping, unmapped state, stale mapping 검증
- free-space manager: free block/page pool, GC trigger, over-provisioning 압력
- write path: out-of-place write, overwrite invalidation, write burst/phase 변화
- erase/wear model: block erase count, wear spread, wear leveling policy
- background work: foreground GC와 background GC의 cadence, host write와의 간섭
- data temperature: hot/cold classification, locality 변화, temperature-aware victim selection
- lifecycle events: TRIM/deallocate 이후 invalidation, GC reclaim, reclaim lag
- reliability proxy: bad block, read disturb, retention, ECC 같은 실제 물리 현상은 단순 proxy로만 단계적으로 검토

SATA, NVMe, PCIe는 이 저장소의 단기 중심 범위가 아닙니다. 이들은 host와 SSD controller를 잇는 interface/protocol layer라서, 여기서는 "host command가 FTL에 들어온 뒤 내부 상태가 어떻게 변하는가"까지만 단순 입력으로 둡니다. queue depth, NVMe command-level test, PCIe path, OS/filesystem 영향, fio 실측, device telemetry는 별도의 SSD mini lab track이 더 잘 맞습니다.

장기적으로는 이 white-box FTL modeling track과 실제 장비를 관찰하는 black-box SSD mini lab track을 하나의 SSD validation portfolio로 묶는 것이 목표입니다. 다만 지금은 두 track을 분리해서 각각의 증거와 해석을 충분히 쌓은 뒤 연결합니다.

## 디렉토리 구조

```text
.
├── ssd_gc_lab/              # 시뮬레이터 본체 패키지
│   ├── config.py            # 시뮬레이션 geometry, capacity ratio, GC threshold
│   ├── models.py            # Page, Block, SSD 상태 모델
│   ├── simulator.py         # workload 실행과 GC trigger orchestration
│   ├── workload.py          # random update / hot-cold / TRIM / burst / phase workload 생성
│   ├── gc_algos.py          # GC victim-selection policy 모음
│   ├── policy_factory.py    # CLI policy 이름과 실제 함수 연결
│   ├── experiment_runner.py # 단일 run 실행, warmup, output, QC 공통 로직
│   ├── metrics.py           # run metric 수집과 summary CSV 기록
│   └── manifest.py          # 재현성 manifest 생성
├── tools/                   # CLI 실행기와 분석 도구
│   ├── run_sim.py
│   ├── experiments.py
│   ├── validation_matrix.py
│   ├── validation_report.py
│   ├── insight_miner.py
│   ├── policy_parameter_tuner.py
│   ├── adversarial_workload_search.py
│   └── llm_report_narrator.py
├── configs/                 # scenario/config 파일
├── docs/                    # test plan, portfolio evidence
├── tests/                   # pytest validation suite
└── results/                 # 로컬 생성 결과물. GitHub에는 올리지 않음
```

## 빠른 실행

필요 패키지 설치:

```bash
pip install pandas matplotlib pytest pyyaml
```

테스트 실행:

```bash
pytest -q
```

단일 smoke simulation:

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

작은 policy matrix 실행:

```bash
python tools/validation_matrix.py ^
  --profile quick ^
  --policies greedy,cb,age_stale,bsgc,atcb,re50315,cota ^
  --seeds 41 ^
  --out_dir results/final_clean ^
  --jobs 3 ^
  --skip_existing
```

matrix 결과 report 생성:

```bash
python tools/validation_report.py ^
  --base_dir results/final_clean
```

## AI-ready 분석 흐름

실험 결과를 만든 뒤, 다음 도구들로 policy trade-off와 다음 실험 후보를 뽑을 수 있습니다.

결과 insight 추출:

```bash
python tools/insight_miner.py ^
  --base_dir results/final_clean ^
  --out_dir results/insights
```

policy parameter sweep:

```bash
python tools/policy_parameter_tuner.py ^
  --policy cota ^
  --param_grid "cota_alpha=0.45,0.55;cota_delta=0.05,0.15" ^
  --out_dir results/policy_tuning
```

정책을 힘들게 하는 workload 조건 탐색:

```bash
python tools/adversarial_workload_search.py ^
  --policy cota ^
  --baseline_policy greedy ^
  --objective waf_gap ^
  --out_dir results/adversarial_workloads
```

LLM에 넣기 좋은 narrative packet 생성:

```bash
python tools/llm_report_narrator.py ^
  --insight_dir results/insights ^
  --out_dir results/llm_narrative
```

여기서 AI는 metric을 직접 만들어내는 역할이 아니라, deterministic script가 뽑은 CSV/Markdown 근거를 사람이 읽기 쉽게 설명하는 역할로 두는 것이 목표입니다.

## 주요 Metric

`ssd_gc_lab/metrics.py`가 만드는 핵심 output row는 다음을 포함합니다.

- `host_writes`, `device_writes`, `waf`
- `gc_count`, `gc_avg_s`
- `free_pages`, `free_blocks`
- `valid_pages`, `invalid_pages`, `trimmed_pages`
- `wear_min`, `wear_max`, `wear_avg`, `wear_std`, `wear_spread`
- `wear_leveling_count`, `wear_leveling_moved_pages`, `wear_leveling_allocations`, `wear_leveling_skipped_*`
- policy, seed, workload ratio, GC trigger, burst/phase setting, COTA weight 등 실험 metadata

이 metric들은 단순 성능 숫자가 아니라 validation signal입니다. 예를 들어:

- host write가 있는 run에서 `WAF >= 1`이어야 함
- `valid + invalid + free == total_pages`여야 함
- mapping과 reverse mapping은 overwrite, TRIM, GC 이후에도 일관되어야 함
- wear_std가 낮으면 erase count가 더 고르게 분포된 것으로 해석할 수 있음

## Workload 조건

현재 workload layer가 다루는 축은 다음입니다.

- `update_ratio`: overwrite/update 압력
- `hot_ratio`, `hot_weight`: hot/cold skew
- `trim_ratio`: delete/deallocate 압력
- `user_capacity_ratio`: over-provisioning 압력
- `warmup_fill`: 측정 전 SSD를 얼마나 채울지
- `bg_gc_every`: background GC cadence
- `gc_free_block_threshold`: free block 기반 GC trigger threshold
- `burst_length`, `burst_ratio`: 짧은 update-heavy burst
- `phase_pattern`: steady, bulk/update/TRIM phase, rocksdb-like phase shift
- `trim_locality`, `trim_burst_length`, `trim_burst_interval`: TRIM target locality와 periodic TRIM burst

## GC Policies Under Test

policy 비교는 COTA 하나를 증명하는 방향이 아니라, 보편 baseline과 custom policy를 같은 workload에서 비교하는 tier 구조로 진행합니다. 자세한 정리는 [GC Policy Tier](docs/policy_tiers.md)에 둡니다.

Tier 1 - core baseline:

- `greedy`: invalid page가 가장 많은 block 선택
- `cb`: invalid ratio와 wear-derived proxy를 쓰는 단순 cost-benefit baseline

Tier 2 - balanced / reference-style baseline:

- `age_stale`: invalid page 수와 block age를 함께 보는 age-staleness baseline
- `bsgc`: invalid ratio와 relative wear를 함께 보는 balance policy
- `atcb`: age / temperature / cost / balance scoring baseline
- `re50315`: invalid / age / wear 기반 lightweight baseline

Tier 3 - custom policy:

- `cota`: invalid ratio, coldness, age, wear를 섞은 custom heuristic

기본 matrix policy set은 `greedy,cb,age_stale,bsgc,atcb,re50315,cota`입니다. COTA는 production-ready firmware claim이 아니라, 검증 대상 policy 중 하나입니다. 이 프로젝트의 가치는 COTA 자체보다 여러 policy를 같은 조건에서 실행하고, trade-off를 검증/해석하는 framework에 있습니다.

## Local Heavy-Run Workflow

짧은 smoke run과 unit test는 Codex가 바로 돌려 코드와 QC를 확인합니다. 시간이 오래 걸리는 full matrix, 많은 seed, policy tuning sweep은 사용자가 로컬에서 실행하고 결과 파일을 첨부하는 방식으로 진행합니다.

사용자가 돌려서 공유하면 좋은 파일:

- `validation_summary.csv`
- `validation_report.md`
- `matrix_manifest.json`
- 필요할 때 각 run의 `manifest.json`

Codex는 이 결과를 받아 policy trade-off, anomaly, 다음 workload 후보, README/포트폴리오 문장 정리를 맡습니다.

## 현재 Model Scope

이 프로젝트는 controlled simulator이며 실제 SSD 성능 예측기가 아닙니다.

현재 포함하는 것:

- 단일 device의 simplified page-mapped FTL model
- out-of-place write
- LPN/PPN mapping과 reverse mapping
- TRIM invalidation
- GC victim selection과 valid-page migration
- erase_count 기반 wear proxy

현재 생략하는 것:

- SATA/NVMe command set과 protocol-level behavior
- PCIe/NVMe queueing model
- PCIe link, driver, OS/filesystem path 영향
- channel / die / plane parallelism
- SLC cache
- firmware scheduling의 실제 복잡도
- ECC, read disturb, retention, bad block management
- 실제 NAND latency와 물리 cell degradation

따라서 결과는 이 simulator model 내부의 상대 비교로 해석해야 합니다. 실제 SSD validation은 hardware trace, NVMe command-level test, firmware instrumentation, vendor-specific telemetry가 필요합니다.

## 한 줄 요약

이 프로젝트는 simplified SSD firmware/FTL 내부 동작을 white-box로 모델링하고, workload stress와 invariant test, reproducible report, AI-ready analysis로 검증하는 작은 SSD validation lab입니다.