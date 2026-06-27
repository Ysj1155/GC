# SSD GC Validation Lab

SSD의 FTL/Garbage Collection 동작을 Python으로 단순화해서 모델링하고, 여러 GC 정책을 같은 조건에서 실행/검증/분석하는 작은 validation lab입니다.

이 저장소는 원래 학부 SSD GC 연구 프로젝트에서 출발했지만, 지금은 **SSD validation / test engineering 포트폴리오**에 맞게 정리하고 있습니다. 핵심은 "새 GC 알고리즘 하나를 주장하는 것"이 아니라, SSD 내부 동작을 모델링하고, stress workload를 만들고, metric과 invariant로 결과를 검증하고, 보고서로 해석하는 흐름을 보여주는 것입니다.

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

> SSD controller firmware 안에서 일어나는 FTL/GC/TRIM 핵심 동작을 Python simulator로 단순화하고, validation workflow로 검증하는 프로젝트입니다.

SSD controller / NAND / storage validation 직무 관점에서 보여주고 싶은 역량은 다음입니다.

- FTL, GC, TRIM, WAF, wear leveling 개념 이해
- Python 기반 실험 자동화와 결과 분석
- invariant, sanity check, reproducibility, stress scenario 설계
- 실험 결과를 과장하지 않고 model scope 안에서 해석하는 태도

포트폴리오 evidence 예시:

- [GC Policy Validation Evidence](docs/portfolio_gc_evidence.md)
- [TRIM + GC Lifecycle Validation Evidence](docs/trim_gc_validation_evidence.md)

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
  --policies greedy,cb,bsgc,cota ^
  --seeds 41 ^
  --out_dir results/final_clean
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
- `wear_min`, `wear_max`, `wear_avg`, `wear_std`
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

## 진행 중: TRIM Lifecycle Validation

다음 확장은 GC policy 비교를 넘어서, TRIM을 FTL lifecycle의 1급 검증 대상으로 올리는 것입니다.

```text
write / overwrite
-> stale physical pages 발생
-> TRIM / deallocate
-> mapping invalidation
-> GC victim selection
-> valid-page migration
-> block erase
-> metrics, invariants, report interpretation
```

작업 순서:

1. TRIM model contract 정리 - 완료
2. TRIM-focused workload scenario 추가 - 1차 완료
3. TRIM-specific metric/counter 추가 - 1차 완료
4. per-TRIM event log 추가 - 1차 완료
5. insight/report 도구에 TRIM-aware 분석 추가 - 1차 완료
6. `docs/trim_gc_validation_evidence.md` 포트폴리오 evidence 작성 - 완료

### 1. TRIM Model Contract

이 프로젝트에서 TRIM은 host가 FTL에게 보내는 deallocate hint로 정의합니다. 즉, host/OS가 "이 logical page는 더 이상 필요 없다"고 알려주는 신호입니다.

중요한 점:

- TRIM은 GC가 아님
- TRIM은 즉시 block erase를 수행하지 않음
- TRIM은 mapping을 무효화해서 나중에 GC가 block을 더 쉽게 회수하도록 도와줌

기대 동작:

- LPN이 현재 mapped 상태라면, TRIM은 `LPN -> PPN` mapping을 제거한다.
- 기존 physical page는 `INVALID`가 된다.
- 기존 PPN의 reverse mapping도 제거된다.
- TRIM은 `host_writes`나 `device_writes`를 증가시키지 않는다.
- 이미 unmapped 상태인 LPN을 다시 TRIM해도 안전해야 한다.
- 나중에 같은 LPN에 write가 오면 새 physical page를 할당하고 새 mapping을 만든다.
- TRIM으로 생긴 invalid page는 이후 GC가 회수할 수 있지만, TRIM 자체를 GC event로 세면 안 된다.

다음에 추가할 validation signal 후보:

- `trim_ops`
- `trim_hits`, `trim_misses`
- `retrim_count`
- `trim_invalidated_pages`
- TRIM-heavy scenario summary in `validation_report.py`
- TRIM-aware anomaly/trade-off analysis in `insight_miner.py`

## GC Policies Under Test

- `greedy`: invalid page가 가장 많은 block 선택
- `cb`: invalid ratio와 wear-derived proxy를 쓰는 단순 cost-benefit baseline
- `bsgc`: invalid ratio와 relative wear를 함께 보는 balance policy
- `atcb`: age / temperature / cost / balance scoring baseline
- `re50315`: age-staleness 기반 lightweight baseline
- `cota`: invalid ratio, coldness, age, wear를 섞은 custom heuristic

COTA는 production-ready firmware claim이 아니라, 검증 대상 policy 중 하나입니다. 이 프로젝트의 가치는 COTA 자체보다 여러 policy를 같은 조건에서 실행하고, trade-off를 검증/해석하는 framework에 있습니다.

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

- PCIe/NVMe queueing model
- channel / die / plane parallelism
- SLC cache
- firmware scheduling의 실제 복잡도
- ECC, read disturb, retention, bad block management
- 실제 NAND latency와 물리 cell degradation

따라서 결과는 이 simulator model 내부의 상대 비교로 해석해야 합니다. 실제 SSD validation은 hardware trace, NVMe command-level test, firmware instrumentation, vendor-specific telemetry가 필요합니다.

## 한 줄 요약

이 프로젝트는 simplified SSD FTL/GC/TRIM 동작을 모델링하고, workload stress와 invariant test, reproducible report, AI-ready analysis로 검증하는 작은 SSD validation lab입니다.