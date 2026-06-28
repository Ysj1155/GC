# TRIM Evidence Matrix

이 문서는 TRIM lifecycle validation의 4번 단계인 multi-policy, multi-seed evidence run 기록입니다.

## 실행 범위

실행 명령:

```bash
python tools/validation_matrix.py ^
  --profile quick ^
  --policies greedy,cb,bsgc,cota ^
  --seeds 41,42,43 ^
  --scenarios trim_burst,delete_after_bulk_load,trim_locality_hot,trim_locality_cold,trim_locality_mixed ^
  --out_dir results/trim_evidence_matrix
```

조건:

- scenarios: `trim_burst`, `delete_after_bulk_load`, `trim_locality_hot`, `trim_locality_cold`, `trim_locality_mixed`
- policies: `greedy`, `cb`, `bsgc`, `cota`
- seeds: `41`, `42`, `43`
- total runs: `60`
- QC: all runs passed strict QC

## 생성 산출물

대표 산출물:

- `results/trim_evidence_matrix/validation_report.md`
- `results/trim_evidence_matrix/validation_runs.csv`
- `results/trim_evidence_matrix/validation_summary.csv`
- `results/trim_evidence_matrix/trim_locality_sensitivity.csv`
- `results/trim_evidence_matrix/insights/insights.md`
- `results/trim_evidence_matrix/llm_narrative/narrative_report.md`

각 run directory에는 다음 artifact가 남습니다.

- `manifest.json`
- `summary.csv`
- `gc_events.csv`
- `trim_events.csv`
- `trim_gc_lag.csv`
- `trim_windows.csv`

## 주요 관찰

주의: 아래 내용은 simulator-relative observation입니다. 실제 SSD 성능 claim이 아닙니다.

### WAF 기준

이번 run에서 평균 WAF가 낮았던 조건:

| scenario | policy | runs | waf_mean | wear_std_mean | gc_count_mean |
|---|---:|---:|---:|---:|---:|
| `trim_burst` | `greedy` | 3 | 1.000037 | 0.662553 | 196.333333 |
| `trim_burst` | `cota` | 3 | 1.001388 | 0.559192 | 197.000000 |
| `trim_burst` | `bsgc` | 3 | 1.004164 | 0.515348 | 198.333333 |
| `trim_burst` | `cb` | 3 | 1.006604 | 0.515824 | 199.333333 |

해석:

- `trim_burst`에서는 `greedy`가 WAF만 보면 가장 낮았습니다.
- 대신 `greedy`의 wear spread는 다른 policy보다 높게 나와, WAF와 wear balance trade-off를 보여줍니다.

### Wear spread 기준

이번 run에서 평균 wear spread가 낮았던 조건:

| scenario | policy | runs | waf_mean | wear_std_mean | trim_gc_lag_avg_mean |
|---|---:|---:|---:|---:|---:|
| `trim_locality_cold` | `bsgc` | 3 | 1.019894 | 0.491354 | 6158.661294 |
| `trim_locality_cold` | `cb` | 3 | 1.019894 | 0.491354 | 6158.661294 |
| `trim_locality_hot` | `bsgc` | 3 | 1.014201 | 0.492549 | 6017.497926 |
| `trim_locality_hot` | `cb` | 3 | 1.014201 | 0.492549 | 6017.497926 |

해석:

- `cb`와 `bsgc`는 이번 simplified model 조건에서 locality scenarios의 결과가 거의 동일하게 나타났습니다.
- WAF만 보면 `greedy`가 강한 구간이 있지만, wear spread는 `cb`/`bsgc` 쪽이 더 낮게 나타나는 구간이 있습니다.

### TRIM locality 기준

`trim_locality_sensitivity.csv` 요약에서 보인 방향:

- hot TRIM은 cold TRIM보다 대체로 WAF와 GC count가 낮았습니다.
- cold TRIM은 reclaim rate가 hot/mixed보다 낮은 조합이 많았습니다.
- mixed는 중간 baseline처럼 동작했지만, policy별 wear spread 차이는 남았습니다.

예시:

| policy | locality | waf_mean | gc_count_mean | trim_gc_reclaim_rate_mean |
|---|---:|---:|---:|---:|
| `greedy` | `hot` | 1.010517 | 148.666667 | 0.352647 |
| `greedy` | `cold` | 1.016915 | 151.000000 | 0.324498 |
| `greedy` | `mixed` | 1.011284 | 148.666667 | 0.341648 |

## 다음 해석 방향

다음 단계에서 보면 좋은 질문:

- `cb`와 `bsgc`가 왜 locality scenarios에서 같은 결과를 내는가?
- `greedy`의 낮은 WAF와 높은 wear spread trade-off를 어떻게 설명할 것인가?
- `cota`가 WAF와 wear 사이에서 어느 시나리오에 강한가?
- TRIM locality가 lag/reclaim rate에는 영향을 주지만, window delta에는 어떤 차이를 만드는가?

이 패킷은 다음 분석의 출발점입니다. 결론을 내릴 때는 반드시 `validation_summary.csv`, `trim_locality_sensitivity.csv`, `insights.md`를 함께 봐야 합니다.
