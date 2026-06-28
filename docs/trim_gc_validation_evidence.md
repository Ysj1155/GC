# TRIM + GC Lifecycle Validation Evidence

이 문서는 이 프로젝트에서 TRIM을 GC와 분리된 FTL lifecycle 이벤트로 검증하기 위한 기준 문서입니다.

## 검증 질문

TRIM/deallocate 요청이 들어왔을 때, 시뮬레이터는 logical mapping을 안전하게 무효화하고, 이후 GC가 회수할 수 있는 invalid page를 정확히 만들고 있는가?

## 모델 정의

이 프로젝트에서 TRIM은 host가 FTL에게 보내는 deallocate hint입니다.

```text
host / OS delete
-> TRIM(LPN)
-> remove LPN -> PPN mapping
-> mark old physical page INVALID
-> remove reverse mapping
-> later GC reclaims the block
```

TRIM은 다음을 하지 않습니다.

- host write 증가 없음
- device write 증가 없음
- 즉시 block erase 없음
- GC event로 count하지 않음

## 구현된 관측 신호

SSD 모델은 다음 counter를 기록합니다.

| Metric | 의미 |
|---|---|
| `trim_ops` | 전체 TRIM 요청 수 |
| `trim_hits` | mapped LPN을 실제로 invalid 처리한 TRIM 수 |
| `trim_misses` | 현재 mapping이 없는 LPN에 대한 TRIM 수 |
| `retrim_count` | 이미 TRIM/unmapped 상태인 LPN에 반복 TRIM된 수 |
| `trim_invalidated_pages` | TRIM으로 VALID -> INVALID가 된 physical page 수 |
| `trimmed_pages` | block 단위로 집계된 TRIM invalidation 수 |

또한 `--trim_events_csv` 옵션으로 per-TRIM event log를 저장할 수 있습니다.

## 검증 계약

- TRIM hit는 mapping과 reverse mapping을 모두 제거해야 한다.
- TRIM hit는 해당 physical page를 `INVALID`로 바꿔야 한다.
- TRIM miss/retrim은 안전해야 하며 state를 깨뜨리면 안 된다.
- TRIM은 write counter를 증가시키지 않아야 한다.
- 같은 LPN에 나중에 write가 오면 새 physical page에 fresh mapping을 만들어야 한다.
- GC는 TRIM으로 생긴 invalid page를 나중에 회수할 수 있다.

## 실행 예시

```bash
python tools/run_sim.py ^
  --gc_policy greedy ^
  --ops 5000 ^
  --seed 41 ^
  --enable_trim ^
  --trim_ratio 0.10 ^
  --phase_pattern bulk_update_trim ^
  --trim_locality cold ^
  --trim_burst_length 16 ^
  --trim_burst_interval 256 ^
  --out_dir results/trim_lifecycle_smoke ^
  --out_csv summary.csv ^
  --trim_events_csv trim_events.csv ^
  --manifest_json manifest.json ^
  --qc strict
```

## 해석 방향

TRIM-heavy workload는 WAF 하나만 보면 안 됩니다. 다음을 같이 봐야 합니다.

- `trim_hits`가 충분히 발생했는가?
- `trim_misses`나 `retrim_count`가 비정상적으로 높은가?
- TRIM 이후 `invalid_pages`가 증가하고, GC가 free block을 회복하는가?
- 같은 policy가 TRIM locality에 따라 WAF/wear trade-off가 달라지는가?

## 다음 확장 후보

- TRIM-to-GC lag: TRIM hit 이후 해당 block이 GC로 erase되기까지 걸린 step 수
- before/after TRIM window 비교: TRIM burst 전후 free block, GC count, WAF 변화
- TRIM locality sensitivity: hot/cold/mixed TRIM에서 policy별 약점 비교
## TRIM-to-GC Lag 1차 분석

TRIM-to-GC lag는 TRIM hit로 invalid 처리된 physical page가 포함된 block이 이후 GC victim으로 erase되기까지 걸린 simulator step 수입니다.

현재 1차 구현은 다음 흐름을 기록합니다.

- TRIM event의 `old_block`, `old_page`, `step`을 기준으로 추적 대상을 만든다.
- GC event의 `victim`, `step`을 기준으로 같은 block의 다음 erase를 찾는다.
- `trim_gc_lag_csv`에는 per-TRIM reclaim 여부와 `lag_steps`를 기록한다.
- summary/report/insight에는 `trim_gc_lag_avg`, `trim_gc_lag_p95`, `trim_gc_lag_pending_count`, `trim_gc_reclaim_rate`를 포함한다.

이 값은 실제 지연 시간이 아니라 simulator step 기준의 lifecycle signal입니다. 따라서 해석할 때는 TRIM이 즉시 erase를 수행했다는 뜻이 아니라, TRIM invalidation이 나중에 GC 회수로 연결됐는지를 보는 지표로 사용해야 합니다.
## TRIM Window Analysis 1차 분석

TRIM window analysis는 TRIM burst 전후의 trace snapshot을 비교해 lifecycle 변화량을 관찰하는 분석입니다.

현재 1차 구현은 다음 흐름을 기록합니다.

- Simulator trace에 `free_blocks`, `valid_pages`, `invalid_pages`, `host_writes`, `trim_ops`를 추가한다.
- TRIM event에 workload 기준 `op_step`을 기록한다.
- 가까이 붙은 TRIM event를 하나의 window로 묶는다.
- window 시작 전과 종료 후 snapshot을 비교해 `invalid_pages_delta`, `free_blocks_delta`, `gc_count_delta`, `waf_delta`를 계산한다.
- `trim_windows_csv`에는 per-window before/after row를 기록한다.

이 값은 원인 확정이 아니라 관찰 신호입니다. 예를 들어 `gc_count_delta > 0`은 window 주변에서 GC가 있었다는 뜻이지, 모든 GC가 해당 TRIM 때문에 발생했다는 뜻은 아닙니다.
## TRIM Locality Sensitivity 1차 분석

TRIM locality sensitivity는 같은 workload 조건에서 TRIM target만 hot, cold, mixed로 바꿔 GC lifecycle metric의 차이를 비교하는 분석입니다.

현재 1차 구현은 다음 흐름을 사용합니다.

- `trim_locality_hot`, `trim_locality_cold`, `trim_locality_mixed` 시나리오를 validation matrix에 추가한다.
- 각 시나리오는 locality 외의 update ratio, trim ratio, burst, warmup, GC cadence를 맞춘다.
- report는 `trim_locality_sensitivity.csv`를 별도로 저장한다.
- 비교 metric은 WAF, GC count, wear spread, TRIM-to-GC lag, reclaim rate, TRIM window delta를 함께 본다.

이 분석은 특정 locality가 항상 좋다는 결론을 내기 위한 것이 아니라, locality 편향이 policy별 trade-off를 바꾸는지 확인하는 실험 축입니다.
