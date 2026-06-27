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