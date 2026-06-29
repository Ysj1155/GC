# Results Folder Inventory

`results/` 폴더는 2026-06-29 기준으로 evidence 중심으로 정리했습니다. 현재 최상위에는 보존할 evidence와 `_legacy_20260629` archive만 남기는 것을 목표로 합니다.

## Top-Level Evidence To Keep

- `trim_evidence_matrix`: TRIM MVP evidence package. `matrix_manifest.json`, `validation_report.md`, `validation_summary.csv`, 60 run manifest가 있습니다.
- `trim_locality_smoke`: TRIM locality sensitivity evidence. report/summary와 8 run manifest가 있습니다.
- `trim_window_smoke`: TRIM window evidence. report/summary와 1 run manifest가 있습니다.
- `trim_gc_lag_smoke`: TRIM-to-GC lag evidence. report/summary와 1 run manifest가 있습니다.
- `wear_leveling_matrix_smoke`: wear-leveling on/off comparison evidence. report/summary와 8 run manifest가 있습니다.
- `portfolio_evidence_20260616`: 예전 portfolio evidence snapshot입니다. 현재 문서 링크 확인 전까지 보존합니다.

## Archive Folder

Archived under `results/_legacy_20260629/`:

- `dry_run_smoke`: dry-run, smoke, development-check 결과입니다.
- `interrupted`: 중간에 멈춘 full matrix 조각입니다.
- `old_legacy`: 현재 white-box FTL/GC 방향과 직접 연결성이 약한 예전 COTA/tune/date-folder 결과입니다.
- `root_files`: 과거 root-level CSV/plot 파일입니다.

## Current Archive Counts

정리 직후 확인 기준:

- `dry_run_smoke`: 20 entries, 75 files, 14 run manifests
- `interrupted`: 3 entries, 46 files, 23 run manifests
- `old_legacy`: 21 entries, 69 files
- `root_files`: 7 files

## Cleanup Rule Going Forward

1. 결과가 README/docs evidence로 쓰이면 최상위에 둡니다.
2. 단일 smoke, dry-run, CLI wiring 확인용 결과는 `_legacy_YYYYMMDD/dry_run_smoke`로 보냅니다.
3. 중간에 멈춘 full matrix는 `_legacy_YYYYMMDD/interrupted`로 보냅니다.
4. 방향이 바뀐 오래된 결과는 `_legacy_YYYYMMDD/old_legacy`로 보냅니다.
5. 삭제는 archive 후 한 번 더 확인한 뒤 결정합니다.