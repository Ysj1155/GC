# GC Policy Tier

이 프로젝트의 policy 비교는 COTA 하나를 증명하는 방향이 아니라, 여러 victim-selection baseline을 같은 workload와 metric으로 비교하는 validation workflow입니다.

## Tier 1: core baseline

- `greedy`: invalid page가 가장 많은 block을 고르는 가장 단순한 기준입니다. WAF와 GC 회수 효율의 기준선으로 둡니다.
- `cb`: invalid ratio와 wear-derived proxy를 쓰는 simplified cost-benefit baseline입니다. greedy보다 policy feature를 하나 더 넣은 기준선입니다.

## Tier 2: balanced / reference-style baseline

- `age_stale`: invalid page 수와 block age를 곱해 오래되고 stale한 block을 선호합니다.
- `bsgc`: invalid ratio와 relative wear를 함께 보는 balance policy입니다.
- `atcb`: age, temperature, cost, balance feature를 섞은 경량 baseline입니다.
- `re50315`: invalid, age, wear 기반 reference-style lightweight baseline입니다.

## Tier 3: custom policy

- `cota`: invalid ratio, coldness, age, wear를 섞은 custom heuristic입니다.

COTA는 production-ready firmware claim이 아니라 비교 대상 중 하나입니다. 포트폴리오 관점의 핵심은 custom policy 자체보다, 서로 다른 policy를 같은 조건에서 돌리고 trade-off를 해석하는 실험 체계입니다.

## Default matrix policy set

기본 validation matrix는 다음 순서로 실행합니다.

```text
greedy,cb,age_stale,bsgc,atcb,re50315,cota
```

이 순서는 단순 baseline에서 시작해 feature가 많은 baseline으로 넓히고, 마지막에 custom policy를 비교 대상으로 붙이는 흐름입니다.

## Local heavy-run workflow

짧은 smoke run은 Codex가 돌려서 코드와 QC를 확인합니다. 시간이 오래 걸리는 full matrix, 많은 seed, policy tuning sweep은 사용자가 로컬에서 실행하고 결과 파일을 첨부하는 방식으로 진행합니다.

사용자가 돌려서 공유하면 좋은 파일:

- `validation_summary.csv`
- `validation_report.md`
- `matrix_manifest.json`
- 필요할 때 각 run의 `manifest.json`

Codex는 이 결과를 받아 policy trade-off, anomaly, 다음 workload 후보, README/포트폴리오 문장 정리를 맡습니다.
## Split full matrix commands

4-core local machine에서는 full matrix를 한 번에 350 runs로 돌리기보다 조각내는 편이 안전합니다.

Core baseline:

```powershell
python tools/validation_matrix.py --profile full --policies greedy,cb,age_stale --outdir results/full_core_jobs3 --jobs 3 --skip_existing --qc strict
```

Balanced/custom baseline:

```powershell
python tools/validation_matrix.py --profile full --policies bsgc,atcb,re50315,cota --outdir results/full_balanced_jobs3 --jobs 3 --skip_existing --qc strict
```

TRIM-heavy scenarios only:

```powershell
python tools/validation_matrix.py --profile full --scenarios trim_burst,delete_after_bulk_load,trim_locality_hot,trim_locality_cold,trim_locality_mixed --outdir results/full_trim_jobs3 --jobs 3 --skip_existing --qc strict
```

Wear-leveling scenarios only:

```powershell
python tools/validation_matrix.py --profile full --scenarios wear_leveling_off,wear_leveling_on --outdir results/full_wear_jobs3 --jobs 3 --skip_existing --qc strict
```

진행률 확인:

```powershell
$count = (Get-ChildItem .\results\full_core_jobs3 -Recurse -Filter manifest.json).Count; "$count completed"
```

각 조각이 끝나면 `tools/validation_report.py --base_dir <result-dir>`로 개별 report를 만들고, 필요한 경우 summary CSV를 합쳐서 전체 해석합니다.