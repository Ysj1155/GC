# SSD GC Policy Lab

페이지 단위 쓰기 / 블록 단위 소거라는 NAND 플래시의 제약 때문에, SSD는 **Garbage Collection(GC)** 없이는 계속 쓸 수 없다.  
하지만 GC는 **쓰기 증폭(WAF)**, **성능 저하(추가 read/program/erase)**, **마모 편중(수명 단축)** 같은 트레이드오프를 만들게 된다.

이 프로젝트는 **여러 GC victim 선택 정책**(Greedy / CB / BSGC / COTA / ATCB / RE50315 …)을 같은 조건에서 실행하고,  
결과를 **summary.csv로 자동 누적**한 뒤, **병합/요약/플로팅까지 한 번에 재현 가능한 실험 루프**를 제공하는 작은 실험실입니다.

## Context
This project started as an undergraduate capstone/thesis project and was later polished into a reproducible mini-lab for GC policy experiments.

---

## 이 레포에서 할 수 있는 것

- **단일 실험 실행** → `summary.csv`에 결과 1행 append (`run_sim.py`)
- **그리드/시나리오(YAML)/멀티시드 반복**으로 실험 묶음 실행 (`experiments.py`, `sweep.py`)
- 흩어진 `summary.csv`를 **자동 수집/병합** + 기본 플롯 생성 (`analyze_results.py`)
- 정책별 대표값(중앙값/사분위)으로 **한 장 요약 테이블** 생성 (`summarize.py`)
- 개념 그림(예: BSGC score heatmap) 같은 **설명용 플롯** 생성 (`plot_maker.py`)

---

## 설치

권장: Python 3.10+

필수 패키지:
- pandas (CSV 읽기/요약)
- matplotlib (플롯 생성)

선택:
- pyyaml (YAML 시나리오 실행 시)

```bash
pip install pandas matplotlib
# YAML 시나리오까지 쓸 거면
pip install pyyaml
```

---

## Quickstart

### 1) 단일 실행 (summary.csv append)

```bash
python run_sim.py ^
  --gc_policy cota ^
  --ops 200000 --seed 42 ^
  --out_dir results/run ^
  --out_csv summary.csv
```

- `results/run/summary.csv`에 1행이 추가됩니다.
- `--qc warn|strict`로 기본 무결성 체크(WAF/페이지 합/wear 범위 등)를 같이 돌릴 수 있습니다.

### 2) Sweep / Grid 실행 (여러 조합 자동)

```bash
python sweep.py ^
  --out_dir results/sweep ^
  --out_csv results/sweep/summary.csv ^
  --grid "gc_policy=greedy,cota,bsgc; update_ratio=0.5,0.8; seed=1,2" ^
  --repeat 1
```

- `grid`는 `키=값1,값2; 키2=...` 형태로 데카르트 곱을 생성합니다.

### 3) 결과 병합 + 기본 플롯 생성

```bash
python analyze_results.py ^
  --base results --merge-subdirs ^
  --out_csv results/_merged.csv ^
  --plots_dir results/_plots
```

- `results/` 하위의 모든 `summary.csv`를 찾아 병합합니다.
- 병합본에는 각 row의 출처를 추적하기 위해 `__source__` 컬럼이 추가됩니다.
- `results/_plots/`에 기본 플롯 PNG가 생성됩니다.

### 4) 정책별 요약(중앙값/사분위)

```bash
python summarize.py results/_merged.csv
```

- `results/comp/summary_by_policy.csv` 같은 형태로 요약 테이블을 생성합니다(스크립트 내부 경로 확인).

---

## 재현성

이 프로젝트에서 재현성은 누가 돌려도 같은 입력이면 같은 출력이 나온다를 목표로 합니다.

- **시드 고정:** `--seed`, `rng_seed`
- **장치/워크로드 조건:** `blocks`, `pages_per_block`, `user_capacity_ratio`, `update_ratio`, `hot_ratio`, `trim_ratio` …
- **정책 하이퍼파라미터:** COTA(α/β/γ/δ), cold bias, top-k …
- **결과 로그:** `summary.csv`에 메타+메트릭을 한 줄로 누적  
- **출처 추적:** 병합 시 `__source__`로 이 row가 어디서 나왔는지를 따라갈 수 있음

---

## 주요 스크립트/모듈 안내

### 실행(실험 생성)
- `run_sim.py`  
  단일 실행용 CLI. 실험 1회 실행 후 `summary.csv`에 append.
- `experiments.py`  
  그리드/시나리오(YAML)/멀티시드 반복을 포함한 “실험 러너”.
- `sweep.py`  
  `experiments.py`를 감싸는 얇은 wrapper(빠른 sweep 용도).

### 분석/시각화(결과 소비)
- `analyze_results.py`  
  흩어진 summary를 모아서 병합 + 기본 플롯 생성.
- `summarize.py`  
  병합된 CSV를 정책별 대표 통계로 축약.
- `plot_maker.py`  
  개념 설명용 플롯(예: BSGC score heatmap).

### 핵심 로직
- `config.py`  
  실험 입력(geometry / GC trigger / latency)을 한 곳에 모아 검증 + 파생값 계산.
- `models.py`  
  페이지 상태(FREE/VALID/INVALID) + 블록/SSD 동작(쓰기/GC/trim) 모델.
- `gc_algos.py`  
  victim 선택 정책 모음. 함수형 정책 `policy(blocks) -> victim_idx`.
- `workload.py`  
  워크로드 생성기(신규 write/업데이트, hot/cold, 선택적 TRIM).
- `simulator.py`  
  워크로드를 SSD 모델에 흘려보내는 실행 엔진(+ 옵션으로 BG-GC/trace).
- `metrics.py`  
  시뮬레이터/SSD 구현 차이를 흡수하면서 메트릭을 견고하게 추출하고 CSV로 저장.

---

## 결과 CSV 스키마(요약)

`metrics.py`가 기본적으로 채우는 대표 컬럼들:

- `host_writes`, `device_writes`, `waf`
- `gc_count`, `gc_avg_s`
- `free_pages`, `free_blocks`
- `wear_min`, `wear_max`, `wear_avg`, `wear_std`
- `valid_pages`, `invalid_pages`, `trimmed_pages`
- (선택) `transition_rate`, `reheat_rate` (분포 기반의 보수적 스냅샷 신호)

여기에 `run_sim.py / experiments.py`가 넣는 메타(예: `ops`, `seed`, `update_ratio`, `cota_alpha` …)가 합쳐져서 한 행을 구성합니다.

---

## 확장하는 법 (새 정책/새 메트릭)

### 새 GC 정책 추가
1) `gc_algos.py`에 함수 추가  
   - 입력: `blocks` (Block 리스트)  
   - 출력: victim index(int) 또는 None
2) `get_gc_policy()`에 이름 등록
3) `run_sim.py` / `experiments.py`의 `--gc_policy` choices에 이름 추가(원하면)

### 새 메트릭 추가
- `metrics.py::collect_run_metrics()`에 키를 추가하면 됩니다.  
- `append_summary_csv()`는 **기존 헤더 + 새 컬럼 자동 추가** 방식이라, 실험 중간에 컬럼이 늘어나도 데이터가 유지됩니다.

---

## 모델링 가정 / 한계

이 레포는 실험 비교가 가능한 단순 모델을 목표로 합니다.

- 지연시간(latency)은 μs 단위의 **상수 비용 모델**(실SSD의 병렬성/큐잉/채널 구조는 생략)
- FTL은 페이지 매핑 기반의 **단순화된 out-of-place 업데이트**(세부 최적화 생략)
- wear는 erase count를 기반으로 한 **거친 수명 지표**
- 실제 장치 수준의 절대 성능 예측이 아니라, **정책 간 상대 비교/경향 관찰**에 초점을 둡니다.

---

## 추천 실험 레시피

- 정책 비교 기본:  
  `ops=200k`, `update_ratio=0.8`, `hot_ratio=0.2`, `seed=1..N` (N>=3 권장)
- steady-state 비교:  
  `--warmup_fill 0.7~0.9`로 선행 채우기 후 본 실험 실행
- TRIM 영향:  
  `--enable_trim --trim_ratio 0.05~0.2`로 삭제 이벤트 추가


---

## 한 줄 요약

> **GC 정책을 바꿔가며** 같은 워크로드를 흘리고,  
> 결과를 **CSV로 쌓고**, **병합/요약/플롯까지 재현 가능하게** 만드는 GC 실험용 미니 랩.