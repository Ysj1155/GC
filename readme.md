# SSD Garbage Collection Simulator

> 본 레포는 **COTA (Cost-Over-Temperature-and-Age)** 와 기존 정책(Greedy / CB / BSGC / ATCB / RE50315)을 **동일 조건**에서 비교하고,  
> **WAF·마모 균등성·GC 오버헤드** 지표를 **CSV로 자동 적재**하는 실험 프레임워크입니다.

---

## 개요
SSD는 erase-before-write 특성 때문에 GC가 필수이며, GC는 **쓰기 증폭(WAF)**·**성능 저하**·**수명 단축**을 유발할 수 있습니다.  
이 프로젝트는 다양한 GC 정책을 소프트웨어로 구현/실행하고, 워크로드 별 **정량 지표**를 수집하여 **정확하고 재현 가능한 비교**를 제공합니다.

**핵심 제공물**
- GC 정책 비교 프레임워크(COTA, Greedy, CB, BSGC, ATCB, RE50315)
- 동일 워크로드에서 자동 **CSV 적재** (메타/지표 일관 스키마)
- 재현성(고정 seed, 설정 스냅샷, 옵션으로 git 커밋 해시)

---

## 연구 목적
- **목표:** 기존 정책 대비 COTA가 다음을 달성하는지 평가  
  1) **WAF 감소** (device_writes/host_writes)  
  2) **마모 균등화** (wear_std, wear_max 감소)  
  3) **GC 오버헤드 감소** (gc_count, gc_avg_s 감소)  
- **판단 기준:** 동일 워크로드·동일 시드에서 통계치(중앙값/사분위) 비교

---

## 연구 질문 (RQs)
- **RQ1.** Hot/Cold 편중, 업데이트 비율, OP(Over-Provisioning) 수준이 WAF/GC 빈도/마모 균형에 미치는 영향은?  
- **RQ2.** Greedy/CB/BSGC/ATCB/RE50315의 강/약점은 무엇이며, 어떤 조건에서 역전이 발생하는가?  
- **RQ3.** hotness·invalid·age·wear를 함께 고려한 **경량 점수식(COTA)** 은 기존 대비 성능/내구성을 개선하는가?  
- **RQ4.** TRIM/OP/백그라운드-GC 조합은 WAF와 wear-leveling을 어떻게 바꾸는가?

---

## 접근 방식
- **시뮬레이터:** 페이지/블록 모델, active-block, reverse map, per-GC 이벤트 로그  
- **워크로드:** update_ratio / hot_ratio / trim / OP(user_capacity_ratio) 스윕  
- **지표:** WAF, GC count/avg_s, wear_avg/std/min/max, free_pages 타임라인, trimmed_pages  
- **결과:** 실험별 CSV 자동 누적(메타 포함), 분석 스크립트로 요약/그림화

---

## 레포 구조 & 파일 역할
├─ config.py # SimConfig (장치/실험 파라미터)  
├─ simulator.py # SSD/Block/Page 모델, GC 실행 엔진, trace/event 로그  
├─ workload.py # 워크로드 생성 (update/hot/trim 비율)  
├─ gc_algos.py # GC 정책 (COTA/Greedy/CB/BSGC/ATCB/RE50315)  
├─ run_sim.py # 단일 실행 → CSV append (메타 포함)  
├─ experiments.py # grid/YAML/multiseed 스윕 실행  
├─ sweep.py # 간단 스윕 + 완료 메시지 출력  
├─ metrics.py # 요약/헤더 병합 CSV 유틸  
├─ analyze_results.py # (옵션) 결과 병합·요약·그림  
└─ results/ # (gitignore 권장) CSV/로그 출력  

---

## 데이터 생성 (명령어)
**스모크(동일 파일에 누적)**  
### Greedy
```
python run_sim.py --gc_policy greedy --ops 50000 \
  --out_dir results/smoke --out_csv results/smoke/summary.csv --note smoke
```
### COTA (가중치는 옵션)
```
python run_sim.py --gc_policy cota --ops 50000 \
  --cota_alpha 0.55 --cota_beta 0.25 --cota_gamma 0.15 --cota_delta 0.05 \
  --out_dir results/smoke --out_csv results/smoke/summary.csv --note smoke
```
---

## 기대 산출물
- 재현 가능한 **실험 스크립트 & CSV/플롯**, 
- **개선 알고리즘(점수식/의사코드/실험결과)**,

---

## ▶️ 데이터 생성 (명령어)

**스모크(동일 파일에 누적)**
```bash
# Greedy
python run_sim.py --gc_policy greedy --ops 50000 \
  --out_dir results/smoke --out_csv results/smoke/summary.csv --note smoke

# COTA (가중치는 옵션)
python run_sim.py --gc_policy cota --ops 50000 \
  --cota_alpha 0.55 --cota_beta 0.25 --cota_gamma 0.15 --cota_delta 0.05 \
  --out_dir results/smoke --out_csv results/smoke/summary.csv --note smoke
```

---

## CSV 스키마(요약)

### 공통 메타
policy, ops, seed, update_ratio, hot_ratio, hot_weight, trim_enabled, trim_ratio, warmup_fill, bg_gc_every, ts

### 성능/내구 지표
waf, device_writes, host_writes, gc_count, gc_avg_s, free_blocks, free_pages, valid_pages, invalid_pages, wear_avg, wear_std, wear_min, wear_max, trimmed_pages, transition_rate, reheat_rate

### COTA 메타
cota_alpha, cota_beta, cota_gamma, cota_delta, cold_victim_bias, trim_age_bonus, victim_prefetch_k

#### metrics.append_summary_csv()가 헤더 병합을 자동 처리하므로, 새 컬럼이 생겨도 기존 CSV에 안전하게 append됩니다.

---

## 참고 문헌
- SSD 가비지 컬렉션을 고려한 IO 스케줄러의 대역폭 분배 기법.pdf
- 플래시 저장장치의 garbage collection 스케줄링.pdf
- 머신러닝 알고리즘을 통한 SSD 가비지 컬렉션 감지 및 관리 기법.pdf
- 고성능 SSD를 위한 펌웨어 설계.pdf
- RocksDB SSTable 크기가 성능에 미치는 영향 분석.pdf
- SSD 기반 저장장치 시스템에서 마모도 균형과 내구성 향상을 위한 가비지 컬렉션 기법.pdf
- analysis of the K2 scheduler for a real-time system with a SSD.pdf
- Design and Implementation of Temperature-Aware Garbage collectors.pdf
- GC 특허.pdf

---

## 실험 결과

- 표 1. 동일 조건에서 Greedy vs COTA의 WAF 중앙값(±IQR) 비교  
- 그림 1. Wear 표준편차(wear_std) 비교(낮을수록 균등)  
- 그림 2. GC 오버헤드(gc_count / gc_avg_s) 비교  
- 부록 A. TRIM/OP/업데이트 비율에 대한 민감도 결과  

--- 

## 앞으로의 계획

- 데이터 적재 확대: 시드/워크로드 스윕 누적

- 분석 자동화: analyze_results.py로 정책별 통계/그림 일괄 생성

- Ablation & 민감도: COTA 가중치/trim_age_bonus/top-K 영향 분해

- 문서화: 결과 표/그림 고정, 논문 본문에 인용 가능한 수치 정리

--- 
