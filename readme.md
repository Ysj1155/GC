# SSD Garbage Collection Simulator

## 프로젝트 개요
본 프로젝트는 **SSD(솔리드 스테이트 드라이브)의 Garbage Collection(GC) 알고리즘을 시뮬레이션하고 시각화**하는 것을 목표로 합니다.  
SSD는 NAND 플래시 메모리를 기반으로 하며, 덮어쓰기 불가(erase-before-write)라는 특성 때문에 **GC 과정**이 필수적입니다.  
그러나 GC는 SSD의 **쓰기 증폭(Write Amplification)**, **성능 저하**, **수명 단축**을 유발할 수 있습니다:contentReference[oaicite:0]{index=0}.  

본 연구에서는 다양한 GC 정책(예: Greedy, Cost-Benefit, BSGC 등)을 소프트웨어로 구현하여, **워크로드별 성능 차이를 정량적으로 분석**합니다.  
이를 통해 GC가 SSD 성능, 내구성, QoS에 미치는 영향을 파악하고, 시각화 및 비교를 통해 이해를 돕는 것이 핵심 목표입니다.

---

## 연구 목적
- **학술적 동기**  
  - SSD GC는 오버헤드가 크고, QoS 저하 및 지연(latency)을 유발.
  - 기존 연구는 효율적인 희생 블록 선택(Greedy, CB, CAT, BSGC 등) 기법을 제안했으나, 실제 워크로드 시뮬레이션 기반 비교가 부족.
- **실무적 동기**  
  - 데이터센터·실시간 시스템에서는 tail latency와 안정적인 성능 보장이 필요.
  - GC 정책 차이가 RocksDB 등 DBMS와 스토리지 계층의 성능에 직접적인 영향을 미침.

따라서 본 프로젝트는 **알고리즘별 GC 효과를 직접 구현·시뮬레이션·분석**하는 것을 통해 학술적/실무적 통찰을 제공하려 합니다.

---

## 현재까지 진행 상황
1. **개발 환경 세팅**
   - `venv` 가상환경 생성 및 `GC` 디렉토리 구축
   - Python 기반 시뮬레이터 코드 초기 작성

2. **GC 알고리즘 구현**
   - Greedy 정책 구현 및 실행 성공
   - 실행 명령어 예시:
     ```bash
     python run_sim.py --gc_policy greedy --ops 5000 --update_ratio 0.8
     ```

3. **실험 결과**  
=== Simulation Result ===  
Host writes (pages):   5,000  
Device writes (pages): 5,639  
WAF (device/host):     1.128  
GC count:              243  
Avg erase per block:   0.95 (min=0, max=2, Δ=2)  
Free pages remaining:  13849 / 16384    
- Host write 대비 Device write가 많아 쓰기 증폭 발생(WAF > 1)
- Garbage Collection이 243회 수행됨    
- 정상적으로 GC 동작 및 성능 지표 산출 확인    

---

## 📅 앞으로의 계획
- [ ] Cost-Benefit, CAT, BSGC 등 다른 GC 알고리즘 구현
- [ ] GC 정책별 WAF, GC 횟수, 지연(latency) 비교
- [ ] RocksDB와 연계된 DB workload 적용 실험
- [ ] 시각화(그래프) 도구를 통해 성능 차이 분석
- [ ] 최종 보고서 및 발표 자료 제작

---

## 📖 참고 문헌
- 김한얼, *머신러닝 알고리즘을 통한 SSD 가비지 컬렉션 감지 및 관리 기법*, 홍익대, 2014:contentReference[oaicite:6]{index=6}  
- 오승진, *RocksDB SSTable 크기가 성능에 미치는 영향 분석*, 성균관대, 2022:contentReference[oaicite:7]{index=7}  
- 김성호 외, *SSD 기반 저장장치 시스템에서 마모도 균형과 내구성 향상을 위한 가비지 컬렉션 기법*, 한국컴퓨터정보학회논문지, 2017:contentReference[oaicite:8]{index=8}  
- 박상혁, *Analysis of the K2 Scheduler for a Real-Time System with a SSD*, 성균관대, 2021:contentReference[oaicite:9]{index=9}

---

## 🗓 Changelog — 2025-09-21

### 1) 성능/안정성 개선
- **Reverse Map 도입**: `(block, page) → LPN` 역매핑 추가로 GC 마이그레이션 탐색을 O(유효페이지)로 단축.
- **Active Block(로그 구조 쓰기)** 적용: 활성 블록에 연속 기록 → 조각화 완화, WAF/GC 감소 기대.

### 2) GC 폭주 방지
- **Simulator 정책 수정**: “호스트 1회 쓰기 전에 GC 최대 1회”로 제한하여 연쇄 GC 발생 억제.

### 3) 측정 지표 확장
- **GC 시간 계측**: `gc_total_time`, `gc_durations` 수집.
- 콘솔 요약에 **GC total/avg/p50/p95/p99(ms)** 출력 추가.
- CSV(`--out_csv`)에도 `gc_time_total_ms, gc_time_avg_ms, gc_time_p50_ms, gc_time_p95_ms, gc_time_p99_ms` 컬럼 기록.

### 4) 결과 시각화 유틸
- **`analyze_results.py` 추가**: `results.csv`로부터 WAF / GC_count / GC p99 그래프 생성(`plots/` 저장).

### 5) GC 정책 확장(옵션)
- **BSGC**(균형형) 간단 구현 추가: 무효비와 마모 균형을 함께 고려.  
  → `--gc_policy bsgc` 로 실행 가능.

### 6) 버그 픽스
- `models.py` 내 **`PageState` 누락으로 인한 NameError** 해결(파일 전면 교체).
- `metrics.py`의 **`summarize_metrics` 미정의 ImportError** 해결 및 CSV 함수 보강.

---

### 🔧 변경 파일
- `models.py` : Reverse Map, Active Block, GC 시간 계측 추가
- `simulator.py` : 1-step 당 GC 최대 1회 로직
- `metrics.py` : GC 시간(총/평균/퍼센타일) 출력 및 CSV 기록
- `gc_algos.py` : `bsgc_policy` 및 `get_gc_policy()` 연동
- `analyze_results.py` : 결과 시각화 스크립트 (신규)

### ▶️ 실행 예시
```bash
# 실험 수행 + CSV 저장
python run_sim.py --gc_policy greedy --ops 5000 --update_ratio 0.8 --hot_ratio 0.2 --hot_weight 0.85 --out_csv results.csv --note "greedy_rl1"
python run_sim.py --gc_policy cb     --ops 5000 --update_ratio 0.8 --hot_ratio 0.2 --hot_weight 0.85 --out_csv results.csv --note "cb_rl1"
python run_sim.py --gc_policy bsgc   --ops 5000 --update_ratio 0.8 --hot_ratio 0.2 --hot_weight 0.85 --out_csv results.csv --note "bsgc_rl1"

# 그래프 생성
python analyze_results.py   # plots/waf_by_run.png, gc_by_run.png, gc_p99_by_run.png