from __future__ import annotations

"""
metrics.py

시뮬레이터 실행 결과를 숫자(메트릭)로 뽑아내고, summary.csv로 저장하는 모듈입니다.

이 프로젝트에서 metrics.py의 역할은 단순합니다.

1) 메트릭 추출(Collect)
   - simulator 구현이 약간씩 달라도(sim.ssd 구조, 필드명 차이 등)
     가능한 값을 최대한 “견고하게” 읽어옵니다.
2) 요약 기록(Log)
   - 실험 1회(run)마다 핵심 지표를 한 행(row)으로 만들고,
     summary.csv에 append 방식으로 누적 기록합니다.
3) 재현성(Reproducibility)
   - meta(실험 파라미터/정책/seed/시간 등)를 row에 합쳐 저장하여
     “이 결과가 어떤 조건에서 나왔는지” CSV만 봐도 추적 가능하게 합니다.

입력/출력 계약(Contract)
-----------------------
Input:
- sim: Simulator 객체 (또는 sim.ssd를 가진 객체)
- meta: (옵션) 실험 설정값 dict. metrics에 merge되어 CSV에 함께 기록됨.

Output:
- collect_run_metrics(sim) -> Dict[str, Any] : 한 run의 핵심 수치
- append_summary_csv(path, sim, meta) : summary.csv에 1행 append
- summary_row(sim, meta) -> Dict[str, Any] : 분석 스크립트에서 재사용 가능한 단일 row

필드명 호환성(왜 _get이 필요한가?)
--------------------------------
프로젝트를 오래 굴리다 보면,
- host write 카운터 이름이 host_write_pages -> host_writes로 바뀌거나,
- device write가 device_write_pages -> device_writes로 바뀌거나,
- pages_per_block이 ppb로 줄어들거나,
같은 일이 자주 생깁니다.

이 모듈은 _get() 유틸을 통해 후보 속성명 리스트 중 첫 번째로 잡히는 값을 사용하여,
코드 변경에도 분석 파이프라인이 쉽게 깨지지 않도록 합니다.

주의(Assumptions / Pitfalls)
----------------------------
- WAF = device_writes / host_writes 입니다.
  host_writes == 0이면 0.0으로 둡니다(division 보호).
- wear 통계는 blocks의 erase_count를 기반으로 합니다.
- stability snapshot(transition_rate, reheat_rate)은 “시계열 기반 실제 전이율”이 아니라
  한 시점의 분포로 만든 보수적 근사 신호입니다(autotune 보조용).

확장 포인트
-----------
- 새로운 메트릭을 추가하려면:
  1) collect_run_metrics에 값을 계산해 dict에 넣고
  2) (필요하면) experiments/run_sim에서 meta에도 같이 기록
  3) analyze_results에서 플롯/필터 컬럼으로 사용하면 됩니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import csv
import math
import os


# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------

def _get(obj: Any, names: List[str], default: Any = None) -> Any:
    """
    여러 후보 속성명 중, 실제로 존재하는 첫 번째 값을 반환합니다.

    특징
    ----
    - names에는 "a", "b", "ssd.host_writes" 같은 후보를 넣을 수 있습니다.
    - "a.b" 형태(중첩 경로)를 지원합니다.

    예시
    ----
    _get(ssd, ["host_write_pages", "host_writes"], 0)
    - ssd.host_write_pages가 있으면 그 값을 쓰고,
    - 없으면 ssd.host_writes를 쓰고,
    - 둘 다 없으면 default(0)를 반환합니다.

    목적
    ----
    - 코드 리팩토링으로 필드명이 바뀌어도 분석 코드가 덜 깨지게 만드는 호환성 레이어.
    """
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if cur is None or not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    return default


def _list_stat(xs: List[float]) -> Dict[str, float]:
    """
    리스트 통계(min/max/avg/std)를 계산합니다.

    표준편차(std)는 모집단 기준(population std, 분모=n)으로 계산합니다.
    (실험을 여러 번 반복할 때 seed 분산은 analyze 단계에서 따로 다루는 것을 권장)
    """
    if not xs:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "std": 0.0}

    n = len(xs)
    mn = min(xs)
    mx = max(xs)
    avg = sum(xs) / n
    var = sum((x - avg) ** 2 for x in xs) / n
    return {"min": mn, "max": mx, "avg": avg, "std": math.sqrt(var)}


# ------------------------------------------------------------
# 스냅샷(선택): autotune/관찰용 보조 신호
# ------------------------------------------------------------

@dataclass
class StabilitySnapshot:
    """
    hot/cold 관련 상태 변화를 정량화하려는 보조 신호.

    주의:
    - 이 값들은 진짜 전이율/재가열율이 아니라,
      한 시점의 분포에서 만든 근사 신호입니다.
    """
    transition_rate: float = 0.0  # hot↔cold 전이 가능성 근사
    reheat_rate: float = 0.0      # cold→hot 재가열 가능성 근사


def make_stability_snapshot(sim: Any, hot_thr: float = 0.33, cold_thr: float = 0.05) -> StabilitySnapshot:
    """
    블록의 inv_ewma 분포로부터 transition/reheat “가능성” 신호를 근사합니다.

    왜 근사인가?
    -----------
    - 실제 전이율을 계산하려면 시간에 따른 상태 변화 기록(시계열)이 필요합니다.
    - 하지만 여기서는 결과 요약 단계에서 간단히 쓰기 위해,
      분포의 꼬리(hot/cold 비율)가 클수록 전이가 활발할 수도 있다는 가정을 둡니다.

    계산 방식(보수적)
    ----------------
    - hot 블록 비율(h/n), cold 블록 비율(c/n)을 계산
    - 분포 양극화 정도(pol) = (h + c) / n
    - transition_rate = min(0.3, pol * 0.2)
    - reheat_rate     = min(0.3, (h/n) * 0.1)

    결과적으로 너무 큰 값으로 튀지 않게 0~0.3 범위로 클램프합니다.
    """
    ssd = getattr(sim, "ssd", sim)
    blocks = getattr(ssd, "blocks", []) or []
    if not blocks:
        return StabilitySnapshot()

    h = sum(1 for b in blocks if float(getattr(b, "inv_ewma", 0.0)) >= hot_thr)
    c = sum(1 for b in blocks if float(getattr(b, "inv_ewma", 0.0)) <= cold_thr)
    n = max(1, len(blocks))

    pol = (h + c) / n
    return StabilitySnapshot(
        transition_rate=min(0.3, pol * 0.2),
        reheat_rate=min(0.3, h / n * 0.1),
    )


# ------------------------------------------------------------
# 메트릭 수집
# ------------------------------------------------------------

def collect_run_metrics(sim: Any) -> Dict[str, Any]:
    """
    시뮬레이터 1회 실행(run)의 핵심 메트릭을 추출합니다.

    설계 목표
    --------
    - “견고함”: simulator 구현 디테일이 달라도 최대한 값을 뽑아오도록 합니다.
    - “안전함”: 누락된 값은 0 또는 None으로 처리해 후처리 파이프라인이 깨지지 않게 합니다.
    - “재현성”: 어떤 값을 어디서 읽었는지 명확한 필드명을 사용합니다.

    반환 메트릭(주요)
    -----------------
    - host_writes, device_writes, waf
    - gc_count, gc_avg_s
    - free_pages, free_blocks
    - total_pages, pages_per_block
    - wear_min/max/avg/std
    - trimmed_pages, valid_pages, invalid_pages
    - transition_rate, reheat_rate (보조 신호)
    """
    ssd = getattr(sim, "ssd", sim)

    # -------------------------
    # Write counters + WAF
    # -------------------------
    host_w = int(_get(ssd, ["host_write_pages", "host_writes", "host_pages"], 0))
    dev_w  = int(_get(ssd, ["device_write_pages", "device_writes", "dev_pages"], 0))
    waf = (dev_w / host_w) if host_w > 0 else 0.0

    # -------------------------
    # GC counters + durations
    # -------------------------
    gc_cnt = int(_get(ssd, ["gc_count"], 0))
    gc_durs = list(_get(ssd, ["gc_durations"], []) or [])
    gc_avg = (sum(gc_durs) / len(gc_durs)) if gc_durs else 0.0

    # -------------------------
    # Free space snapshot
    # -------------------------
    free_pages  = int(_get(ssd, ["free_pages"], 0))
    free_blocks = int(_get(ssd, ["free_blocks"], 0))

    # -------------------------
    # Blocks / geometry
    # -------------------------
    blocks = list(getattr(ssd, "blocks", []) or [])
    pages_per_block = int(_get(ssd, ["pages_per_block", "ppb"], 0)) or int(_get(sim, ["pages_per_block", "ppb"], 0))

    # -------------------------
    # Wear statistics
    # -------------------------
    wear_list = [int(getattr(b, "erase_count", 0)) for b in blocks]
    wear_stat = _list_stat([float(x) for x in wear_list])

    # -------------------------
    # TRIM / valid / invalid aggregates
    # -------------------------
    total_trimmed = sum(int(getattr(b, "trimmed_pages", 0)) for b in blocks)
    total_invalid = sum(int(getattr(b, "invalid_count", 0)) for b in blocks)
    total_valid   = sum(int(getattr(b, "valid_count", 0)) for b in blocks)

    # -------------------------
    # Policy name (function name / lambda 등)
    # -------------------------
    policy_name = None
    pol = getattr(sim, "gc_policy", None)
    if pol is not None:
        policy_name = getattr(pol, "__name__", str(pol))

    # -------------------------
    # Total pages estimation (가능하면 계산)
    # -------------------------
    total_pages = None
    nb = int(_get(ssd, ["num_blocks", "blocks", "total_blocks"], 0))
    if isinstance(getattr(ssd, "num_blocks", None), int):
        nb = getattr(ssd, "num_blocks")
    if nb and pages_per_block:
        total_pages = nb * pages_per_block

    # -------------------------
    # Stability snapshot (선택)
    # -------------------------
    snap = make_stability_snapshot(sim)

    return {
        "policy": policy_name,
        "host_writes": host_w,
        "device_writes": dev_w,
        "waf": round(waf, 6),

        "gc_count": gc_cnt,
        "gc_avg_s": round(gc_avg, 6),

        "free_pages": free_pages,
        "free_blocks": free_blocks,

        "total_pages": total_pages if total_pages is not None else 0,
        "pages_per_block": pages_per_block,

        "wear_min": wear_stat["min"],
        "wear_max": wear_stat["max"],
        "wear_avg": round(wear_stat["avg"], 6),
        "wear_std": round(wear_stat["std"], 6),

        "trimmed_pages": total_trimmed,
        "valid_pages": total_valid,
        "invalid_pages": total_invalid,

        # autotune/관찰용 보조 신호
        "transition_rate": round(snap.transition_rate, 6),
        "reheat_rate": round(snap.reheat_rate, 6),
    }


# ------------------------------------------------------------
# 요약 CSV 저장
# ------------------------------------------------------------

def append_summary_csv(path: str, sim: Any, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    summary.csv에 “한 run의 결과”를 1행 append합니다.

    동작 규칙
    --------
    - 파일이 없으면:
      - fieldnames를 row.keys()를 알파벳 정렬한 순서로 만들고,
      - 헤더를 생성한 뒤 1행을 기록합니다.
    - 파일이 있으면:
      - 기존 헤더 순서를 유지합니다(컬럼 순서 안정성).
      - 새 컬럼이 생기면 헤더 뒤에 추가하여 확장합니다(하위 호환).

    meta 병합 규칙
    -------------
    - row = metrics + meta (meta가 같은 키를 갖는 경우 meta 우선)
      => 실험 설정값이 메트릭 이름과 겹칠 때도 실험 사양 기록이 최종값이 되도록 합니다.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    metrics = collect_run_metrics(sim)
    row = {**metrics}
    if meta:
        row.update(meta)

    write_header = not os.path.exists(path)

    if write_header:
        # 새 파일: 컬럼 순서를 안정적으로 만들기 위해 알파벳 정렬
        fieldnames = sorted(row.keys())
    else:
        # 기존 파일: 헤더를 읽어 순서를 유지
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            try:
                header = next(r)
            except StopIteration:
                header = []
        fieldnames = list(header)

        # 새 컬럼이 등장했다면 뒤에 추가(확장)
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------
# (선택) 분석 스크립트에서 재사용 가능한 “단일 요약 행” 생성기
# ------------------------------------------------------------

def summary_row(sim: Any, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    sim + meta를 합쳐 summary.csv 1행과 같은 형태의 dict를 반환합니다.

    사용처
    ----
    - experiments.py에서 QC/콘솔 출력용
    - analyze/튜닝 스크립트에서 임시 테이블 생성용
    """
    row = collect_run_metrics(sim)
    if meta:
        row.update(meta)
    return row