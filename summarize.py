"""
summarize.py

실험 결과(summary.csv)를 정책별로 한 장 요약하는 미니 스크립트.

왜 존재하나?
------------
run_sim.py / experiments.py는 실행을 돌리면서 summary.csv에 계속 append 한다.
그런데 실험이 쌓이면,
- seed가 여러 개라 평균/분산을 보기 어렵고
- 정책 간 비교도 한 눈에 안 들어온다.

그래서 이 스크립트는 summary.csv를 읽고,
정책(policy)별로 핵심 지표의 중앙값(median)과 사분위수(25%, 75%)를 뽑아서
대표값 + 퍼짐을 빠르게 확인할 수 있게 한다.

특징(재현/전달 관점)
--------------------
- 입력 파일 경로를 인자로 받을 수 있다.
  - 인자 없으면 기본값: results/comp/summary.csv
- summary.csv의 컬럼 구성이 실험마다 조금 달라도,
  있는 컬럼만 골라서 돌아가게 만들었다.
- 출력은 두 가지:
  1) results/comp/summary_by_policy.csv 저장
  2) 콘솔에 DataFrame 그대로 print

사용법
------
1) 기본 경로로 실행:
   python summarize.py

2) 다른 summary.csv를 대상으로 실행:
   python summarize.py results/smoke/summary.csv

출력 파일
---------
- results/comp/summary_by_policy.csv
  (현재 코드는 출력 경로가 고정되어 있음 → 필요하면 입력 파일 위치 기반으로 바꿀 수 있음)
-------------------
"""

from __future__ import annotations

import sys
import pandas as pd


# ------------------------------------------------------------
# 입력 경로 결정
# ------------------------------------------------------------

# argv[1]이 있으면 그걸 쓰고, 없으면 기본 파일을 사용한다.
path = sys.argv[1] if len(sys.argv) > 1 else "results/comp/summary.csv"

df = pd.read_csv(path)


# ------------------------------------------------------------
# 필요한 열만 선택 (있는 것만)
# ------------------------------------------------------------

# 실험마다 summary.csv 컬럼이 조금씩 달라도 버티도록
# "우리가 관심 있는 후보 목록" 중 실제로 존재하는 것만 고른다.
cols = [c for c in [
    "policy", "ops", "seed",
    "waf", "gc_count", "gc_avg_s",
    "wear_avg", "wear_std", "wear_min", "wear_max",
    "device_writes", "host_writes",
    "trimmed_pages",
    "transition_rate", "reheat_rate",
    "cota_alpha", "cota_beta", "cota_gamma", "cota_delta",
] if c in df.columns]

df = df[cols]


# ------------------------------------------------------------
# 정책별 요약 함수
# ------------------------------------------------------------

def agg_block(group: pd.DataFrame) -> pd.Series:
    """
    policy 그룹(=하나의 정책에 해당하는 여러 실행 rows)을 받아
    대표값(중앙값) + 퍼짐(사분위수)을 반환한다.

    왜 평균(mean)이 아니라 중앙값(median)이냐
    ---------------------------------------
    실험 데이터는 가끔 한 번 튀는 실행(outlier)이 생긴다.
    평균은 그 한 번에 흔들리기 쉬워서,
    여기서는 중앙값/사분위수가 더 실험 보고서용으로 안정적이다.

    반환 컬럼
    ---------
    - waf_med / waf_p25 / waf_p75
    - gc_med  / gc_p25  / gc_p75
    - wear_std_med
    - wear_max_med (컬럼이 있을 때만)
    """
    q = group.quantile  # 편의: q(0.25)처럼 호출하려고 변수로 잡음

    # quantile은 숫자 컬럼에 대해 계산된다.
    # 만약 특정 컬럼이 문자열로 들어왔다면, upstream에서 numeric coercion이 필요할 수 있다.
    return pd.Series({
        "waf_med": group["waf"].median(),
        "waf_p25": q(0.25)["waf"],
        "waf_p75": q(0.75)["waf"],

        "gc_med": group["gc_count"].median(),
        "gc_p25": q(0.25)["gc_count"],
        "gc_p75": q(0.75)["gc_count"],

        "wear_std_med": group["wear_std"].median(),

        # wear_max는 어떤 실험 CSV에는 없을 수도 있어서 조건 처리
        "wear_max_med": group["wear_max"].median() if "wear_max" in group else None,
    })


# ------------------------------------------------------------
# groupby → 정책별 요약 생성
# ------------------------------------------------------------

# groupby(...).apply(...)는 각 그룹에 대해 agg_block을 호출한다.
# as_index=False를 줬지만 apply를 쓰면 결과 인덱스가 꼬일 수 있어,
# 필요하면 reset_index()를 추가하거나 group_keys=False를 줄 수 있다.
out = df.groupby("policy", as_index=False).apply(agg_block)

# 현재는 출력 경로를 고정해둠.
# 입력 파일이 results/smoke/summary.csv라면,
# results/smoke/summary_by_policy.csv로 저장되게 바꾸고 싶을 수도 있다.
out.to_csv("results/comp/summary_by_policy.csv", index=False)

print(out)