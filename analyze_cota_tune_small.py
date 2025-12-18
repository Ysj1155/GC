"""
analyze_cota_tune_small.py

Purpose
-------
COTA(policy='cota') 하이퍼파라미터 튜닝(sweep) 결과를 요약하고,
WAF(성능 측면) vs wear_std(마모 균형 측면) 트레이드오프를 시각화하는 분석 스크립트.

Why this file exists
--------------------
- COTA는 score 식에 alpha/beta/gamma/delta 같은 하이퍼파라미터가 들어가며,
  동일한 workload/ops에서도 파라미터에 따라 결과(WAF, wear_std)가 달라진다.
- sweep로 쌓인 summary.csv를 파라미터 조합별로 모아서 평균/분산을 계산하면,
  (1) 어떤 조합이 안정적인지(분산이 작은지)
  (2) 어떤 조합이 성능-마모의 좋은 균형인지
  를 빠르게 판단할 수 있다.
- 논문/부록/발표에서 필요한 산점도(파레토 느낌)와 표(csv)를 자동 생성한다.

Inputs / Outputs (Contract)
---------------------------
Input:
- results/cota_tune_small/summary.csv
  - 필수 컬럼(최소): policy, ops, update_ratio, hot_ratio, seed, waf, wear_std,
    cota_alpha, cota_gamma
  - (참고) beta/delta도 있을 수 있으나, 이 스크립트는 기본적으로 고정값이라 가정하고 생략 가능.

Output:
- results/cota_tune_small/summary_by_param.csv
  - (cota_alpha, cota_gamma) 조합별 집계:
    waf_mean, waf_std, wear_mean, wear_std, n(=seed 개수)
- results/cota_tune_small/cota_pareto_scatter.png
  - x: WAF(mean), y: wear_std(mean), 각 점은 (alpha, gamma) 조합

Assumptions / Notes
-------------------
- summary.csv는 여러 seed 반복 결과를 포함한다고 가정한다.
- waf_std / wear_std(집계의 std)는 seed 간 변동성(재현성/안정성)의 힌트이다.
- ops=200000, update_ratio=0.8, hot_ratio=0.2 조건은 튜닝 실험 조건을 고정하려는 의도다.
  => 다른 조건의 결과가 섞이면 평균이 왜곡될 수 있어 필터링이 필수.

Failure modes (common pitfalls)
-------------------------------
- CSV 경로가 잘못되면 FileNotFoundError
- 컬럼명이 변경되면 KeyError
- seed 반복이 1개(n=1)이면 std가 NaN이 나올 수 있음 (판단 시 주의)
- float 비교(0.8, 0.2)가 데이터 저장 방식에 따라 미세 오차가 있을 수 있음
  => 필요하면 np.isclose로 비교하는 방식으로 바꿀 수 있다.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# 입력 CSV 위치 (실험 sweep가 떨어뜨린 summary.csv)
CSV_PATH = "results/cota_tune_small/summary.csv"

# 출력 디렉토리: CSV_PATH 기준이 아니라 "프로젝트 표준 결과 폴더"로 고정
OUT_DIR = "results/cota_tune_small"


def main():
    """
    Main pipeline
    -------------
    1) summary.csv 로드
    2) 특정 조건(튜닝 조건 고정) + policy=cota 필터링
    3) (cota_alpha, cota_gamma) 조합별로 WAF / wear_std 통계 요약(평균/표준편차/seed개수)
    4) 요약 테이블 저장
    5) WAF–wear_std 산점도 저장 (각 점에 α,γ 라벨)
    """

    # --- (0) output 폴더가 없으면 생성: 재현성/편의성 ---
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- (1) CSV 로드 ---
    df = pd.read_csv(CSV_PATH)

    # --- (1-1) 스키마(컬럼) 검증: "다른 사람이 돌려도 어디서 깨졌는지" 명확하게 ---
    required_cols = {
        "policy", "ops", "update_ratio", "hot_ratio", "seed",
        "waf", "wear_std", "cota_alpha", "cota_gamma"
    }
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        raise KeyError(
            f"[analyze_cota_tune_small] Missing columns in {CSV_PATH}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # --- (2) COTA 튜닝 결과만 필터링 ---
    # NOTE: float 비교는 데이터가 딱 0.8/0.2로 저장된다는 가정.
    # 만약 0.8000000001 같은 값이 섞여서 필터가 비면, np.isclose 비교로 바꾸는 게 안전.
    cond = (
        (df["policy"] == "cota") &
        (df["ops"] == 200000) &
        (df["update_ratio"] == 0.8) &
        (df["hot_ratio"] == 0.2)
    )
    df_cota = df.loc[cond].copy()

    # 필터 결과가 0개면, 사용자 입장에서 "왜 비었는지"가 중요하니까 명확히 알려줌
    if df_cota.empty:
        raise ValueError(
            "[analyze_cota_tune_small] Filtered dataframe is empty.\n"
            "Check whether summary.csv contains rows with:\n"
            "policy=cota, ops=200000, update_ratio=0.8, hot_ratio=0.2"
        )

    # --- (3) (alpha, gamma) 조합별 집계 ---
    # 왜 alpha/gamma만?
    # - 보통 beta/delta를 고정하고, alpha/gamma를 sweep 했다는 전제.
    # - 만약 beta/delta도 sweep했다면 group_cols에 추가해서 4D 그룹으로 집계해야 왜곡이 없다.
    group_cols = ["cota_alpha", "cota_gamma"]

    agg = (
        df_cota
        .groupby(group_cols, dropna=False)  # dropna=False: 혹시 NaN이 있으면 그것도 그룹으로 잡아 원인 추적 가능
        .agg(
            waf_mean=("waf", "mean"),
            waf_std=("waf", "std"),          # seed가 1개면 NaN 가능
            wear_mean=("wear_std", "mean"),
            wear_std=("wear_std", "std"),    # seed가 1개면 NaN 가능
            n=("seed", "count"),             # 반복 횟수(샘플 수)
        )
        .reset_index()
        .sort_values(["waf_mean", "wear_mean"], ascending=[True, True])  # 보기 좋게 정렬
    )

    print("=== COTA 하이퍼파라미터별 요약 ===")
    print(agg.to_string(index=False))

    out_table = os.path.join(OUT_DIR, "summary_by_param.csv")
    agg.to_csv(out_table, index=False)

    # --- (4) WAF–wear_std 산점도 ---
    # 의미:
    # - x축(WAF)이 낮을수록 일반적으로 "쓰기증폭 적음(효율 좋음)" => 성능/수명 측면 유리한 해석 가능
    # - y축(wear_std)이 낮을수록 "마모 균형" => 내구성/균일 마모 측면 유리
    # - 둘 다 낮은 점이 '좋은 후보' (하지만 실제 목표는 상황에 따라 가중이 다를 수 있음)
    fig, ax = plt.subplots()

    for _, row in agg.iterrows():
        x = row["waf_mean"]
        y = row["wear_mean"]

        # 라벨은 너무 길면 그림이 더러워질 수 있음:
        # - 조합 개수가 많아지면 annotate 대신 범례/색상/마커로 바꾸는 것도 고려.
        label = f"α={row['cota_alpha']}, γ={row['cota_gamma']}"

        ax.scatter(x, y)
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(5, 3),
            fontsize=8
        )

    ax.set_xlabel("WAF (mean over seeds)")
    ax.set_ylabel("wear_std (mean over seeds)")
    ax.set_title("COTA Hyperparameter Sweep (ops=200k, update_ratio=0.8, hot_ratio=0.2)")

    # (선택) seed가 1개인 점(std NaN)은 불확실하다는 표시를 하고 싶다면 여기서 마커 스타일 분기 가능.
    # (선택) 파레토 전면(frontier)을 계산해서 선으로 이어주면 “파레토 느낌”이 더 강해짐.

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "cota_pareto_scatter.png")
    plt.savefig(out_fig, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()