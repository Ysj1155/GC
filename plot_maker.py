from __future__ import annotations

"""
plot_maker.py

이 파일은 정책(알고리즘) 설명용 그림을 만드는 재현 가능한 도식 생성기입니다.

왜 필요한가?
------------
실험 결과(plot)들은 숫자/분포를 보여주지만,
정책이 어떤 방향을 선호하는지 (의사결정의 감각)는 한 장의 개념도(concept plot)가 더 빨리 전달합니다.

여기서는 BSGC의 핵심 아이디어를 2D로 시각화합니다.

BSGC 점수(컨셉)
---------------
BSGC는 victim 블록을 고를 때,
- invalid_ratio가 클수록 좋고(=지울 때 이득이 큼),
- wear_norm(마모 정규화)이 낮을수록 좋다(=이미 많이 닳은 블록은 덜 건드리고 싶음)

이라는 *방향성*을 가집니다.

컨셉 점수는 다음과 같이 정의합니다:

    score = α * invalid_ratio + β * (1 - wear_norm)

- α: invalid_ratio 가중치
- β: wear 보정 항의 가중치 (wear가 낮을수록 score 증가)

주의
----
- 이 파일은 실제 시뮬레이션 데이터로부터 점수를 계산하는 것이 아니라,
  invalid_ratio와 wear_norm을 0~1 범위 격자로 가정하고 의사결정 지형을 그립니다.
- 즉, 논문/README에서 정책 아이디어를 설명하는 그림용입니다(실험 결과 그래프가 아님).

실행 결과
--------
results/plot/bsgc_concept.png 에 저장됩니다.
- matplotlib는 GUI 없이 저장만 하도록 Agg 백엔드를 사용합니다(서버/CLI 환경 안전).
"""

import os

import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일 저장만 (headless 환경 안전)

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) 컨셉 파라미터 (그림에서 "무엇을 더 중요하게 보나"를 결정)
# ============================================================

# invalid_ratio에 대한 가중치: invalid가 많을수록 점수↑
alpha = 0.7

# wear 보정 가중치: wear_norm이 낮을수록 점수↑  (1 - wear_norm)
beta = 0.3


# ============================================================
# 2) (invalid_ratio, wear_norm) 가상 격자 생성
# ============================================================

# invalid_ratio: 0~1
invalid = np.linspace(0.0, 1.0, 100)

# wear_norm: 0~1
wear = np.linspace(0.0, 1.0, 100)

# X축=invalid_ratio, Y축=wear_norm인 2D 그리드
X, Y = np.meshgrid(invalid, wear)


# ============================================================
# 3) BSGC 컨셉 점수 계산
# ============================================================

# BSGC score = α * invalid_ratio + β * (1 - wear_norm)
# - 오른쪽( invalid↑ )으로 갈수록 score↑
# - 아래쪽( wear↓ )으로 갈수록 score↑
score = alpha * X + beta * (1.0 - Y)


# ============================================================
# 4) Plot (heatmap)
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

# heatmap:
# - origin="lower": (0,0)을 왼쪽 아래로 두어 직관적인 좌표계로
# - extent=[0,1,0,1]: 축을 invalid_ratio/wear_norm 값으로 라벨링
# - aspect="auto": 그림 비율을 자동 조정
im = ax.imshow(
    score,
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="auto",
)

# colorbar: score의 크기를 색으로 해석할 수 있게 추가
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("BSGC score", fontsize=11)

# 축/제목: "이 그림이 무엇을 뜻하는지" 한 번에 읽히도록 명확히
ax.set_xlabel("invalid_ratio (fraction of invalid pages)", fontsize=11)
ax.set_ylabel("wear_norm (normalized erase count)", fontsize=11)
ax.set_title("BSGC preferred region (concept)", fontsize=13)

# ------------------------------------------------------------
# 5) 주석(annotate): Greedy vs BSGC의 차이를 그림 위에서 설명
# ------------------------------------------------------------

# Greedy는 invalid_ratio만 보는 느낌이므로,
# “wear 축을 무시한다”는 메시지를 하나 넣어줌.
ax.annotate(
    "Greedy: uses only invalid_ratio",
    xy=(0.8, 0.8),          # 화살표가 가리키는 점(대충 높은 invalid 영역)
    xytext=(0.35, 0.9),     # 텍스트 위치
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10,
)

# BSGC는 “오른쪽 아래”를 좋아한다:
# invalid_ratio 높고, wear_norm 낮은 영역
ax.annotate(
    "BSGC: prefers high invalid_ratio\n& low wear_norm",
    xy=(0.9, 0.1),          # 오른쪽 아래
    xytext=(0.45, 0.25),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10,
)

# 축 범위 고정(개념도라 0~1로 깔끔하게)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()


# ============================================================
# 6) Output (재현 가능한 저장 경로)
# ============================================================

# 결과 파일은 results/plot 폴더에 저장
out_dir = os.path.join("results", "plot")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "bsgc_concept.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")