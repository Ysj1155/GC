import os

import matplotlib
matplotlib.use("Agg")  # GUI 안 띄우고 파일만 저장하는 백엔드

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# BSGC concept parameters
# -----------------------------
alpha = 0.7  # weight for invalid_ratio
beta  = 0.3  # weight for (1 - wear_norm)

# 0~1 range virtual invalid_ratio / wear_norm grid
invalid = np.linspace(0, 1, 100)
wear    = np.linspace(0, 1, 100)
X, Y    = np.meshgrid(invalid, wear)        # X: invalid_ratio, Y: wear_norm

# BSGC score = α * invalid_ratio + β * (1 - wear_norm)
score = alpha * X + beta * (1 - Y)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 5))

# score heatmap (higher score -> warmer color)
im = ax.imshow(
    score,
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="auto"
)

# colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("BSGC score", fontsize=11)

# axis labels / title
ax.set_xlabel("invalid_ratio (fraction of invalid pages)", fontsize=11)
ax.set_ylabel("wear_norm (normalized erase count)", fontsize=11)
ax.set_title("BSGC preferred region (concept)", fontsize=13)

# Greedy annotation: only looks at invalid_ratio
ax.annotate(
    "Greedy: uses only invalid_ratio",
    xy=(0.8, 0.8), xytext=(0.35, 0.9),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

# BSGC annotation: prefers high invalid_ratio & low wear_norm
ax.annotate(
    "BSGC: prefers high invalid_ratio\n& low wear_norm",
    xy=(0.9, 0.1), xytext=(0.45, 0.25),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=10
)

# fix axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()

# ------------------------
# Output path: results/plot
# ------------------------
out_dir = os.path.join("results", "plot")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "bsgc_concept.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")