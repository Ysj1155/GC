import itertools
import subprocess
import platform
import sys
import datetime
import json
import os
import re

# ---------- Settings ----------
PY = sys.executable
OPS = 10000
HOT_RATIO = 0.2
USER_CAPS = [1.0, 0.9]               # OP axis (user capacity ratio)

UPDATE_RATIOS = [0.5, 0.8]
HOT_WEIGHTS   = [0.6, 0.85]
SEEDS         = [42, 123]

# ATCB weight ablation (alpha, beta, gamma, eta)
ATCB_WEIGHTS = [
    (0.5, 0.3, 0.1, 0.1),  # default
    (0.6, 0.2, 0.1, 0.1),  # invalid ratio ↑
    (0.4, 0.4, 0.1, 0.1),  # temperature ↑
]

# Optional tag for run folder name ("" to omit)
TAG = "atcb"  # or ""

# ---------- Output folders ----------
DAY_DIR = os.path.join("results", datetime.datetime.now().strftime("%Y-%m-%d"))
os.makedirs(DAY_DIR, exist_ok=True)

def next_run_dir(base: str, tag: str = "") -> str:
    """Create results/YYYY-MM-DD/runNN[_tag] and return its path."""
    runs = []
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and re.match(r"^run\d{2}", d):
            try:
                runs.append(int(d.split("_", 1)[0][3:]))  # runNN[_tag]
            except ValueError:
                pass
    idx = (max(runs) if runs else 0) + 1
    name = f"run{idx:02d}" + (f"_{tag}" if tag else "")
    out = os.path.join(base, name)
    os.makedirs(out, exist_ok=True)
    return out

OUT_DIR = next_run_dir(DAY_DIR, TAG)

# ---------- Meta snapshot ----------
def git_commit_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None

TOTAL_RUNS = len(UPDATE_RATIOS) * len(HOT_WEIGHTS) * len(SEEDS) * len(USER_CAPS) * (2 + len(ATCB_WEIGHTS))

meta = {
    "ops": OPS,
    "hot_ratio": HOT_RATIO,
    "user_caps": USER_CAPS,
    "update_ratios": UPDATE_RATIOS,
    "hot_weights": HOT_WEIGHTS,
    "seeds": SEEDS,
    "atcb_weights": ATCB_WEIGHTS,
    "total_runs": TOTAL_RUNS,
    "git_commit": git_commit_short(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "out_dir": OUT_DIR,
}
with open(os.path.join(OUT_DIR, "sweep_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

with open(os.path.join("results", "LATEST.txt"), "w", encoding="utf-8") as f:
    f.write(OUT_DIR)

# ---------- Common args ----------
BASE_ARGS = [
    "--ops", str(OPS),
    "--hot_ratio", str(HOT_RATIO),
    "--out_csv", "results.csv",   # saved under OUT_DIR
    "--out_dir", OUT_DIR,
]

def run(cmd):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)

# ---------- Sweep ----------
for u, hw, seed, ucap in itertools.product(UPDATE_RATIOS, HOT_WEIGHTS, SEEDS, USER_CAPS):
    # Greedy / CB
    for pol in ["greedy", "cb"]:
        note = f"{pol}_u{u}_hw{hw}_uc{ucap}_s{seed}"
        cmd = [
            PY, "run_sim.py",
            "--gc_policy", pol,
            "--update_ratio", str(u),
            "--hot_weight", str(hw),
            "--user_capacity_ratio", str(ucap),
            "--seed", str(seed),
            "--note", note,
            *BASE_ARGS,
        ]
        run(cmd)

    # ATCB with ablations
    for (a, b, g, e) in ATCB_WEIGHTS:
        note = f"atcb_u{u}_hw{hw}_uc{ucap}_s{seed}_a{a}_b{b}_g{g}_e{e}"
        cmd = [
            PY, "run_sim.py",
            "--gc_policy", "atcb",
            "--update_ratio", str(u),
            "--hot_weight", str(hw),
            "--user_capacity_ratio", str(ucap),
            "--seed", str(seed),
            "--atcb_alpha", str(a),
            "--atcb_beta",  str(b),
            "--atcb_gamma", str(g),
            "--atcb_eta",   str(e),
            "--note", note,
            *BASE_ARGS,
        ]
        run(cmd)

print(f"[OK] Sweep complete.\nAnalyze with:\n  python analyze_results.py --base {OUT_DIR}")