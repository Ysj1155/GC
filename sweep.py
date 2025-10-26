import itertools, subprocess, platform
import sys, json
import os, re
from datetime import datetime

# ---------- Settings ----------
PY = sys.executable

# 실험 스펙 (스모크 통과 후 숫자 늘리기)
OPS         = 200000
HOT_RATIO   = 0.2
USER_CAPS   = [0.90]          # 필요 시 0.88로 낮추면 더 안정(OP 12%)
UPDATE_RATIOS = [0.8]
HOT_WEIGHTS   = [0.7]
SEEDS         = [101]

# ATCB weight ablation (alpha, beta, gamma, eta)
ATCB_WEIGHTS = [
    (0.5, 0.3, 0.1, 0.1),
]

# 안전 마진(여기서 바꾸면 모든 런에 반영)
WARMUP_FILL = 0.60
GC_FBT      = 0.12            # gc_free_block_threshold (기본 0.05 → 0.12)
BG_GC_EVERY = 256             # 주기적 백그라운드 GC

TAG = "atcb"                  # 결과 폴더 접미사

# ---------- 결과 폴더 ----------
TODAY   = datetime.now().strftime("%Y-%m-%d")
DAY_DIR = os.path.join("results", TODAY)
os.makedirs(DAY_DIR, exist_ok=True)

def next_run_dir(base: str, tag: str = "") -> str:
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

# ---------- 공통 유틸 ----------
def run(cmd):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)

# 공통 인자(여기 꼭 임계치/배경GC/워밍업 포함!)
BASE_ARGS = [
    "--ops", str(OPS),
    "--hot_ratio", str(HOT_RATIO),
    "--warmup_fill", str(WARMUP_FILL),
    "--gc_free_block_threshold", str(GC_FBT),
    "--bg_gc_every", str(BG_GC_EVERY),
    "--out_dir", OUT_DIR,
    "--out_csv", "results.csv",          # OUT_DIR 하위에 저장
]

# ---------- 메타 스냅샷 ----------
def git_commit_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None

meta = {
    "ops": OPS,
    "hot_ratio": HOT_RATIO,
    "user_caps": USER_CAPS,
    "update_ratios": UPDATE_RATIOS,
    "hot_weights": HOT_WEIGHTS,
    "seeds": SEEDS,
    "atcb_weights": ATCB_WEIGHTS,
    "warmup_fill": WARMUP_FILL,
    "gc_free_block_threshold": GC_FBT,
    "bg_gc_every": BG_GC_EVERY,
    "git_commit": git_commit_short(),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "out_dir": OUT_DIR,
}
with open(os.path.join(OUT_DIR, "sweep_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
with open(os.path.join("results", "LATEST.txt"), "w", encoding="utf-8") as f:
    f.write(OUT_DIR)

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

    # ATCB
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

print(f"[OK] Sweep complete. Analyze with:\n  python analyze_results.py --base {OUT_DIR}")