import os
import sys
import csv
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Args ----------------
ap = argparse.ArgumentParser(description="Analyze SSD GC results.")
ap.add_argument("--base", default="results",
                help=r"Base dir (e.g., results\2025-10-04\run01_atcb or results\2025-10-04)")
ap.add_argument("--csv", default="results.csv",
                help="CSV file name to read inside base or its subdirs (default: results.csv)")
ap.add_argument("--merge-subdirs", action="store_true",
                help="Merge all subdirs under --base that contain the CSV, then analyze.")
ap.add_argument("--use-latest", action="store_true",
                help="Use path stored in results/LATEST.txt (overrides --base).")
args = ap.parse_args()

def resolve_base():
    if args.use_latest:
        latest = os.path.join("results", "LATEST.txt")
        if os.path.exists(latest):
            with open(latest, "r", encoding="utf-8") as f:
                b = f.read().strip()
            if b:
                return b
            print("[WARN] LATEST.txt is empty, falling back to --base.")
        else:
            print("[WARN] results/LATEST.txt not found, falling back to --base.")
    return args.base

BASE = resolve_base()
CSV_NAME = args.csv

# ---------------- Reader (flex schema) ----------------
SCHEMA17 = [
    "policy","ops","update_ratio","hot_ratio","hot_weight","seed",
    "host_writes","device_writes","WAF","GC_count",
    "avg_erase","min_erase","max_erase","wear_delta",
    "free_pages","total_pages","note",
]
SCHEMA22 = [
    "policy","ops","update_ratio","hot_ratio","hot_weight","seed",
    "host_writes","device_writes","WAF","GC_count",
    "avg_erase","min_erase","max_erase","wear_delta",
    "free_pages","total_pages",
    "gc_time_total_ms","gc_time_avg_ms","gc_time_p50_ms","gc_time_p95_ms","gc_time_p99_ms",
    "note",
]

def read_results_flex(path: str) -> pd.DataFrame:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r: continue
            if r[0] == "policy": continue
            L = len(r)
            if L == len(SCHEMA22):
                keys = SCHEMA22
            elif L == len(SCHEMA17):
                keys = SCHEMA17
            elif L > len(SCHEMA22):
                merged = r[:len(SCHEMA22)-1] + [",".join(r[len(SCHEMA22)-1:])]
                r = merged; keys = SCHEMA22
            else:
                r = r + [""] * (len(SCHEMA22) - L)
                keys = SCHEMA22
            rows.append(dict(zip(keys, r)))
    df = pd.DataFrame(rows)
    numeric_cols = [
        "ops","update_ratio","hot_ratio","hot_weight","seed",
        "host_writes","device_writes","WAF","GC_count",
        "avg_erase","min_erase","max_erase","wear_delta",
        "free_pages","total_pages",
        "gc_time_total_ms","gc_time_avg_ms","gc_time_p50_ms","gc_time_p95_ms","gc_time_p99_ms",
        "wear_std","wear_cv","wear_gini",  # optional (newer CSVs)
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------------- Load (single or merge) ----------------
def load_single(base: str, csv_name: str) -> pd.DataFrame:
    csv_path = os.path.join(base, csv_name)
    if not os.path.exists(csv_path):
        print(f"[ERROR] Results CSV not found: {csv_path}")
        sys.exit(1)
    df = read_results_flex(csv_path)
    df["__source__"] = base
    return df

def load_merge_subdirs(base: str, csv_name: str) -> pd.DataFrame:
    # candidates: base/*/csv_name (1 depth)
    pats = glob.glob(os.path.join(base, "*", csv_name))
    if not pats:
        print(f"[ERROR] No subdir CSVs found under: {base} (looked for */{csv_name})")
        sys.exit(1)
    dfs = []
    for p in sorted(pats):
        try:
            df = read_results_flex(p)
            df["__source__"] = os.path.dirname(p)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    if not dfs:
        print("[ERROR] No readable CSVs found.")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)

if args.merge_subdirs:
    df = load_merge_subdirs(BASE, CSV_NAME)
    OUT = os.path.join(BASE, "plots_merged")
    CLEAN_OUT = os.path.join(BASE, "results_merged_clean.csv")
else:
    df = load_single(BASE, CSV_NAME)
    OUT = os.path.join(BASE, "plots")
    CLEAN_OUT = os.path.join(BASE, "results_clean.csv")

os.makedirs(OUT, exist_ok=True)
df.to_csv(CLEAN_OUT, index=False)

if df.empty:
    print("[WARN] No rows in results; nothing to plot/summarize.")
    sys.exit(0)

# ---------------- Plot helpers ----------------
def run_name(row):
    note = str(row.get("note") or "").strip()
    src  = os.path.basename(str(row.get("__source__", "")))
    base = f'{row["policy"]} ({note})' if note else str(row["policy"])
    # 소스 폴더(run01 등)도 보이게 하면 병합 때 구분이 쉬움
    return f"{base} | {src}" if src else base

df["run"] = df.apply(run_name, axis=1)

def save_bar(ycol: str, title: str, fname: str):
    if ycol in df.columns and df[ycol].notna().any():
        ax = df.plot(kind="bar", x="run", y=ycol, legend=False, figsize=(12,5), title=title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, fname), bbox_inches="tight")
        plt.close()

save_bar("WAF", "WAF by run", "waf_by_run.png")
save_bar("GC_count", "GC count by run", "gc_by_run.png")
save_bar("gc_time_p99_ms", "GC time p99 (ms) by run", "gc_p99_by_run.png")

# ---------------- Summaries (columns-if-exist) ----------------
def cols_exist(cols): return [c for c in cols if c in df.columns]

summary_cols = cols_exist(["WAF","GC_count","gc_time_p99_ms","wear_delta","wear_std","wear_cv","wear_gini"])
g1 = df[["policy"] + summary_cols].groupby("policy", dropna=False).mean(numeric_only=True).reset_index()
if "WAF" in g1.columns:
    g1 = g1.sort_values("WAF")
g1.to_csv(os.path.join(BASE, "results_summary_policy.csv"), index=False)

have_cols = cols_exist(["WAF","GC_count","gc_time_p99_ms","wear_delta","wear_std","wear_cv","wear_gini"])
keys = cols_exist(["update_ratio","hot_weight"]) + ["policy"]
g2 = df[keys + have_cols].groupby(keys, dropna=False).mean(numeric_only=True).reset_index()
g2.to_csv(os.path.join(BASE, "results_summary_by_workload.csv"), index=False)

def improvement_vs_greedy(dfin: pd.DataFrame) -> pd.DataFrame:
    base_keys = cols_exist(["update_ratio","hot_weight"])
    if not base_keys: return pd.DataFrame()
    if "greedy" not in dfin["policy"].unique(): return pd.DataFrame()
    sub = dfin[base_keys + ["policy","WAF","GC_count"]].copy()
    piv = sub.pivot_table(index=base_keys, columns="policy", values=["WAF","GC_count"], aggfunc="mean")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    for met in ["WAF","GC_count"]:
        gcol = f"{met}_greedy"
        for pol in ["cb","bsgc","atcb"]:
            pcol = f"{met}_{pol}"
            if gcol in piv.columns and pcol in piv.columns:
                piv[f"impr_{met}_{pol}_pct"] = (piv[gcol] - piv[pcol]) / piv[gcol] * 100.0
    return piv.reset_index()

g3 = improvement_vs_greedy(df)
g3.to_csv(os.path.join(BASE, "results_improvement_vs_greedy.csv"), index=False)

# ---------------- Console prints ----------------
print("\n[SUMMARY] Policy means (sorted by WAF if available):")
print(g1.to_string(index=False))

if not g3.empty:
    show_cols = [c for c in g3.columns if c.startswith("impr_")]
    if show_cols:
        print("\n[IMPROVEMENTS] Greedy vs others (%):")
        print(g3[cols_exist(['update_ratio','hot_weight']) + show_cols].to_string(index=False))

print(f"\n[OK] Saved plots -> {OUT}")
print("[OK] CSVs ->",
      CLEAN_OUT, ",",
      os.path.join(BASE, "results_summary_policy.csv"), ",",
      os.path.join(BASE, "results_summary_by_workload.csv"), ",",
      os.path.join(BASE, "results_improvement_vs_greedy.csv"))
