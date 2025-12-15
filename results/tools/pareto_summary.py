
import os, sys, argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default="results/cota_verify/summary.csv")
    ap.add_argument("--tune",   default="results/cota_tune/summary.csv")
    ap.add_argument("--ops",    type=int, default=200000)
    ap.add_argument("--wear_cut", type=float, default=None)
    args = ap.parse_args()

    # ?? ????: verify -> tune
    for cand in (args.verify, args.tune):
        if os.path.exists(cand):
            src = cand; break
    else:
        print("summary.csv not found in results/cota_verify or results/cota_tune"); sys.exit(1)

    df = pd.read_csv(src)
    if "ops" in df.columns:
        df = df[df["ops"].eq(args.ops)]

    key = "policy" if "policy" in df.columns else "gc_policy"
    print(f"[source] {src} | ops={args.ops}\n")

    # ??? WAF ???
    med = df.groupby(key)["waf"].median().sort_values()
    print("Median WAF by policy:"); print(med.to_string(), "\n")

    base = med.get("greedy", np.nan)
    cota = med.get("cota",   np.nan)
    if pd.notna(base) and pd.notna(cota):
        abs_gain = base - cota
        rel_gain = (base / cota - 1.0) * 100.0
        print(f"COTA vs Greedy: {abs_gain:.4f} abs, {rel_gain:.2f}% rel\n")
    else:
        print("Greedy missing ? skip gain calc\n")

    # COTA ??? ?? (?? ??? ??)
    cols = [c for c in ["gc_policy","cota_alpha","cota_beta","cota_gamma","cota_delta"] if c in df.columns]
    cota_df = df[df[key].eq("cota")].groupby(cols).agg(
        waf_med=("waf","median"),
        wear_med=("wear_std","median"),
        gc_med=("gc_count","median")
    ).reset_index()

    # ?? Pareto (waf?, wear?)
    pts = cota_df[["waf_med","wear_med"]].to_numpy()
    is_dom = np.zeros(len(cota_df), dtype=bool)
    for i,(waf_i,wear_i) in enumerate(pts):
        better_or_equal = (pts[:,0] <= waf_i) & (pts[:,1] <= wear_i)
        strictly_better = (pts[:,0] <  waf_i) |  (pts[:,1] <  wear_i)
        is_dom[i] = np.any(better_or_equal & strictly_better)
    pareto = cota_df[~is_dom].sort_values(["waf_med","wear_med"])
    print("Pareto COTA (waf_med, wear_med) top:")
    print(pareto.head(12).to_string(index=False))

    if args.wear_cut is not None:
        filt = cota_df.query("wear_med < @args.wear_cut").sort_values(["waf_med","wear_med"]).head(12)
        print(f"\n(wear_med < {args.wear_cut}) top:")
        print(filt.to_string(index=False))

if __name__ == "__main__":
    main()
