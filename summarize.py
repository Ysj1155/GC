import pandas as pd, sys
path = sys.argv[1] if len(sys.argv)>1 else "results/comp/summary.csv"
df = pd.read_csv(path)

# 필요한 열만 (있으면 쓰고, 없으면 자동 제외)
cols = [c for c in [
    "policy","ops","seed","waf","gc_count","gc_avg_s",
    "wear_avg","wear_std","wear_min","wear_max",
    "device_writes","host_writes","trimmed_pages",
    "transition_rate","reheat_rate",
    "cota_alpha","cota_beta","cota_gamma","cota_delta"
] if c in df.columns]
df = df[cols]

# 정책별 통계 (중앙값/사분위)
def agg_block(x):
    q = x.quantile
    return pd.Series({
        "waf_med": x["waf"].median(),
        "waf_p25": q(0.25)["waf"], "waf_p75": q(0.75)["waf"],
        "gc_med": x["gc_count"].median(),
        "gc_p25": q(0.25)["gc_count"], "gc_p75": q(0.75)["gc_count"],
        "wear_std_med": x["wear_std"].median(),
        "wear_max_med": x["wear_max"].median() if "wear_max" in x else None
    })

out = df.groupby("policy", as_index=False).apply(agg_block)
out.to_csv("results/comp/summary_by_policy.csv", index=False)
print(out)