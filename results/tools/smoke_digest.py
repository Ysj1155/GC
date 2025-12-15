import argparse, os, sys
import numpy as np
import pandas as pd

def find_source(base: str, filename: str):
    # 우선순위: 명시한 base/filename → cota_verify → cota_tune
    cands = [
        os.path.join(base, filename),
        os.path.join("results", "cota_verify", filename),
        os.path.join("results", "cota_tune", filename),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None

def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    (x: 작을수록 좋음, y: 작을수록 좋음) 기준의 비지배 집합(Pareto front)만 남긴다.
    """
    if df.empty:
        return df
    pts = df[[x_col, y_col]].to_numpy()
    is_dom = np.zeros(len(df), dtype=bool)
    for i, (x_i, y_i) in enumerate(pts):
        better_or_equal = (pts[:,0] <= x_i) & (pts[:,1] <= y_i)
        strictly_better = (pts[:,0] <  x_i) |  (pts[:,1] <  y_i)
        is_dom[i] = np.any(better_or_equal & strictly_better)
    out = df[~is_dom].copy()
    return out.sort_values([x_col, y_col], kind="mergesort")

def main():
    ap = argparse.ArgumentParser(description="Summarize smoke (OPS=80k) runs by hyperparams, rank & Pareto.")
    ap.add_argument("--base", default="results/cota_verify", help="폴더 베이스 (summary.csv가 있는 상위)")
    ap.add_argument("--filename", default="summary.csv", help="요약 CSV 파일명")
    ap.add_argument("--ops", type=int, default=80000, help="스모크 OPS 필터")
    ap.add_argument("--policy", default="cota", help="정책 필터명 (예: cota)")
    ap.add_argument("--wear-cut", type=float, default=None, help="wear_std 중앙값 상한 필터 (예: 1.0)")
    ap.add_argument("--topk", type=int, default=12, help="랭킹 출력 상위 K")
    args = ap.parse_args()

    src = find_source(args.base, args.filename)
    if not src:
        print("summary.csv를 찾을 수 없습니다.")
        sys.exit(1)

    print(f"[source] {src}")

    df = pd.read_csv(src)
    if df.empty:
        print("입력 CSV가 비어 있습니다.")
        sys.exit(0)

    # policy 컬럼명 정규화
    key = "policy" if "policy" in df.columns else ("gc_policy" if "gc_policy" in df.columns else None)
    if key is None:
        print("policy/gc_policy 열이 없습니다.")
        sys.exit(1)

    # 필터: ops, policy
    q = df[key].eq(args.policy) & df["ops"].eq(args.ops)
    sub = df.loc[q].copy()
    if sub.empty:
        print(f"필터 결과 없음: policy={args.policy}, ops={args.ops}")
        sys.exit(0)

    # 하이퍼 파라미터 컬럼 추출
    # - COTA 가중치들
    hyper_cols = [c for c in sub.columns if c.startswith("cota_")]
    # - 선택적 옵션들(있으면 포함)
    for c in ["cold_victim_bias", "victim_prefetch_k"]:
        if c in sub.columns:
            hyper_cols.append(c)
    # policy 컬럼은 그룹키에 굳이 포함할 필요 없음(이미 필터링)
    if not hyper_cols:
        print("하이퍼 파라미터 열(cota_*, cold_victim_bias, victim_prefetch_k)을 찾지 못했습니다.")
        sys.exit(1)

    # seed-중앙값 집계
    agg = (
        sub.groupby(hyper_cols, dropna=False)
           .agg(
               waf_med = ("waf", "median"),
               wear_med = ("wear_std", "median"),
               gc_med = ("gc_count", "median"),
               seeds = ("seed", "nunique"),
           )
           .reset_index()
    )
    if agg.empty:
        print("집계 결과가 비었습니다.")
        sys.exit(0)

    # 선택적 wear 컷
    agg_cut = agg
    if args.wear_cut is not None:
        agg_cut = agg_cut[agg_cut["wear_med"] < args.wear_cut].copy()

    # 랭킹: waf_med 오름차순 → wear_med → gc_med → seeds 내림차순
    ranked = agg_cut.sort_values(
        by=["waf_med", "wear_med", "gc_med", "seeds"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Pareto front (전체 agg 기준; 또는 컷 적용본으로 보고 싶으면 agg_cut로 바꿔도 됨)
    pareto = pareto_front(agg, "waf_med", "wear_med")

    # 출력 경로
    out_dir = os.path.dirname(src)  # summary.csv가 있는 폴더
    top_path = os.path.join(out_dir, "smoke_top.csv")
    pareto_path = os.path.join(out_dir, "smoke_pareto.csv")

    ranked.head(args.topk).to_csv(top_path, index=False)
    pareto.to_csv(pareto_path, index=False)

    # 콘솔 요약
    print("\n[Summary]")
    print(f"- rows in source: {len(df):,}")
    print(f"- rows after filter (policy={args.policy}, ops={args.ops}): {len(sub):,}")
    print(f"- grouped hyper combos: {len(agg):,}")
    if args.wear_cut is not None:
        print(f"- after wear_cut<{args.wear_cut}: {len(agg_cut):,}")

    print(f"\n[Saved]")
    print(f"- TOP  {args.topk:>2} : {top_path}")
    print(f"- Pareto    : {pareto_path}")

    print("\n[Preview: TOP]")
    print(ranked.head(args.topk).to_string(index=False))

    print("\n[Preview: Pareto]")
    print(pareto.head(args.topk).to_string(index=False))

if __name__ == "__main__":
    main()