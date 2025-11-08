from __future__ import annotations
import os
import sys
import argparse
import glob
from typing import List, Dict, Any

# pandas/matplotlib 의존 — 없으면 에러 메시지 명확히
try:
    import pandas as pd
except Exception as e:
    print("[analyze_results] pandas가 필요합니다: pip install pandas", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("[analyze_results] matplotlib이 필요합니다: pip install matplotlib", file=sys.stderr)
    raise


# ------------------------------------------------------------
# 파일 수집/로딩
# ------------------------------------------------------------

def _find_summary_csvs(base_dir: str, merge_subdirs: bool, filename: str = "summary.csv") -> List[str]:
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"base 디렉토리가 존재하지 않습니다: {base_dir}")
    paths: List[str] = []
    if merge_subdirs:
        for p in glob.glob(os.path.join(base_dir, "**", filename), recursive=True):
            paths.append(p)
    else:
        p = os.path.join(base_dir, filename)
        if os.path.exists(p):
            paths.append(p)
    return sorted(paths)


def _read_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source__"] = p
            frames.append(df)
        except Exception as e:
            print(f"[WARN] CSV 읽기 실패: {p}: {e}")
    if not frames:
        raise RuntimeError("읽을 수 있는 summary CSV가 없습니다.")
    # 컬럼 차이를 보존하면서 병합(outer)
    all_cols = sorted(set().union(*[set(f.columns) for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# 시각화 유틸
# ------------------------------------------------------------

def _ensure_out_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _safe_float(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def plot_waf_by_policy(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    col_policy = "policy"
    col_waf = "waf"
    if col_policy not in df.columns or col_waf not in df.columns:
        print("[plot] 필요한 컬럼이 없습니다(policy, waf)")
        return
    # 박스플롯: 정책별 WAF 분포
    order = sorted(df[col_policy].dropna().unique())
    data = [df[df[col_policy]==p][col_waf].astype(float) for p in order]
    plt.figure()
    plt.boxplot(data, tick_labels=order, showmeans=True)
    plt.title("WAF by Policy")
    plt.ylabel("WAF")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gc_vs_ops(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    if not {"gc_count","ops"}.issubset(df.columns):
        print("[plot] 필요한 컬럼이 없습니다(gc_count, ops)")
        return
    plt.figure()
    plt.scatter(df["ops"].astype(float), df["gc_count"].astype(float), s=12, alpha=0.6)
    plt.xlabel("ops")
    plt.ylabel("gc_count")
    plt.title("GC Count vs Ops")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_waf_vs_update_ratio(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    if not {"waf","update_ratio"}.issubset(df.columns):
        print("[plot] 필요한 컬럼이 없습니다(waf, update_ratio)")
        return
    plt.figure()
    plt.scatter(df["update_ratio"].astype(float), df["waf"].astype(float), s=12, alpha=0.6)
    plt.xlabel("update_ratio")
    plt.ylabel("WAF")
    plt.title("WAF vs Update Ratio")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_wear_hist(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    cols = [c for c in df.columns if c.startswith("wear_")]
    need = {"wear_min","wear_avg","wear_max"}
    if not need.issubset(set(cols)):
        print("[plot] 필요한 컬럼이 없습니다(wear_min/wear_avg/wear_max)")
        return
    # 평균 wear의 히스토그램
    plt.figure()
    plt.hist(df["wear_avg"].astype(float), bins=30, alpha=0.8)
    plt.xlabel("wear_avg")
    plt.ylabel("count")
    plt.title("Wear Average Distribution")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trim_vs_policy(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    if not {"policy","trimmed_pages"}.issubset(df.columns):
        print("[plot] 필요한 컬럼이 없습니다(policy, trimmed_pages)")
        return
    # 정책별 평균 트림 페이지 수
    g = df.groupby("policy")["trimmed_pages"].mean().sort_values()
    plt.figure()
    g.plot(kind="bar")
    plt.ylabel("avg trimmed pages")
    plt.title("Trimmed Pages by Policy (avg)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stability(df: pd.DataFrame, out_path: str):
    _ensure_out_dir(out_path)
    if not {"transition_rate","reheat_rate"}.issubset(df.columns):
        print("[plot] 필요한 컬럼이 없습니다(transition_rate, reheat_rate)")
        return
    plt.figure()
    plt.scatter(df["transition_rate"].astype(float), df["reheat_rate"].astype(float), s=12, alpha=0.6)
    plt.xlabel("transition_rate")
    plt.ylabel("reheat_rate")
    plt.title("Stability Snapshot")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# 메인
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze GC results (merge/plots)")
    ap.add_argument("--base", type=str, required=True, help="기준 디렉토리")
    ap.add_argument("--merge-subdirs", action="store_true", help="하위 폴더의 summary.csv까지 병합")
    ap.add_argument("--filename", type=str, default="summary.csv", help="요약 파일명(기본 summary.csv)")
    ap.add_argument("--out_csv", type=str, default=None, help="병합 결과 CSV 저장 경로")
    ap.add_argument("--plots_dir", type=str, default=None, help="플롯 저장 디렉토리 (지정 시 기본 플롯 생성)")
    args = ap.parse_args()

    csvs = _find_summary_csvs(args.base, args.merge_subdirs, filename=args.filename)
    if not csvs:
        print("[analyze_results] 합칠 CSV가 없습니다.")
        return

    df = _read_csvs(csvs)

    # 병합 CSV 저장
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"[analyze_results] merged CSV saved: {args.out_csv}  (rows={len(df)})")

    # 플롯
    if args.plots_dir:
        os.makedirs(args.plots_dir, exist_ok=True)
        plot_waf_by_policy(df, os.path.join(args.plots_dir, "waf_by_policy.png"))
        plot_gc_vs_ops(df, os.path.join(args.plots_dir, "gc_vs_ops.png"))
        plot_waf_vs_update_ratio(df, os.path.join(args.plots_dir, "waf_vs_update_ratio.png"))
        plot_wear_hist(df, os.path.join(args.plots_dir, "wear_avg_hist.png"))
        plot_trim_vs_policy(df, os.path.join(args.plots_dir, "trim_by_policy.png"))
        plot_stability(df, os.path.join(args.plots_dir, "stability_scatter.png"))
        print(f"[analyze_results] plots saved to: {args.plots_dir}")

    # 콘솔에 머리 몇 줄만 미리보기
    preview_cols = [c for c in [
        "policy","ops","seed","waf","gc_count","free_blocks",
        "wear_avg","wear_std","trimmed_pages","transition_rate","reheat_rate","__source__"
    ] if c in df.columns]
    try:
        print(df[preview_cols].head(20).to_string(index=False))
    except Exception:
        print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()