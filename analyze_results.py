from __future__ import annotations

"""
analyze_results.py

SSD GC 시뮬레이터의 실험 결과(summary.csv)를 수집/병합하고,
기본적인 분석 플롯을 생성하는 “후처리(analysis) 엔트리포인트”입니다.

이 스크립트의 목적은 크게 3가지입니다.
1) 결과 수집(Collection)
   - 여러 실험 폴더에 흩어진 summary.csv를 찾아 한 번에 병합합니다.
2) 재현/추적 가능성(Reproducibility / Provenance)
   - 병합한 각 row에 "__source__" 컬럼을 추가해, 해당 결과가 어떤 파일에서 왔는지 추적 가능합니다.
3) 빠른 검증/요약(Quick sanity-check)
   - 정책별 WAF, wear_avg 분포 등 핵심 플롯을 자동으로 생성해
     결과가 “정상 범위에서 움직이는지” 빠르게 확인할 수 있습니다.

입력/출력 계약(Contract)
-----------------------
Input:
- base_dir 아래의 summary.csv (기본 파일명) 또는 --filename으로 지정한 파일명
- summary.csv는 파일마다 컬럼이 다를 수 있으나, 최소한 일부 공통 컬럼이 존재해야 합니다.
  (이 스크립트는 outer-merge를 사용하여 컬럼이 달라도 합칠 수 있게 설계됨)

Output (옵션):
- --out_csv: 병합된 전체 결과를 CSV로 저장
- --plots_dir: 기본 플롯(PNG) 저장
- 콘솔: 필터 적용 후 미리보기(상위 N행) 출력

재현성(Reproducibility) 포인트
------------------------------
- "__source__" 컬럼: row 단위 출처 추적(어느 run의 결과인지)
- 숫자형 안전 변환: pd.to_numeric(errors="coerce")로 문자열/결측 혼입에도 견고하게 처리

가정/주의(Assumptions / Pitfalls)
---------------------------------
- 숫자 컬럼이 문자열로 저장된 경우 astype(float)가 실패할 수 있어, 안전 변환 유틸을 사용합니다.
- 비율(float) 값은 저장 방식에 따라 미세 오차가 있을 수 있습니다.
  (필터 결과가 비면 np.isclose 방식으로 변경 고려)
- matplotlib은 헤드리스 환경에서도 저장되도록 Agg 백엔드를 사용합니다.
"""

import os
import sys
import argparse
import glob
from typing import List, Optional

# pandas 의존 — 없으면 에러 메시지를 명확히
try:
    import pandas as pd
except Exception:
    print("[analyze_results] pandas가 필요합니다: pip install pandas", file=sys.stderr)
    raise

# matplotlib 의존 (+ 헤드리스 환경 안전)
try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이 파일 저장만 수행
    import matplotlib.pyplot as plt
except Exception:
    print("[analyze_results] matplotlib이 필요합니다: pip install matplotlib", file=sys.stderr)
    raise


# ------------------------------------------------------------
# 1) 결과 파일 수집 / 로딩
# ------------------------------------------------------------

def _find_summary_csvs(base_dir: str, merge_subdirs: bool, filename: str = "summary.csv") -> List[str]:
    """
    base_dir에서 summary.csv(또는 filename) 경로를 수집합니다.

    Parameters
    ----------
    base_dir : str
        기준 디렉토리(존재해야 함).
    merge_subdirs : bool
        True면 base_dir 하위 폴더를 재귀적으로 탐색합니다(**/filename).
        False면 base_dir 바로 아래의 filename만 확인합니다.
    filename : str
        찾을 결과 파일명(기본: summary.csv).

    Returns
    -------
    List[str]
        발견된 파일 경로 리스트(정렬됨). 없으면 빈 리스트.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"[analyze_results] base 디렉토리가 존재하지 않습니다: {base_dir}")

    if merge_subdirs:
        paths = glob.glob(os.path.join(base_dir, "**", filename), recursive=True)
    else:
        p = os.path.join(base_dir, filename)
        paths = [p] if os.path.exists(p) else []

    return sorted(paths)


def _read_csvs(paths: List[str]) -> pd.DataFrame:
    """
    여러 summary.csv를 읽어 하나의 DataFrame으로 병합합니다.

    설계 의도
    --------
    - 실험 시점/버전/옵션에 따라 summary.csv 컬럼이 달라질 수 있으므로,
      컬럼 차이를 유지한 채로 합치기 위해 outer-merge(concat + reindex)를 사용합니다.
    - 각 행에 "__source__"를 추가하여 출처 추적을 가능하게 합니다.

    Failure handling
    ---------------
    - 일부 파일이 손상/형식 문제로 읽기 실패해도 전체 파이프라인이 멈추지 않도록
      경고를 출력하고 해당 파일은 스킵합니다.
    - 다 실패하면 RuntimeError를 발생시킵니다.
    """
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source__"] = p
            frames.append(df)
        except Exception as e:
            print(f"[WARN] CSV 읽기 실패: {p}: {e}", file=sys.stderr)

    if not frames:
        raise RuntimeError("[analyze_results] 읽을 수 있는 summary CSV가 없습니다.")

    # 컬럼의 union을 만들고, 각 DF를 동일 컬럼셋으로 reindex한 뒤 concat
    all_cols = sorted(set().union(*[set(f.columns) for f in frames]))
    frames = [f.reindex(columns=all_cols) for f in frames]
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# 2) 데이터 안전 처리 / 필터
# ------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성합니다."""
    os.makedirs(path, exist_ok=True)


def _ensure_out_dir(file_path: str) -> None:
    """파일 저장 경로의 부모 디렉토리가 없으면 생성합니다."""
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    """
    안전한 숫자 변환 유틸.

    - 숫자처럼 보이는 문자열/공백/결측 등이 섞여 있을 때 astype(float)가 실패할 수 있습니다.
    - pd.to_numeric(errors="coerce")는 변환 불가 값을 NaN으로 바꿔 파이프라인을 견고하게 만듭니다.
    """
    return pd.to_numeric(s, errors="coerce")


def _coerce_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    df에 존재하는 숫자 후보 컬럼들을 미리 숫자형으로 강제 변환합니다.

    목적
    ----
    - 이후 필터 비교/플롯에서 타입 문제로 깨지는 것을 줄입니다.
    - 변환 불가 값은 NaN으로 남아 플롯/집계에서 자연스럽게 제외(dropna) 가능.

    NOTE:
    - 이 함수는 df를 copy하여 반환합니다(원본 변경 방지).
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = _to_num(out[c])
    return out


def apply_filters(
    df: pd.DataFrame,
    policy: Optional[str] = None,
    filter_ops: Optional[int] = None,
    filter_update_ratio: Optional[float] = None,
    filter_hot_ratio: Optional[float] = None,
    min_waf: Optional[float] = None,
    max_waf: Optional[float] = None,
) -> pd.DataFrame:
    """
    사용자가 원하는 조건만 빠르게 보기 위한 필터를 적용합니다.

    이 필터는 논문용 고정 필터가 아니라,
    결과 탐색/재현 확인/부분 비교를 위한 실용적인 필터입니다.

    Parameters
    ----------
    policy : Optional[str]
        특정 정책만 보기 (예: "cota")
    filter_ops : Optional[int]
        특정 ops만 보기
    filter_update_ratio : Optional[float]
        특정 update_ratio만 보기
    filter_hot_ratio : Optional[float]
        특정 hot_ratio만 보기
    min_waf, max_waf : Optional[float]
        WAF 범위 제한

    Returns
    -------
    pd.DataFrame
        필터 적용 후 DataFrame(원본은 유지).
    """
    out = df.copy()

    if policy is not None and "policy" in out.columns:
        out = out[out["policy"] == policy]

    if filter_ops is not None and "ops" in out.columns:
        out = out[_to_num(out["ops"]) == float(filter_ops)]

    # NOTE: ratio 비교는 데이터 저장 방식에 따라 미세 오차 문제가 있을 수 있습니다.
    #       필터가 비는 일이 잦으면 np.isclose 비교로 바꾸는 것을 추천합니다.
    if filter_update_ratio is not None and "update_ratio" in out.columns:
        out = out[_to_num(out["update_ratio"]) == float(filter_update_ratio)]

    if filter_hot_ratio is not None and "hot_ratio" in out.columns:
        out = out[_to_num(out["hot_ratio"]) == float(filter_hot_ratio)]

    if min_waf is not None and "waf" in out.columns:
        out = out[_to_num(out["waf"]) >= float(min_waf)]

    if max_waf is not None and "waf" in out.columns:
        out = out[_to_num(out["waf"]) <= float(max_waf)]

    return out


# ------------------------------------------------------------
# 3) 플롯 생성
# ------------------------------------------------------------

def plot_waf_by_policy(df: pd.DataFrame, out_path: str) -> None:
    """
    정책별 WAF 분포(박스플롯)를 저장합니다.

    Input contract
    --------------
    - df에 policy, waf 컬럼이 존재해야 합니다.

    Interpretation
    --------------
    - WAF는 실험에서 쓰기 증폭 정도를 나타내는 핵심 지표입니다.
    - 박스플롯은 seed 반복 결과의 분포/변동성을 함께 보여줍니다.
    """
    _ensure_out_dir(out_path)
    if not {"policy", "waf"}.issubset(df.columns):
        print("[plot] skip: missing columns(policy, waf)")
        return

    order = sorted(df["policy"].dropna().unique())
    data = [_to_num(df.loc[df["policy"] == p, "waf"]).dropna() for p in order]

    plt.figure()
    plt.boxplot(data, labels=order, showmeans=True)
    plt.title("WAF by Policy")
    plt.ylabel("WAF")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_gc_vs_ops(df: pd.DataFrame, out_path: str) -> None:
    """
    ops 대비 gc_count 산점도를 저장합니다.

    Input contract
    --------------
    - df에 ops, gc_count 컬럼이 존재해야 합니다.

    Use case
    --------
    - 실험 규모(ops) 대비 GC 호출 빈도가 비정상적으로 튀는지 빠르게 확인(QC).
    """
    _ensure_out_dir(out_path)
    if not {"gc_count", "ops"}.issubset(df.columns):
        print("[plot] skip: missing columns(gc_count, ops)")
        return

    x = _to_num(df["ops"])
    y = _to_num(df["gc_count"])

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.xlabel("ops")
    plt.ylabel("gc_count")
    plt.title("GC Count vs Ops")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_waf_vs_update_ratio(df: pd.DataFrame, out_path: str) -> None:
    """
    update_ratio 대비 WAF 산점도를 저장합니다.

    Input contract
    --------------
    - df에 update_ratio, waf 컬럼이 존재해야 합니다.

    Use case
    --------
    - update_ratio 변화에 따른 WAF 민감도/추세를 확인(QC 및 탐색).
    """
    _ensure_out_dir(out_path)
    if not {"waf", "update_ratio"}.issubset(df.columns):
        print("[plot] skip: missing columns(waf, update_ratio)")
        return

    x = _to_num(df["update_ratio"])
    y = _to_num(df["waf"])

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.xlabel("update_ratio")
    plt.ylabel("WAF")
    plt.title("WAF vs Update Ratio")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_wear_avg_by_policy(df: pd.DataFrame, out_path: str) -> None:
    """
    정책별 wear_avg 분포(박스플롯)를 저장합니다.

    Input contract
    --------------
    - df에 policy, wear_avg 컬럼이 존재해야 합니다.

    Interpretation
    --------------
    - wear_avg는 평균 마모 수준을 나타내는 지표입니다.
    - wear_std와 함께 보면 얼마나 고르게 마모를 분배했는지 해석에 도움이 됩니다.
    """
    _ensure_out_dir(out_path)
    if not {"policy", "wear_avg"}.issubset(df.columns):
        print("[plot] skip: missing columns(policy, wear_avg)")
        return

    order = sorted(df["policy"].dropna().unique())
    data = [_to_num(df.loc[df["policy"] == p, "wear_avg"]).dropna() for p in order]

    plt.figure()
    plt.boxplot(data, labels=order, showmeans=True)
    plt.title("wear_avg by Policy")
    plt.ylabel("wear_avg")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_trim_by_policy(df: pd.DataFrame, out_path: str) -> None:
    """
    정책별 trimmed_pages 평균(bar plot)을 저장합니다.

    Input contract
    --------------
    - df에 policy, trimmed_pages 컬럼이 존재해야 합니다.

    Interpretation
    --------------
    - trimmed_pages는 TRIM으로 무효화된 페이지 수(또는 처리된 양)를 나타냅니다.
    - TRIM 사용 실험이라면 정책/조건에 따른 차이를 확인할 수 있습니다.
    """
    _ensure_out_dir(out_path)
    if not {"policy", "trimmed_pages"}.issubset(df.columns):
        print("[plot] skip: missing columns(policy, trimmed_pages)")
        return

    tmp = df.copy()
    tmp["trimmed_pages"] = _to_num(tmp["trimmed_pages"])
    g = tmp.groupby("policy")["trimmed_pages"].mean().sort_values()

    plt.figure()
    g.plot(kind="bar")
    plt.ylabel("avg trimmed pages")
    plt.title("Trimmed Pages by Policy (avg)")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_stability(df: pd.DataFrame, out_path: str) -> None:
    """
    transition_rate vs reheat_rate 산점도를 저장합니다.

    Input contract
    --------------
    - df에 transition_rate, reheat_rate 컬럼이 존재해야 합니다.

    Use case
    --------
    - 프로젝트에서 정의한 상태 전이/재가열 지표의 분포를 빠르게 스냅샷으로 확인.
    """
    _ensure_out_dir(out_path)
    if not {"transition_rate", "reheat_rate"}.issubset(df.columns):
        print("[plot] skip: missing columns(transition_rate, reheat_rate)")
        return

    x = _to_num(df["transition_rate"])
    y = _to_num(df["reheat_rate"])

    plt.figure()
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.xlabel("transition_rate")
    plt.ylabel("reheat_rate")
    plt.title("Stability Snapshot")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# 4) 엔트리포인트
# ------------------------------------------------------------

def main() -> None:
    """
    CLI 엔트리포인트.

    Flow
    ----
    1) base 디렉토리에서 summary.csv 수집
    2) 파일들을 병합(df)
    3) 숫자 컬럼 안전 변환
    4) 사용자 필터 적용(df_view)
    5) (옵션) 병합 CSV 저장
    6) (옵션) 플롯 저장
    7) 콘솔 미리보기 출력
    """
    ap = argparse.ArgumentParser(description="Analyze GC results (merge/plots)")
    ap.add_argument("--base", type=str, required=True, help="기준 디렉토리")
    ap.add_argument("--merge-subdirs", action="store_true", help="하위 폴더의 summary.csv까지 병합")
    ap.add_argument("--filename", type=str, default="summary.csv", help="요약 파일명(기본 summary.csv)")
    ap.add_argument("--out_csv", type=str, default=None, help="병합 결과 CSV 저장 경로")
    ap.add_argument("--plots_dir", type=str, default=None, help="플롯 저장 디렉토리 (지정 시 기본 플롯 생성)")
    ap.add_argument("--preview", type=int, default=20, help="콘솔 미리보기 행 수")

    # 선택 필터(재현/탐색용)
    ap.add_argument("--policy", type=str, default=None, help="특정 policy만 보기 (예: cota)")
    ap.add_argument("--filter_ops", type=int, default=None, help="특정 ops만 보기 (예: 200000)")
    ap.add_argument("--filter_update_ratio", type=float, default=None, help="특정 update_ratio만 보기 (예: 0.8)")
    ap.add_argument("--filter_hot_ratio", type=float, default=None, help="특정 hot_ratio만 보기 (예: 0.2)")
    ap.add_argument("--min_waf", type=float, default=None, help="waf 하한 필터")
    ap.add_argument("--max_waf", type=float, default=None, help="waf 상한 필터")

    args = ap.parse_args()

    # 1) 수집
    csvs = _find_summary_csvs(args.base, args.merge_subdirs, filename=args.filename)
    if not csvs:
        print("[analyze_results] 합칠 CSV가 없습니다.")
        return

    # 2) 병합
    df = _read_csvs(csvs)

    # 3) 숫자형 안전 변환(플롯/필터의 안정성 목적)
    df = _coerce_numeric_cols(df, [
        "ops", "seed", "waf", "gc_count", "free_blocks",
        "wear_avg", "wear_std", "trimmed_pages",
        "update_ratio", "hot_ratio",
        "transition_rate", "reheat_rate",
    ])

    # 4) 필터 적용(사용자가 보고 싶은 subset)
    df_view = apply_filters(
        df,
        policy=args.policy,
        filter_ops=args.filter_ops,
        filter_update_ratio=args.filter_update_ratio,
        filter_hot_ratio=args.filter_hot_ratio,
        min_waf=args.min_waf,
        max_waf=args.max_waf,
    )

    print(f"[analyze_results] rows: view={len(df_view)} / total={len(df)}")
    if len(df_view) == 0:
        print("[analyze_results] (WARN) 필터 결과가 비었습니다. 조건을 완화해보세요.")

    # 5) 병합 CSV 저장(원본 손실 방지를 위해 전체 df 저장이 기본)
    if args.out_csv:
        _ensure_out_dir(args.out_csv)
        df.to_csv(args.out_csv, index=False)
        print(f"[analyze_results] merged CSV saved: {args.out_csv}  (rows={len(df)})")

    # 6) 플롯 생성
    if args.plots_dir:
        _ensure_dir(args.plots_dir)

        # subset 기준(비교/재현을 위해 동일 조건으로 보기 위함)
        plot_waf_by_policy(df_view, os.path.join(args.plots_dir, "waf_by_policy.png"))
        plot_wear_avg_by_policy(df_view, os.path.join(args.plots_dir, "wear_avg_by_policy.png"))

        # 전체 df 기준(QC/전반적인 분포 확인)
        plot_gc_vs_ops(df, os.path.join(args.plots_dir, "gc_vs_ops.png"))
        plot_waf_vs_update_ratio(df, os.path.join(args.plots_dir, "waf_vs_update_ratio.png"))
        plot_trim_by_policy(df, os.path.join(args.plots_dir, "trim_by_policy.png"))
        plot_stability(df, os.path.join(args.plots_dir, "stability_scatter.png"))

        print(f"[analyze_results] plots saved to: {args.plots_dir}")

    # 7) 콘솔 미리보기
    preview_cols = [c for c in [
        "policy", "ops", "update_ratio", "hot_ratio", "seed",
        "waf", "gc_count", "free_blocks",
        "wear_avg", "wear_std", "trimmed_pages",
        "transition_rate", "reheat_rate", "__source__"
    ] if c in df_view.columns]

    n = max(int(args.preview), 0)
    if n > 0:
        try:
            print(df_view[preview_cols].head(n).to_string(index=False))
        except Exception:
            print(df_view.head(n).to_string(index=False))


if __name__ == "__main__":
    main()