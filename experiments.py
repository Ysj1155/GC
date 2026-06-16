from __future__ import annotations

"""
experiments.py

실험 실행(Experiment Runner) 스크립트입니다.

이 스크립트는 `run_sim.py`의 단일 실행 기능을 확장하여,
- 여러 정책/파라미터 조합(grid),
- 여러 시나리오(YAML),
- 여러 seed 반복(repeat)
을 한 번에 돌리고, 결과를 summary.csv로 누적 저장하는 역할을 합니다.

이 파일의 핵심 목적은 3가지입니다.
1) 실행 사양의 표준화(Standardization)
   - 실험을 CLI 옵션(또는 YAML)로 명세하고, 동일한 형식으로 실행합니다.
2) 재현성(Reproducibility)
   - seed/파라미터/정책/시간 정보를 meta로 기록하고, summary.csv에 append합니다.
3) 무결성/품질 검증(QC)
   - 결과 row에 대해 간단한 sanity-check를 수행하여 이상치를 조기에 발견합니다.
     (warn 모드: 경고 출력, strict 모드: 경고 발생 시 즉시 종료)

입력/출력 계약(Contract)
-----------------------
Input:
- Simulator / Workload 생성에 필요한 파라미터들
- (옵션) --grid: "k=v1,v2; k2=v3,v4" 형태의 간단 그리드 명세
- (옵션) --scenarios: YAML 파일(여러 실험 사양 리스트 또는 {scenarios: [...]})

Output:
- results 디렉토리 생성(--out_dir)
- summary.csv에 결과 append(--out_csv)
- 콘솔에 실행별 요약(탭 구분) 출력

재현성(Reproducibility) 포인트
------------------------------
- `meta`에 policy/seed/ratio/hot_weight/trim/warmup/bg_gc_every 및 COTA 관련 파라미터를 기록
- summary.csv는 append 방식이라, 동일 파일에 여러 실험을 “누적 기록”할 수 있음
- "__source__"는 analyze_results.py 병합 시에 추가되는 개념이며,
  experiments.py 단계에서는 out_csv 경로 자체가 provenance 역할을 함

가정/주의(Assumptions / Pitfalls)
---------------------------------
- `gc_algos`는 일부 설정을 전역 함수(config_*)로 받도록 되어 있을 수 있습니다.
  이 경우, 한 프로세스에서 여러 실험을 돌릴 때 전역 설정이 다음 run에 영향을 줄 수 있으니,
  해당 config_* 함수는 실험마다 반드시 호출(또는 기본값 복귀)된다는 전제가 필요합니다.
- YAML 로딩은 PyYAML이 필요합니다.
- warmup_fill은 실험 전 SSD를 일부 채워 초기 과도 상태를 줄이기 위한 옵션이며,
  워밍업 중에도 GC가 발생할 수 있습니다(의도된 동작).

참고
----
- 이 스크립트는 실험 오케스트레이션(관리)에 집중합니다.
  실제 SSD 모델/GC/워크로드의 구체 동작은 simulator.py / gc_algos.py / workload.py를 참고하세요.
"""

import argparse
from typing import Any, Dict, List

from experiment_runner import ensure_dir, run_once


# ------------------------------------------------------------
# Grid 빌더 / 값 파싱
# ------------------------------------------------------------

def _parse_csv_list(s: str) -> List[str]:
    """'a,b,c' 형태 문자열을 리스트로 변환합니다(빈 토큰 제거)."""
    return [x for x in (s or "").split(",") if x != ""]


def _coerce_value(x: str) -> Any:
    """
    그리드/YAML 문자열 값을 적절한 타입으로 캐스팅합니다.

    규칙
    ----
    - "none"/"null" -> None
    - "true"/"false" -> bool
    - 숫자 형태 -> int 또는 float
    - 그 외 -> 원문 문자열
    """
    if x.lower() in ("none", "null"):
        return None
    if x.lower() in ("true", "false"):
        return x.lower() == "true"
    try:
        if "." in x:
            return float(x)
        return int(x)
    except Exception:
        return x


def build_grid(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    --grid 옵션을 파싱하여 매개변수 데카르트 곱 실험 목록을 생성합니다.

    예시
    ----
    --grid gc_policy=cota,greedy; update_ratio=0.5,0.9; seed=1,2
    => 모든 조합(데카르트 곱)을 생성하여 각각 1 run으로 실행 가능하게 합니다.

    반환
    ----
    - 각 원소는 argparse Namespace로 만들 수 있는 dict(conf)입니다.
    """
    items: List[List[tuple[str, Any]]] = []
    if not args.grid:
        return [vars(args).copy()]

    for pair in args.grid.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        vals = _parse_csv_list(v)
        items.append([(k.strip(), _coerce_value(x.strip())) for x in vals])

    grids: List[Dict[str, Any]] = []

    def _dfs(i: int, acc: Dict[str, Any]) -> None:
        if i == len(items):
            d = vars(args).copy()
            d.update(acc)
            grids.append(d)
            return
        for k, v in items[i]:
            acc[k] = v
            _dfs(i + 1, acc)
            acc.pop(k, None)

    _dfs(0, {})
    return grids


# ------------------------------------------------------------
# YAML 시나리오 로더(옵션)
# ------------------------------------------------------------

def load_scenarios(path: str) -> List[Dict[str, Any]]:
    """
    YAML 시나리오 파일을 로드합니다.

    지원 형식
    --------
    1) 리스트 형태:
       - [ {scenario1}, {scenario2}, ... ]
    2) dict + scenarios 키:
       - { scenarios: [ {scenario1}, ... ] }

    각 scenario는 args dict에 merge될 오버라이드 파라미터 모음입니다.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        raise RuntimeError("PyYAML이 필요합니다: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "scenarios" in data:
        return list(data["scenarios"])  # type: ignore
    if isinstance(data, list):
        return data  # type: ignore

    raise RuntimeError("시나리오 파일 형식이 잘못되었습니다 (list 또는 {scenarios: [...]})")


# ------------------------------------------------------------
# CLI 엔트리포인트
# ------------------------------------------------------------

def main() -> None:
    """
    experiments.py CLI 엔트리포인트.

    실행 모드
    --------
    1) YAML 모드: --scenarios <path>
       - YAML의 각 scenario를 args에 merge하여 실행
    2) Grid/단일 모드: --grid ... 또는 단일 args
       - 그리드 조합마다 실행

    repeat 처리
    -----------
    - --repeat N이면 seed를 base_seed부터 +1씩 증가시키며 N번 반복 실행합니다.

    QC 처리
    -------
    - --qc off   : QC 실행하지 않음
    - --qc warn  : QC 경고를 출력만 하고 계속 진행
    - --qc strict: QC 경고가 나오면 즉시 종료
    """
    ap = argparse.ArgumentParser(description="Experiments runner (grid/YAML/multiseed)")

    # 공통 파라미터(기본값은 run_sim.py와 맞춤)
    ap.add_argument("--ops", type=int, default=200_000)
    ap.add_argument("--update_ratio", type=float, default=0.8)
    ap.add_argument("--hot_ratio", type=float, default=0.2)
    ap.add_argument("--hot_weight", type=float, default=0.7)
    ap.add_argument("--blocks", type=int, default=256)
    ap.add_argument("--pages_per_block", type=int, default=64)
    ap.add_argument("--gc_free_block_threshold", type=float, default=0.12)
    ap.add_argument("--user_capacity_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bg_gc_every", type=int, default=0)

    ap.add_argument("--enable_trim", action="store_true")
    ap.add_argument("--trim_ratio", type=float, default=0.0)
    ap.add_argument("--warmup_fill", type=float, default=0.0)

    ap.add_argument(
        "--gc_policy",
        type=str,
        default="greedy",
        choices=["greedy", "cb", "cost_benefit", "bsgc", "cota", "atcb", "re50315"],
    )

    # COTA 확장
    ap.add_argument("--cota_alpha", type=float, default=None)
    ap.add_argument("--cota_beta", type=float, default=None)
    ap.add_argument("--cota_gamma", type=float, default=None)
    ap.add_argument("--cota_delta", type=float, default=None)
    ap.add_argument("--cold_victim_bias", type=float, default=1.0)
    ap.add_argument("--trim_age_bonus", type=float, default=0.0)
    ap.add_argument("--victim_prefetch_k", type=int, default=1)

    # ATCB / RE50315
    ap.add_argument("--atcb_alpha", type=float, default=0.5)
    ap.add_argument("--atcb_beta", type=float, default=0.3)
    ap.add_argument("--atcb_gamma", type=float, default=0.1)
    ap.add_argument("--atcb_eta", type=float, default=0.1)
    ap.add_argument("--re50315_K", type=float, default=1.0)

    # Sweep 옵션
    ap.add_argument("--grid", type=str, default=None, help="키=값1,값2; 키2=... 형식")
    ap.add_argument("--repeat", type=int, default=1, help="시드 반복 횟수(시작 시드부터 +1 증가)")
    ap.add_argument("--scenarios", type=str, default=None, help="YAML 파일 경로(여러 실험 사양)")

    # 출력
    ap.add_argument("--out_dir", type=str, default="results/exp")
    ap.add_argument("--out_csv", type=str, default="results/exp/summary.csv")
    ap.add_argument("--note", type=str, default="")
    ap.add_argument(
        "--qc",
        type=str,
        default="warn",
        choices=["off", "warn", "strict"],
        help="off=미실행, warn=경고만 출력, strict=경고 시 종료",
    )

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    out_csv = args.out_csv

    runs: List[Dict[str, Any]] = []

    def handle_qc(_row: Dict[str, Any], ok: bool) -> None:
        """QC 정책에 따라 경고 처리(종료 여부)를 결정합니다."""
        if args.qc == "off":
            return
        if args.qc == "warn":
            # quick_qc 안에서 이미 출력하므로 추가 동작 없음
            return
        if args.qc == "strict" and not ok:
            raise SystemExit("QC failed (strict mode). 실험을 중단합니다.")

    # --------------------------------------------------------
    # 실행 분기: YAML 시나리오 vs grid/단일
    # --------------------------------------------------------
    if args.scenarios:
        scs = load_scenarios(args.scenarios)

        for sc in scs:
            d = vars(args).copy()
            d.update(sc)

            # note/run_id 보조: scenario에서 note가 있으면 우선 적용
            d["note"] = sc.get("note", args.note)

            # repeat 실행: seed를 base_seed부터 +1씩 증가
            base_seed = int(d.get("seed", args.seed))
            rep = int(d.get("repeat", args.repeat))

            for r in range(rep):
                d2 = d.copy()
                d2["seed"] = base_seed + r
                ns = argparse.Namespace(**d2)

                row, ok = run_once(ns, args.out_dir, out_csv)
                runs.append(row)
                handle_qc(row, ok)

    else:
        grid = build_grid(args)

        for conf in grid:
            rep = int(conf.get("repeat", args.repeat))
            base_seed = int(conf.get("seed", args.seed))

            for r in range(rep):
                conf2 = conf.copy()
                conf2["seed"] = base_seed + r
                ns = argparse.Namespace(**conf2)

                row, ok = run_once(ns, args.out_dir, out_csv)
                runs.append(row)
                handle_qc(row, ok)

    # --------------------------------------------------------
    # 콘솔 요약 출력(탭 구분)
    # --------------------------------------------------------
    if runs:
        cols = [
            "policy", "ops", "seed", "waf", "gc_count", "free_blocks",
            "wear_avg", "wear_std", "trimmed_pages", "transition_rate", "reheat_rate",
        ]
        print("\t".join(cols))
        for r in runs:
            print("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))


if __name__ == "__main__":
    main()
