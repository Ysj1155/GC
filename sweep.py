"""
sweep.py

GC 실험을 여러 번/여러 조합 돌리고 결과(summary.csv)를 쌓아주는 실행 스크립트.

------------------
experiments.py가 이미 대부분의 기능(정책 주입, 워크로드 생성, run_once, grid/YAML)을 갖고 있는데,
실제로 실험할 때는 보통 아래 2가지 패턴을 많이 쓴다.

1) Grid sweep:
   - policy=greedy,cota; seed=1,2,3; update_ratio=0.5,0.8 같은 식으로
     매개변수 조합(데카르트 곱)을 한 번에 돌리기
2) Scenario(YAML):
   - 아예 실험 사양을 파일로 적어두고(재현성/기록성),
     여러 시나리오를 한 번에 실행하기

sweep.py는 그 용도를 깔끔하게 포장한 thin wrapper다.
- experiments.py의 빌딩 블록(build_grid, load_scenarios, run_once)을 그대로 재사용한다.
- 출력 디렉토리 생성, 반복 실행, 최종 CSV 경로 안내까지 담당한다.

재현성 관점에서의 약속(Contract)
-------------------------------
- 결과는 args.out_csv에 append된다. (파일이 없으면 헤더 생성)
- 실험 조건은 CLI 인자로 고정되며, 시나리오(YAML) 사용 시 각 항목이 args를 덮어쓴다.
- seed 반복(--repeat)은 base_seed + r 방식으로 진행한다.

주의(현재 코드에서 꼭 알고 있어야 하는 디테일)
----------------------------------------------
1) experiments.run_once의 반환 형태
   - 네가 올려준 experiments.py 버전에서는 `return row, ok` (튜플) 형태였다.
   - 그런데 sweep.py는 `row = run_once(...)`로 row만 온다고 가정.
   - 즉, 지금 상태 그대로면 runs에 (row, ok) 튜플이 들어가고,
     아래 콘솔 출력에서 r.get(...)에서 터질 가능성이 매우 높다.

   해결 방법:
   - sweep.py에서 run_once 결과를 받아서 (row, ok)면 row만 사용하도록 처리해준다.
   - (QC를 여기서 쓰지 않을 거면 ok는 버리면 됨)

2) out_csv 경로 해석
   - args.out_csv 기본값이 results/sweep/summary.csv로 이미 폴더를 포함하므로,
     보통은 out_dir와 독립적으로 동작한다.
   - 사용자가 --out_csv summary.csv처럼 파일명만 주면
     사실상 현재 작업 폴더 기준이 될 수 있다.
   - 이 스크립트는 마지막에 출력할 때만 out_dir 아래로 보정해 보기 좋게 만들었는데,
     실제로 append가 일어난 경로와 print 경로가 어긋날 수 있다.
   - 가장 안전한 방식은 out_csv는 실행 직후 절대경로로 정규화해서
     append와 안내를 동일 경로로 맞추는 것.

그래도 이 파일의 역할은 단순하다:
- 실험 조합 돌리고 summary.csv 잘 쌓이면 성공.

사용 예시
---------
1) grid sweep:
   python sweep.py --grid "gc_policy=greedy,cota; seed=41,42; update_ratio=0.8" --repeat 1

2) YAML scenario:
   python sweep.py --scenarios scenarios.yaml

"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

# experiments 모듈의 빌딩 블록을 그대로 재사용
from experiments import (
    build_grid,
    load_scenarios,
    run_once,
)


# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------

def _normalize_out_csv(out_csv: str | None, out_dir: str) -> str | None:
    """
    out_csv 경로를 실제 실행에 쓰이는 형태로 정규화한다.

    규칙:
    - None이면 None 반환
    - 절대경로면 그대로 사용
    - 상대경로인데 디렉토리 포함이면 cwd 기준
    - 상대경로인데 파일명만 있으면 out_dir 밑으로 붙인다
      (사용자 실수를 줄이기 위한 안전 장치)
    """
    if out_csv is None:
        return None
    if os.path.isabs(out_csv):
        return os.path.abspath(out_csv)
    if os.path.dirname(out_csv):
        return os.path.abspath(out_csv)
    return os.path.abspath(os.path.join(out_dir, out_csv))


def _unwrap_run_once(result: Any) -> Dict[str, Any]:
    """
    experiments.run_once 반환값 호환.
    - run_once가 row만 반환하면 그대로 사용
    - (row, ok) 튜플이면 row만 꺼내서 사용
    """
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        row, _ok = result
        return row
    if isinstance(result, dict):
        return result
    raise TypeError(f"run_once() returned unexpected type: {type(result)}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sweep runner (wrapper over experiments.py)")

    # 공통 파라미터(기본값은 run_sim/experiments와 일치)
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

    # COTA 확장 파라미터 (gc_algos에 주입 — experiments.run_once 내부에서 처리)
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
    ap.add_argument("--out_dir", type=str, default="results/sweep")
    ap.add_argument("--out_csv", type=str, default="results/sweep/summary.csv")
    ap.add_argument("--note", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # out_csv 경로를 실행 초기에 정규화해서, append 경로와 안내 경로가 일치하도록 만든다.
    out_csv = _normalize_out_csv(args.out_csv, args.out_dir)

    runs: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # 실행 루프: scenarios 우선, 없으면 grid/단일 실행
    # --------------------------------------------------------

    if args.scenarios:
        scs = load_scenarios(args.scenarios)
        for sc in scs:
            d = vars(args).copy()
            d.update(sc)

            # scenario별 note 우선
            d["note"] = sc.get("note", args.note)

            rep = int(d.get("repeat", args.repeat))
            base_seed = int(d.get("seed", args.seed))

            for r in range(rep):
                d2 = d.copy()
                d2["seed"] = base_seed + r

                res = run_once(argparse.Namespace(**d2), args.out_dir, out_csv)
                row = _unwrap_run_once(res)
                runs.append(row)

    else:
        grid = build_grid(args)
        for conf in grid:
            rep = int(conf.get("repeat", args.repeat))
            base_seed = int(conf.get("seed", args.seed))

            for r in range(rep):
                conf2 = conf.copy()
                conf2["seed"] = base_seed + r

                res = run_once(argparse.Namespace(**conf2), args.out_dir, out_csv)
                row = _unwrap_run_once(res)
                runs.append(row)

    # --------------------------------------------------------
    # 콘솔 요약
    # --------------------------------------------------------

    if runs:
        cols = [
            "policy", "ops", "seed", "waf", "gc_count", "free_blocks",
            "wear_avg", "wear_std", "trimmed_pages", "transition_rate", "reheat_rate",
        ]
        print("\t".join(cols))
        for r in runs:
            print("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))

    # --------------------------------------------------------
    # 완료 알림: 결과 CSV 경로 출력
    # --------------------------------------------------------

    if out_csv:
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
            print(f"[SWEEP DONE] 결과 CSV 생성 완료 → {out_csv}")
        else:
            print(f"[SWEEP DONE] 실행 종료. CSV가 보이지 않습니다. 경로 확인: {out_csv}")


if __name__ == "__main__":
    main()