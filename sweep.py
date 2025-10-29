from __future__ import annotations
import argparse
import os
from typing import Any, Dict, List

# experiments 모듈의 빌딩 블록을 그대로 재사용
from experiments import (
    build_grid,
    load_scenarios,
    run_once,
)

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

    ap.add_argument("--gc_policy", type=str, default="greedy",
                    choices=["greedy","cb","cost_benefit","bsgc","cat","atcb","re50315"])

    # CAT 확장 파라미터 (gc_algos에 주입 — experiments.run_once 내부에서 처리)
    ap.add_argument("--cat_alpha", type=float, default=None)
    ap.add_argument("--cat_beta", type=float, default=None)
    ap.add_argument("--cat_gamma", type=float, default=None)
    ap.add_argument("--cat_delta", type=float, default=None)
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

    runs: List[Dict[str, Any]] = []

    if args.scenarios:
        scs = load_scenarios(args.scenarios)
        for sc in scs:
            d = vars(args).copy()
            d.update(sc)
            d["note"] = sc.get("note", args.note)
            rep = int(d.get("repeat", args.repeat))
            base_seed = int(d.get("seed", args.seed))
            for r in range(rep):
                d2 = d.copy()
                d2["seed"] = base_seed + r
                row = run_once(argparse.Namespace(**d2), args.out_dir, args.out_csv)
                runs.append(row)
    else:
        grid = build_grid(args)
        for conf in grid:
            rep = int(conf.get("repeat", args.repeat))
            base_seed = int(conf.get("seed", args.seed))
            for r in range(rep):
                conf2 = conf.copy()
                conf2["seed"] = base_seed + r
                row = run_once(argparse.Namespace(**conf2), args.out_dir, args.out_csv)
                runs.append(row)

    # 콘솔 요약
    if runs:
        cols = [
            "policy","ops","seed","waf","gc_count","free_blocks",
            "wear_avg","wear_std","trimmed_pages","transition_rate","reheat_rate"
        ]
        print("\t".join(cols))
        for r in runs:
            print("\t".join(str(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()