from __future__ import annotations
import os
import sys
import csv
import json
import argparse
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from config import SimConfig
from simulator import Simulator
from workload import make_workload
from metrics import append_summary_csv, summary_row
import gc_algos

# ------------------------------------------------------------
# 유틸
# ------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _infer_user_total_pages(cfg) -> int:
    # run_sim.py와 동일 로직(간단판)
    for attr in ("user_total_pages", "total_user_pages", "ssd_total_pages"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    blocks = (
        getattr(cfg, "num_blocks", None)
        or getattr(cfg, "blocks", None)
        or getattr(cfg, "total_blocks", None)
    )
    ppb = getattr(cfg, "pages_per_block", None) or getattr(cfg, "ppb", None)
    ratio = (
        getattr(cfg, "user_capacity_ratio", None)
        or getattr(cfg, "capacity_ratio", None)
        or 1.0
    )
    if blocks and ppb:
        return int(int(blocks) * int(ppb) * float(ratio))
    total_pages = getattr(cfg, "total_pages", None)
    if total_pages and isinstance(total_pages, int) and total_pages > 0:
        return int(total_pages * float(ratio))
    raise RuntimeError("user_total_pages 추정 실패: SimConfig를 확인하세요.")


def _inject_policy(args, sim: Simulator):
    name = (args.gc_policy or "").lower()

    # 공통 전역 설정(있을 때만)
    if hasattr(gc_algos, "config_cold_bias"):
        gc_algos.config_cold_bias(getattr(args, "cold_victim_bias", 1.0))
    if hasattr(gc_algos, "config_trim_age_bonus"):
        gc_algos.config_trim_age_bonus(getattr(args, "trim_age_bonus", 0.0))
    if hasattr(gc_algos, "config_victim_prefetch_k"):
        gc_algos.config_victim_prefetch_k(getattr(args, "victim_prefetch_k", 1))

    if name in ("greedy",):
        sim.gc_policy = getattr(gc_algos, "greedy_policy"); return
    if name in ("cb","cost_benefit"):
        sim.gc_policy = getattr(gc_algos, "cb_policy"); return
    if name in ("bsgc",):
        sim.gc_policy = getattr(gc_algos, "bsgc_policy"); return
    if name in ("cota",):
        def cota_with_weights(blocks,
                              a=args.cota_alpha, b=args.cota_beta,
                              g=args.cota_gamma, d=args.cota_delta):
            kw = {}
            if a is not None: kw["alpha"]=a
            if b is not None: kw["beta"]=b
            if g is not None: kw["gamma"]=g
            if d is not None: kw["delta"]=d
            return gc_algos.cota_policy(blocks, **kw) if kw else gc_algos.cota_policy(blocks)
        sim.gc_policy = cota_with_weights
        return
    if name in ("atcb","atcb_policy"):
        atcb = getattr(gc_algos, "atcb_policy")
        def atcb_with_now(blocks, _sim=sim):
            return atcb(blocks,
                        alpha=getattr(args,"atcb_alpha",0.5),
                        beta=getattr(args,"atcb_beta",0.3),
                        gamma=getattr(args,"atcb_gamma",0.1),
                        eta=getattr(args,"atcb_eta",0.1),
                        now_step=_sim.ssd._step)
        sim.gc_policy = atcb_with_now; return

    if name in ("re50315","re50315_policy"):
        re = getattr(gc_algos, "re50315_policy")
        def re_with_now(blocks, _sim=sim):
            return re(blocks, K=getattr(args,"re50315_K",1.0), now_step=_sim.ssd._step)
        sim.gc_policy = re_with_now; return

    raise ValueError(f"지원하지 않는 GC 정책: {args.gc_policy}")

def _quick_qc(row: dict) -> bool:
    warn = []
    def g(k, default=None): return row.get(k, default)

    waf = g("waf")
    host = g("host_writes")
    dev  = g("device_writes")
    gc_n = g("gc_count")
    gc_t = g("gc_avg_s")
    fb   = g("free_blocks")
    fp   = g("free_pages")
    vp   = g("valid_pages")
    ip   = g("invalid_pages")
    tp   = g("total_pages")
    ws   = g("wear_std")
    wmin = g("wear_min")
    wmax = g("wear_max")
    tr   = g("transition_rate")
    rr   = g("reheat_rate")
    trim = g("trimmed_pages")

    # 기본 물리/지표 무결성
    if waf is None or waf < 1.0: warn.append(f"WAF={waf} (기본적으로 >= 1.0 이어야 정상)")
    if host is not None and dev is not None and dev < host: warn.append(f"device_writes({dev}) < host_writes({host})")
    if gc_n is not None and gc_n < 0: warn.append(f"gc_count={gc_n} (<0)")
    if gc_t is not None and gc_t < 0: warn.append(f"gc_avg_s={gc_t} (<0)")
    if fb is not None and fb <= 0: warn.append(f"free_blocks={fb} (<=0)")

    # 페이지 합 일치성(가능할 때만 검사)
    if all(x is not None for x in (vp, ip, fp, tp)):
        if (vp + ip + fp) != tp:
            warn.append(f"valid+invalid+free != total_pages ({vp}+{ip}+{fp} != {tp})")

    # wear 일관성
    if ws is not None and ws < 0: warn.append(f"wear_std={ws} (<0)")
    if wmin is not None and wmax is not None and wmax < wmin:
        warn.append(f"wear_max({wmax}) < wear_min({wmin})")

    # 비율 범위
    for name, val in [("transition_rate", tr), ("reheat_rate", rr)]:
        if val is not None and not (0.0 <= val <= 1.0):
            warn.append(f"{name}={val} (0~1 범위 밖)")

    # TRIM
    if trim is not None and trim < 0:
        warn.append(f"trimmed_pages={trim} (<0)")
    if trim is not None and tp is not None and trim > tp:
        warn.append(f"trimmed_pages({trim}) > total_pages({tp})")

    if warn:
        print("[QC] WARN:", " | ".join(warn))
        return False
    else:
        print("[QC] OK  :", f"policy={g('policy')} seed={g('seed')} waf={waf}")
        return True

# ------------------------------------------------------------
# 단일 실행
# ------------------------------------------------------------

def run_once(args, out_dir: str, out_csv: Optional[str]) -> Dict[str, Any]:
    cfg = SimConfig(
        num_blocks=args.blocks,
        pages_per_block=args.pages_per_block,
        gc_free_block_threshold=args.gc_free_block_threshold,
        rng_seed=args.seed,
        user_capacity_ratio=args.user_capacity_ratio,
    )
    # user_total_pages 명시
    user_total_pages = _infer_user_total_pages(cfg)
    try:
        setattr(cfg, "user_total_pages", user_total_pages)
    except Exception:
        pass

    sim = Simulator(cfg, enable_trace=False, bg_gc_every=args.bg_gc_every)
    _inject_policy(args, sim)

    wl = make_workload(
        n_ops=args.ops,
        update_ratio=args.update_ratio,
        ssd_total_pages=user_total_pages,
        rng_seed=args.seed,
        hot_ratio=args.hot_ratio,
        hot_weight=args.hot_weight,
        enable_trim=args.enable_trim,
        trim_ratio=args.trim_ratio,
    )

    # 워밍업
    if args.warmup_fill > 0.0:
        reserve_free_blocks = 2
        ppb = getattr(cfg, "pages_per_block", getattr(cfg, "ppb", 64))
        max_warm_pages = max(0, user_total_pages - reserve_free_blocks * ppb)
        target_pages = min(int(user_total_pages * min(max(args.warmup_fill, 0.0), 0.99)), max_warm_pages)
        wrote, lpn = 0, 0
        while wrote < target_pages and lpn < user_total_pages:
            if getattr(sim.ssd, "free_pages", 1) <= ppb:
                sim.ssd.collect_garbage(sim.gc_policy, cause="warmup")
            sim.ssd.write_lpn(lpn)
            wrote += 1
            lpn += 1

    sim.run(wl)

    meta = {
        "run_id": getattr(args, "note", None) or f"{args.gc_policy}_{args.seed}",
        "policy": args.gc_policy,
        "ops": args.ops,
        "seed": args.seed,
        "update_ratio": args.update_ratio,
        "hot_ratio": args.hot_ratio,
        "hot_weight": args.hot_weight,
        "trim_enabled": 1 if getattr(args, "enable_trim", False) else 0,
        "trim_ratio": getattr(args, "trim_ratio", 0.0),
        "warmup_fill": args.warmup_fill,
        "bg_gc_every": args.bg_gc_every,
        "ts": datetime.now().isoformat(timespec="seconds"),

    # COTA 표준 컬럼
    "cota_alpha": getattr(args, "cota_alpha", None),
    "cota_beta":  getattr(args, "cota_beta",  None),
    "cota_gamma": getattr(args, "cota_gamma", None),
    "cota_delta": getattr(args, "cota_delta", None),
    "cold_victim_bias": getattr(args, "cold_victim_bias", 1.0),
    "trim_age_bonus":   getattr(args, "trim_age_bonus", 0.0),
    "victim_prefetch_k":getattr(args, "victim_prefetch_k", 1),
    }

    if out_csv:
        append_summary_csv(out_csv, sim, meta)

    row = summary_row(sim, meta)
    _quick_qc(row)  # ← 여기 추가 (경고/OK를 콘솔에 출력)
    return row

# ------------------------------------------------------------
# GRID 빌더
# ------------------------------------------------------------

def _parse_csv_list(s: str) -> List[str]:
    return [x for x in (s or "").split(",") if x != ""]


def build_grid(args) -> List[Dict[str, Any]]:
    """간단한 그리드 생성기.
    예: --grid "gc_policy=cota,greedy; update_ratio=0.5,0.9; seed=1,2"
    → 매개변수 데카르트 곱 생성.
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

    # 데카르트 곱
    grids: List[Dict[str, Any]] = []
    def _dfs(i: int, acc: Dict[str, Any]):
        if i == len(items):
            d = vars(args).copy()
            d.update(acc)
            grids.append(d)
            return
        for k, v in items[i]:
            acc[k] = v
            _dfs(i+1, acc)
            acc.pop(k, None)
    _dfs(0, {})
    return grids


def _coerce_value(x: str) -> Any:
    # 숫자/불리언/None 자동 캐스팅, 실패 시 원문 문자열
    if x.lower() in ("none","null"): return None
    if x.lower() in ("true","false"): return x.lower() == "true"
    try:
        if "." in x: return float(x)
        return int(x)
    except Exception:
        return x


# ------------------------------------------------------------
# YAML 시나리오 로더(옵션)
# ------------------------------------------------------------

def load_scenarios(path: str) -> List[Dict[str, Any]]:
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
# CLI
# ------------------------------------------------------------

def main():
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

    ap.add_argument("--gc_policy", type=str, default="greedy",
                    choices=["greedy","cb","cost_benefit","bsgc","cota","atcb","re50315"])

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
    ap.add_argument("--qc", type=str, default="warn", choices=["off", "warn", "strict"],
                    help="off=미실행, warn=경고만 출력, strict=경고 시 종료")

    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    out_csv = args.out_csv

    runs: List[Dict[str, Any]] = []

    if args.scenarios:
        # YAML 파일의 각 항목을 args에 병합하여 실행
        scs = load_scenarios(args.scenarios)
        for i, sc in enumerate(scs):
            d = vars(args).copy()
            d.update(sc)
            # note/run_id 보조
            d["note"] = sc.get("note", args.note)
            # 반복
            base_seed = int(d.get("seed", args.seed))
            rep = int(d.get("repeat", args.repeat))
            for r in range(rep):
                d2 = d.copy()
                d2["seed"] = base_seed + r
                ns = argparse.Namespace(**d2)
                row = run_once(ns, args.out_dir, out_csv)
                runs.append(row)
    else:
        # grid 또는 단일 실행
        grid = build_grid(args)
        for conf in grid:
            rep = int(conf.get("repeat", args.repeat))
            base_seed = int(conf.get("seed", args.seed))
            for r in range(rep):
                conf2 = conf.copy()
                conf2["seed"] = base_seed + r
                ns = argparse.Namespace(**conf2)
                row = run_once(ns, args.out_dir, out_csv)
                runs.append(row)

    # 콘솔 요약 출력
    if runs:
        cols = ["policy", "ops", "seed", "waf", "gc_count", "free_blocks",
                "wear_avg", "wear_std", "trimmed_pages", "transition_rate", "reheat_rate"]
        print("\t".join(cols))
        for r in runs:
            # None -> "" 처리
            print("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))

if __name__ == "__main__":
    main()