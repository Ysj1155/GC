from __future__ import annotations

"""
run_sim.py

SSD GC 시뮬레이터 실행기(Entry Point).

이 파일의 역할
--------------
이 프로젝트에는 여러 모듈이 있지만, 결국 사용자가 원하는 건 딱 3가지

1) 이 조건으로 한 번 돌려보고
2) 결과를 CSV로 남기고
3) 필요하면 trace/GC 이벤트 로그도 저장해서 재현 가능하게 만들기

run_sim.py는 그 3가지를 명령줄(CLI)로 한 번에 묶어주는 실행 스크립트다.

재현성(Reproducibility) 관점에서 중요한 이유
-------------------------------------------
- 실험 입력값(ops, update_ratio, hot_ratio, seed, 정책 파라미터...)을
  CLI 인자로 고정하면, 같은 커맨드 = 같은 실험 조건이 된다.
- 결과 요약(summary.csv)에 그 인자(meta)가 함께 저장되므로
  이 결과가 어떤 조건에서 나왔는지가 CSV만 봐도 복원된다.
- trace_csv / gc_events_csv는 디버깅/설명용 근거 로그로서,
  숫자 하나(WAF)만 보고 끝내지 않고 행동 과정을 따라갈 수 있게 한다.

파일 구조 요약
--------------
- helpers:
  - _resolve_path(): out_dir 기준으로 상대 경로를 절대/정규 경로로 바꿔줌
  - _infer_user_total_pages(): cfg에서 LPN 범위를 추정(호환성 방어)
  - _quick_qc(): 결과 지표 기본 무결성 체크(경고/중단)
  - _inject_policy(): 정책 문자열 → sim.gc_policy 함수 주입(+ 파라미터 래핑)

- main():
  - argparse로 인자 파싱
  - SimConfig 생성/보정
  - Simulator 생성 + 정책 주입
  - workload 생성
  - (옵션) warmup_fill로 선행 채우기
  - run()
  - (옵션) summary.csv append
  - (옵션) trace_csv 저장
  - (옵션) gc_events_csv 저장

PowerShell 실행 예시 (중요)
---------------------------
PowerShell에서는 bash처럼 줄 끝에 '\' 쓰면 에러가 난다.
대신 아래 중 하나를 써야 한다.

(1) 한 줄로 쓰기:
    python run_sim.py --gc_policy cota --ops 200000 --out_dir results/smoke --out_csv results/smoke/summary.csv --trace_csv results/smoke/trace.csv

(2) PowerShell 줄바꿈은 백틱(`) 사용:
    python run_sim.py `
      --gc_policy cota `
      --ops 200000 `
      --out_dir results/smoke `
      --out_csv results/smoke/summary.csv

주의 / 흔한 함정
----------------
- out_csv를 results/smoke/summary.csv처럼 주면서 out_dir도 results/smoke로 주면,
  _resolve_path가 out_dir을 다시 붙여서 results/smoke/results/smoke/summary.csv가 될 수 있다.
  -> out_csv는 보통 summary.csv처럼 파일명만 주는 것을 권장(또는 절대경로 사용).

- warmup_fill은 steady-state 비교에 유용하지만,
  내가 지금 보는 WAF가 워밍업 이후 구간의 WAF인지를 항상 의식해야 한다.
"""

import os
import argparse
from datetime import datetime

from config import SimConfig
from simulator import Simulator
from workload import make_workload
from metrics import append_summary_csv, summary_row
import gc_algos


# ============================================================
# Helpers
# ============================================================

def _resolve_path(path: str | None, out_dir: str) -> str | None:
    """
    출력 경로 해석기.

    목적:
    - 사용자가 --out_csv summary.csv 처럼 '상대 경로'만 주면 out_dir 밑으로 저장되게 한다.
    - 사용자가 절대 경로를 주면 그대로 존중한다.

    반환:
    - path가 None이면 None
    - 절대경로면 그대로
    - 상대경로면 os.path.join(out_dir, path)

    중요:
    - out_dir 자체가 results/smoke 같은 값이고,
      path도 results/smoke/summary.csv처럼 이미 out_dir을 포함하면
      out_dir/out_dir/... 꼴이 될 수 있다.
      → 그래서 실무적으로는 path는 summary.csv처럼 주는 걸 추천한다.
    """
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(out_dir, path)


def _infer_user_total_pages(cfg) -> int:
    """
    다양한 SimConfig 형태(과거/실험 버전 차이)와 호환되도록,
    user_total_pages(=워크로드에서 사용할 LPN 개수)를 최대한 안전하게 추정한다.

    이 값이 왜 중요한가?
    --------------------
    - 워크로드 생성(make_workload)에서 LPN 범위를 정할 때 필요하다.
    - 너무 크게 잡으면 논리 주소가 장치보다 커지는 비현실 모델이 되고,
      너무 작게 잡으면 오버프로비저닝이 과도하게 큰 모델이 된다.

    우선순위:
      1) cfg.user_total_pages / total_user_pages / ssd_total_pages
      2) (blocks 후보) * (pages_per_block 후보) * (capacity ratio 후보)
      3) cfg.total_pages * ratio

    없으면 RuntimeError(실험 정의가 불완전하다는 뜻).
    """
    # 1) 직접 필드
    for attr in ("user_total_pages", "total_user_pages", "ssd_total_pages"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v

    # 2) blocks * ppb * ratio
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

    try:
        if blocks and ppb:
            return int(int(blocks) * int(ppb) * float(ratio))
    except Exception:
        pass

    # 3) total_pages * ratio
    total_pages = getattr(cfg, "total_pages", None)
    try:
        if total_pages and isinstance(total_pages, int) and total_pages > 0:
            return int(total_pages * float(ratio))
    except Exception:
        pass

    raise RuntimeError(
        "user_total_pages 추정 실패: SimConfig 구조가 예상과 다릅니다. "
        "필요 필드(user_total_pages / (blocks*ppb*ratio) / total_pages)를 확인하세요."
    )


def _quick_qc(row: dict) -> bool:
    """
    결과 요약 row에 대해 기본적인 상식 선의 무결성 점검.

    QC의 철학
    ---------
    - 이 QC는 정답 판별기가 아니다.
    - 다만 아래 같은 상황을 빠르게 잡아준다:
      - WAF가 1보다 작은데 아무 경고도 없이 넘어가는 경우
      - free_blocks가 0 이하로 떨어지는 비정상 상태
      - 페이지 합(valid+invalid+free)이 total_pages와 안 맞는 로직 버그

    반환:
    - 문제가 없으면 True
    - 경고가 있으면 False (strict 모드에서는 실험 중단 트리거가 됨)
    """
    warn = []
    g = row.get

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

    # 기본 무결성
    if waf is None or waf < 1.0:
        warn.append(f"WAF={waf} (기본적으로 >= 1.0 이어야 정상)")
    if host is not None and dev is not None and dev < host:
        warn.append(f"device_writes({dev}) < host_writes({host})")
    if gc_n is not None and gc_n < 0:
        warn.append(f"gc_count={gc_n} (<0)")
    if gc_t is not None and gc_t < 0:
        warn.append(f"gc_avg_s={gc_t} (<0)")
    if fb is not None and fb <= 0:
        warn.append(f"free_blocks={fb} (<=0)")

    # 페이지 합 일치성(가능할 때만)
    if all(x is not None for x in (vp, ip, fp, tp)):
        if (vp + ip + fp) != tp:
            warn.append(f"valid+invalid+free != total_pages ({vp}+{ip}+{fp} != {tp})")

    # wear 일관성
    if ws is not None and ws < 0:
        warn.append(f"wear_std={ws} (<0)")
    if wmin is not None and wmax is not None and wmax < wmin:
        warn.append(f"wear_max({wmax}) < wear_min({wmin})")

    # 비율 범위 (대부분 0~1로 설계)
    for name, val in [("transition_rate", tr), ("reheat_rate", rr)]:
        if val is not None and not (0.0 <= val <= 1.0):
            warn.append(f"{name}={val} (0~1 범위 밖)")

    # TRIM 기본 검사
    if trim is not None and trim < 0:
        warn.append(f"trimmed_pages={trim} (<0)")
    if trim is not None and tp is not None and trim > tp:
        warn.append(f"trimmed_pages({trim}) > total_pages({tp})")

    if warn:
        print("[QC] WARN:", " | ".join(warn))
        return False

    print("[QC] OK  :", f"policy={g('policy')} seed={g('seed')} waf={waf}")
    return True


def _inject_policy(args, sim: Simulator):
    """
    args.gc_policy 문자열을 보고, sim.gc_policy에 실제 함수를 꽂는다.

    왜 '주입(inject)' 형태인가?
    --------------------------
    - simulator는 "policy(blocks)->victim_idx"라는 인터페이스만 알면 되고,
      어떤 정책을 쓸지는 run_sim.py에서 결정하는 게 깔끔하다.
    - now_step 같은 '시뮬레이터 내부 상태'가 필요한 정책은
      여기서 래핑해서 넣어야 policy 함수 시그니처를 깔끔히 유지할 수 있다.

    또한, COTA 확장 옵션(콜드 바이어스, TRIM 보너스, top-k)은
    gc_algos 내부 전역 설정 함수가 있으면 그것을 통해 반영한다.
    """
    name = (args.gc_policy or "").lower()

    # ---- COTA 확장 설정 주입 (함수가 있을 때만) ----
    if hasattr(gc_algos, "config_cold_bias"):
        gc_algos.config_cold_bias(args.cold_victim_bias)
    if hasattr(gc_algos, "config_trim_age_bonus"):
        gc_algos.config_trim_age_bonus(args.trim_age_bonus)
    if hasattr(gc_algos, "config_victim_prefetch_k"):
        gc_algos.config_victim_prefetch_k(args.victim_prefetch_k)

    # ---- 기본 정책들 ----
    if name == "greedy":
        sim.gc_policy = gc_algos.greedy_policy
        return

    if name in ("cb", "cost_benefit"):
        sim.gc_policy = gc_algos.cb_policy
        return

    if name == "bsgc":
        sim.gc_policy = gc_algos.bsgc_policy
        return

    if name == "cota":
        base = gc_algos.cota_policy
        a, b, g, d = args.cota_alpha, args.cota_beta, args.cota_gamma, args.cota_delta

        # 파라미터가 하나라도 지정되면 래핑(하이퍼파라 튜닝/재현용)
        if any(v is not None for v in (a, b, g, d)):
            def cota_with_w(blocks, _a=a, _b=b, _g=g, _d=d):
                kw = {}
                if _a is not None: kw["alpha"] = _a
                if _b is not None: kw["beta"]  = _b
                if _g is not None: kw["gamma"] = _g
                if _d is not None: kw["delta"] = _d
                return base(blocks, **kw)
            sim.gc_policy = cota_with_w
        else:
            sim.gc_policy = base
        return

    # ---- 확장 정책들 (시뮬레이터 step이 필요) ----
    if name in ("atcb", "atcb_policy"):
        atcb_policy = getattr(gc_algos, "atcb_policy", None)
        if atcb_policy is None:
            raise RuntimeError("gc_algos.atcb_policy 가 없습니다. gc_algos.py 를 업데이트하세요.")

        def atcb_with_now(blocks, _sim=sim):
            return atcb_policy(
                blocks,
                alpha=args.atcb_alpha, beta=args.atcb_beta,
                gamma=args.atcb_gamma, eta=args.atcb_eta,
                now_step=_sim.ssd._step,
            )
        sim.gc_policy = atcb_with_now
        return

    if name in ("re50315", "re50315_policy"):
        re50315_policy = getattr(gc_algos, "re50315_policy", None)
        if re50315_policy is None:
            raise RuntimeError("gc_algos.re50315_policy 가 없습니다. gc_algos.py 를 업데이트하세요.")

        def p_with_now(blocks, _sim=sim):
            return re50315_policy(blocks, K=args.re50315_K, now_step=_sim.ssd._step)

        sim.gc_policy = p_with_now
        return

    raise ValueError(f"지원하지 않는 GC 정책: {args.gc_policy}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="SSD GC simulator runner: reproducible experiment entry point"
    )

    # --------------------------------------------------------
    # 시뮬레이션/장치 파라미터
    # --------------------------------------------------------
    ap.add_argument("--ops", type=int, default=200_000, help="호스트 write(페이지) 횟수")
    ap.add_argument("--update_ratio", type=float, default=0.8, help="업데이트(덮어쓰기) 비율 (0~1)")
    ap.add_argument("--hot_ratio", type=float, default=0.2, help="핫 데이터 비율 (0~1)")
    ap.add_argument("--hot_weight", type=float, default=0.7, help="핫 주소로 보낼 가중치 (0~1)")
    ap.add_argument("--blocks", type=int, default=256, help="블록 수")
    ap.add_argument("--pages_per_block", type=int, default=64, help="블록당 페이지 수")
    ap.add_argument("--gc_free_block_threshold", type=float, default=0.12, help="free blocks 비율 임계치 (0~1)")
    ap.add_argument("--user_capacity_ratio", type=float, default=0.9, help="유저 영역 비율 (0~1)")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    # --------------------------------------------------------
    # TRIM & WARMUP
    # --------------------------------------------------------
    ap.add_argument("--enable_trim", action="store_true", help="워크로드에 TRIM 포함")
    ap.add_argument("--trim_ratio", type=float, default=0.0, help="TRIM 확률(0~1)")
    ap.add_argument(
        "--warmup_fill", type=float, default=0.0,
        help="실행 전 선행 채우기 비율(0.0~0.99). steady-state 비교용"
    )

    # --------------------------------------------------------
    # 정책 선택 & 파라미터
    # --------------------------------------------------------
    ap.add_argument(
        "--gc_policy", type=str, default="greedy",
        choices=["greedy", "cb", "cost_benefit", "bsgc", "cota", "atcb", "re50315"],
        help="GC 정책 선택"
    )

    # COTA 파라미터
    ap.add_argument("--cota_alpha", type=float, default=None, help="COTA α (invalid)")
    ap.add_argument("--cota_beta",  type=float, default=None, help="COTA β (1-hot)")
    ap.add_argument("--cota_gamma", type=float, default=None, help="COTA γ (age)")
    ap.add_argument("--cota_delta", type=float, default=None, help="COTA δ (1-wear)")
    ap.add_argument("--cold_victim_bias", type=float, default=1.0, help="cold 풀 가점(>1.0)")
    ap.add_argument("--trim_age_bonus", type=float, default=0.0, help="TRIM 비율 기반 age 보너스")
    ap.add_argument("--victim_prefetch_k", type=int, default=1, help="victim 후보 top-K")

    # ATCB / RE50315 파라미터
    ap.add_argument("--atcb_alpha", type=float, default=0.5)
    ap.add_argument("--atcb_beta",  type=float, default=0.3)
    ap.add_argument("--atcb_gamma", type=float, default=0.1)
    ap.add_argument("--atcb_eta",   type=float, default=0.1)
    ap.add_argument("--re50315_K",  type=float, default=1.0)

    # --------------------------------------------------------
    # 실행/출력
    # --------------------------------------------------------
    ap.add_argument(
        "--bg_gc_every", type=int, default=0,
        help="K>0이면 매 K ops마다 백그라운드 GC 시도(시뮬레이터가 지원할 때)"
    )
    ap.add_argument(
        "--out_dir", type=str, default="results/run",
        help="결과/로그를 저장할 디렉토리"
    )
    ap.add_argument("--out_csv", type=str, default=None, help="요약 CSV append 경로 (권장: summary.csv)")
    ap.add_argument("--trace_csv", type=str, default=None, help="옵션: trace CSV (시뮬레이터가 지원 시)")
    ap.add_argument("--gc_events_csv", type=str, default=None, help="per-GC 이벤트 로그 CSV (실행 후 저장)")
    ap.add_argument("--note", type=str, default="", help="메모/주석")
    ap.add_argument(
        "--qc", type=str, default="warn", choices=["off", "warn", "strict"],
        help="off=미실행, warn=경고만 출력, strict=경고 시 비정상 종료"
    )

    args = ap.parse_args()

    # ---- 출력 디렉토리 ----
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # out_dir 기준 상대 경로 처리
    out_csv_path       = _resolve_path(args.out_csv, out_dir) if args.out_csv else None
    trace_csv_path     = _resolve_path(args.trace_csv, out_dir) if args.trace_csv else None
    gc_events_csv_path = _resolve_path(args.gc_events_csv, out_dir) if args.gc_events_csv else None

    # --------------------------------------------------------
    # Config + Simulator 생성
    # --------------------------------------------------------
    cfg = SimConfig(
        num_blocks=args.blocks,
        pages_per_block=args.pages_per_block,
        gc_free_block_threshold=args.gc_free_block_threshold,
        rng_seed=args.seed,
        user_capacity_ratio=args.user_capacity_ratio,
    )

    # SimConfig 내부에 user_total_pages가 없는 버전도 있을 수 있어 보정
    user_total_pages = _infer_user_total_pages(cfg)
    try:
        setattr(cfg, "user_total_pages", user_total_pages)
    except Exception:
        pass

    sim = Simulator(cfg, enable_trace=bool(trace_csv_path), bg_gc_every=args.bg_gc_every)
    _inject_policy(args, sim)

    # --------------------------------------------------------
    # Workload 생성
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Warmup: steady-state 비교용 선행 채우기
    # --------------------------------------------------------
    if args.warmup_fill > 0.0:
        # 프로젝트 안전장치: free 블록 최소 2개는 남겨 둔다.
        reserve_free_blocks = 2
        ppb = getattr(cfg, "pages_per_block", getattr(cfg, "ppb", 64))

        # warmup이 너무 과하면 "막혀서 계속 GC만 도는" 상태가 될 수 있으니 상한을 둔다.
        max_warm_pages = max(0, user_total_pages - reserve_free_blocks * ppb)
        target_pages = min(
            int(user_total_pages * min(max(args.warmup_fill, 0.0), 0.99)),
            max_warm_pages,
        )

        wrote = 0
        lpn = 0
        while wrote < target_pages and lpn < user_total_pages:
            # free_pages가 바닥나기 전에 GC로 숨통을 틔운다.
            if getattr(sim.ssd, "free_pages", 1) <= ppb:
                sim.ssd.collect_garbage(sim.gc_policy, cause="warmup")
            sim.ssd.write_lpn(lpn)
            wrote += 1
            lpn += 1

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    sim.run(wl)

    # --------------------------------------------------------
    # 결과 저장 (가능한 것만)
    # --------------------------------------------------------
    if out_csv_path:
        meta = {
            # 재현성 핵심: 이 row가 어떤 커맨드 조건이었는지
            "run_id": args.note or f"{args.gc_policy}_{args.seed}",
            "policy": args.gc_policy,
            "ops": args.ops,
            "update_ratio": args.update_ratio,
            "hot_ratio": args.hot_ratio,
            "hot_weight": args.hot_weight,
            "seed": args.seed,
            "trim_enabled": 1 if args.enable_trim else 0,
            "trim_ratio": args.trim_ratio,
            "warmup_fill": args.warmup_fill,
            "bg_gc_every": args.bg_gc_every,
            "note": args.note,
            "ts": datetime.now().isoformat(timespec="seconds"),

            # COTA 계열 메타
            "cota_alpha": getattr(args, "cota_alpha", None),
            "cota_beta": getattr(args, "cota_beta", None),
            "cota_gamma": getattr(args, "cota_gamma", None),
            "cota_delta": getattr(args, "cota_delta", None),
            "cold_victim_bias": getattr(args, "cold_victim_bias", 1.0),
            "trim_age_bonus": getattr(args, "trim_age_bonus", 0.0),
            "victim_prefetch_k": getattr(args, "victim_prefetch_k", 1),
        }

        # QC는 summary_row를 대상으로 수행(저장될 행을 검사)
        row = summary_row(sim, meta)
        if args.qc != "off":
            ok = _quick_qc(row)
            if args.qc == "strict" and not ok:
                raise SystemExit(2)

        append_summary_csv(out_csv_path, sim, meta)
        print(f"[RUN DONE] 결과 CSV append → {out_csv_path}")

    # trace 저장 (시뮬레이터가 trace를 제공할 때만)
    if trace_csv_path and getattr(sim, "trace", None):
        import csv
        with open(trace_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "free_pages", "device_writes", "gc_count", "gc_event"])
            for i in range(len(sim.trace["step"])):
                w.writerow([
                    sim.trace["step"][i],
                    sim.trace["free_pages"][i],
                    sim.trace["device_writes"][i],
                    sim.trace["gc_count"][i],
                    sim.trace["gc_event"][i],
                ])

    # per-GC 이벤트 로그 저장
    if gc_events_csv_path and getattr(sim.ssd, "gc_event_log", None):
        import csv
        with open(gc_events_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted(sim.ssd.gc_event_log[0].keys()))
            w.writeheader()
            for ev in sim.ssd.gc_event_log:
                w.writerow(ev)


if __name__ == "__main__":
    main()