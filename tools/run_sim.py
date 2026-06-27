from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
- main():
  - argparse로 인자 파싱
  - experiment_runner.run_single_experiment() 호출
  - (옵션) summary.csv append
  - (옵션) trace_csv 저장
  - (옵션) gc_events_csv 저장

PowerShell 실행 예시 (중요)
---------------------------
PowerShell에서는 bash처럼 줄 끝에 '\' 쓰면 에러가 난다.
대신 아래 중 하나를 써야 한다.

(1) 한 줄로 쓰기:
    python tools/run_sim.py --gc_policy cota --ops 200000 --out_dir results/smoke --out_csv results/smoke/summary.csv --trace_csv results/smoke/trace.csv

(2) PowerShell 줄바꿈은 백틱(`) 사용:
    python tools/run_sim.py `
      --gc_policy cota `
      --ops 200000 `
      --out_dir results/smoke `
      --out_csv results/smoke/summary.csv

주의 / 흔한 함정
----------------
- out_csv를 results/smoke/summary.csv처럼 주면서 out_dir도 results/smoke로 주면,
  resolve_output_path가 out_dir을 다시 붙여서 results/smoke/results/smoke/summary.csv가 될 수 있다.
  -> out_csv는 보통 summary.csv처럼 파일명만 주는 것을 권장(또는 절대경로 사용).

- warmup_fill은 steady-state 비교에 유용하지만,
  내가 지금 보는 WAF가 워밍업 이후 구간의 WAF인지를 항상 의식해야 한다.
"""

import os
import argparse

from ssd_gc_lab.experiment_runner import (
    ensure_dir,
    quick_qc,
    resolve_output_path,
    run_single_experiment,
    write_gc_events_csv,
    write_trace_csv,
    write_trim_events_csv,
)
from ssd_gc_lab.metrics import append_summary_csv
from ssd_gc_lab.manifest import build_run_manifest, write_manifest


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
    ap.add_argument("--burst_length", type=int, default=0, help="update-heavy burst length in operations")
    ap.add_argument("--burst_ratio", type=float, default=0.0, help="probability of starting a burst at each operation")
    ap.add_argument("--phase_pattern", type=str, default="steady", choices=["steady", "bulk_update_trim", "phased", "rocksdb_like"], help="workload phase pattern")
    ap.add_argument("--trim_locality", type=str, default="mixed", choices=["mixed", "hot", "cold"], help="TRIM target locality")
    ap.add_argument("--trim_burst_length", type=int, default=0, help="length of periodic TRIM-heavy windows")
    ap.add_argument("--trim_burst_interval", type=int, default=0, help="period between TRIM-heavy windows")

    # --------------------------------------------------------
    # GC policy selection and policy parameters
    # --------------------------------------------------------
    ap.add_argument(
        "--gc_policy", type=str, default="greedy",
        choices=["greedy", "cb", "cost_benefit", "bsgc", "cota", "atcb", "re50315"],
        help="GC policy"
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
    ap.add_argument("--trim_events_csv", type=str, default=None, help="per-TRIM event log CSV")
    ap.add_argument("--manifest_json", type=str, default=None, help="옵션: 재현성 manifest JSON 저장 경로")
    ap.add_argument("--note", type=str, default="", help="메모/주석")
    ap.add_argument(
        "--qc", type=str, default="warn", choices=["off", "warn", "strict"],
        help="off=미실행, warn=경고만 출력, strict=경고 시 비정상 종료"
    )

    args = ap.parse_args()

    # ---- 출력 디렉토리 ----
    out_dir = args.out_dir
    ensure_dir(out_dir)

    # out_dir 기준 상대 경로 처리
    out_csv_path = resolve_output_path(args.out_csv, out_dir) if args.out_csv else None
    trace_csv_path = resolve_output_path(args.trace_csv, out_dir) if args.trace_csv else None
    gc_events_csv_path = resolve_output_path(args.gc_events_csv, out_dir) if args.gc_events_csv else None
    trim_events_csv_path = resolve_output_path(args.trim_events_csv, out_dir) if args.trim_events_csv else None
    manifest_json_path = resolve_output_path(args.manifest_json, out_dir) if args.manifest_json else None

    sim, meta, row = run_single_experiment(args, enable_trace=bool(trace_csv_path))

    if out_csv_path:
        # QC는 summary_row를 대상으로 수행(저장될 행을 검사)
        if args.qc != "off":
            ok = quick_qc(row)
            if args.qc == "strict" and not ok:
                raise SystemExit(2)

        append_summary_csv(out_csv_path, sim, meta)
        print(f"[RUN DONE] 결과 CSV append → {out_csv_path}")

    # trace 저장 (시뮬레이터가 trace를 제공할 때만)
    if trace_csv_path and getattr(sim, "trace", None):
        write_trace_csv(trace_csv_path, sim)

    # per-GC 이벤트 로그 저장
    if gc_events_csv_path and getattr(sim.ssd, "gc_event_log", None):
        write_gc_events_csv(gc_events_csv_path, sim)

    if trim_events_csv_path and getattr(sim.ssd, "trim_event_log", None):
        write_trim_events_csv(trim_events_csv_path, sim)

    if manifest_json_path:
        artifacts = {
            "summary_csv": out_csv_path,
            "trace_csv": trace_csv_path,
            "gc_events_csv": gc_events_csv_path,
            "trim_events_csv": trim_events_csv_path,
            "manifest_json": manifest_json_path,
        }
        manifest = build_run_manifest(
            args=args,
            metrics_row=row,
            out_dir=out_dir,
            artifacts=artifacts,
            cwd=os.getcwd(),
        )
        write_manifest(manifest_json_path, manifest)
        print(f"[RUN DONE] manifest JSON 저장 → {manifest_json_path}")


if __name__ == "__main__":
    main()
