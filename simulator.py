"""
simulator.py

Simulator (Entry/Orchestrator)

이 모듈은 SSD 모델(Device/SSD)과 워크로드(workload) 사이에 서서,
실험을 일관된 규칙으로 실행하고, 필요하면 trace/QC/백그라운드 GC까지
한 곳에서 묶어주는 실행 오케스트레이터다.

이 프로젝트에서의 위치
----------------------
- models.py: SSD/Block/PageState 같은 상태와 동작의 물리 모델
- gc_algos.py: victim 선택 로직(정책)
- workload.py: 어떤 순서로 write/trim을 날릴지(입력 시퀀스)
- metrics.py: 실행이 끝난 후 결과를 요약하는 집계기
- simulator.py: 위 것들을 연결해서 한 번의 실험(run)을 굴리는 엔진

설계 목표 (정보 전달 / 재현성 관점)
----------------------------------
1) **실험 재현성**:
   - 같은 workload + 같은 policy + 같은 seed + 같은 cfg이면,
     Simulator는 동일한 동작 경로를 밟도록 최대한 단순한 규칙을 유지한다.

2) **하위호환(Backward Compatibility)**:
   - run_sim.py / experiments.py가 기대하는 인터페이스를 안전하게 흡수한다.
     예: Simulator(cfg, enable_trace=..., bg_gc_every=...)
         sim.ssd, sim.gc_policy, sim.trace['gc_event']

3) **Fail-safe(안전장치)**:
   - 목적지 블록이 없어서 마이그레이션이 막히는 문제 등은
     SSD 모델 쪽에서 방어하고,
     Simulator는 GC 트리거/호출 시점을 안정적으로 관리한다.

핵심 계약(Contract)
-------------------
- 외부에서 정책을 주입할 수 있다:
    sim.gc_policy = callable(blocks)->victim_idx
  (주입이 없으면 policy_name으로 gc_algos.get_gc_policy를 사용)

- run(workload):
  workload는 두 형태를 지원한다.
  1) [lpn, lpn, ...]                 -> write 연산으로 간주
  2) [("write"| "trim", lpn), ...]   -> 명시 연산(워크로드 생성기가 TRIM을 넣는 경우)

- enable_trace=True 이면, 실행 과정이 sim.trace에 기록된다.
  run_sim.py가 CSV로 저장하기 쉬운 “평평한 리스트” 구조를 유지한다.

BG-GC(백그라운드 GC)
-------------------
- bg_gc_every 같은 주기 기반(cadence) 트리거를 지원한다.
- 또한 cfg.gc_free_block_threshold(비율)가 있으면
  free ratio가 임계치보다 낮아졌을 때 GC를 한 번 시도한다.

주의(유지보수 포인트)
---------------------
- Simulator의 ops 카운터(self.ops)와 SSD 내부 step(예: ssd._step)이
  완전히 동일할 필요는 없지만, 혼동되지 않게 역할을 분리해야 한다.
  - self.ops: 워크로드에서 몇 번째 op인지(실험 오케스트레이션 기준)
  - ssd._step: SSD 모델이 내부 상태(온도/recency 등) 갱신에 쓰는 step

- _free_block_count는 완전 빈 블록(used==0)을 free block으로 정의한다.
  SSD 모델의 정의와 다르면, 이 함수부터 프로젝트 정의에 맞게 조정해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, List, Any, Dict
import time


# ============================================================
# 안전한 의존성 로딩
# ============================================================

try:
    # 프로젝트 모델: SSD 이름으로 정의되어 있음
    from models import SSD as Device, Block, PageState
except Exception:
    # 타입 안전을 위해 최소 더미 제공(의존성 깨진 환경에서도 import는 되게)
    Device = object  # type: ignore
    Block = object   # type: ignore

    class PageState:  # type: ignore
        FREE = 0
        VALID = 1
        INVALID = 2


try:
    # 정책 팩토리: get_gc_policy(name) -> policy(blocks)->victim_idx
    from gc_algos import get_gc_policy
except Exception:
    # 최후방어: invalid_count가 최대인 블록 선택 (greedy 비슷한 형태)
    def get_gc_policy(name: str) -> Callable[[List[Any]], Optional[int]]:
        def greedy(blocks: List[Any]) -> Optional[int]:
            if not blocks:
                return None
            best, best_inv = None, -1
            for i, b in enumerate(blocks):
                inv = int(getattr(b, "invalid_count", 0))
                if inv > best_inv:
                    best, best_inv = i, inv
            return best
        return greedy


# ============================================================
# BG-GC cadence(주기 스케줄)
# ============================================================

@dataclass
class BGSchedule:
    """
    pool별 GC 호출 주기(ops 기준). 0이면 해당 주기는 off.

    - every_gen: 일반(general) 주기
    - every_hot/every_cold: 필요하면 확장 가능한 훅
      (현재 SSD 모델이 pool/stream 개념을 실제로 굴린다면 여기와 연결 가능)
    """
    every_hot: int = 0
    every_cold: int = 0
    every_gen: int = 0


# ============================================================
# 내부 유틸
# ============================================================

def _free_block_count(blocks: List[Any]) -> int:
    """
    free block을 used==0(VALID+INVALID가 0)인 블록으로 정의한다.

    이 정의는 프로젝트 모델(models.Block)의 기본 상태(전부 FREE)와 잘 맞는다.
    만약 SSD 모델에서 free block을 완전히 빈 블록이 아니라
    free 페이지가 있는 블록으로 보고 싶다면, 이 정의는 바뀌어야 한다.
    """
    cnt = 0
    for b in blocks:
        used = int(getattr(b, "valid_count", 0)) + int(getattr(b, "invalid_count", 0))
        if used == 0:
            cnt += 1
    return cnt


def _pages_per_block(ssd: Any) -> int:
    """
    SSD에서 pages_per_block을 최대한 호환적으로 추정한다.
    - ssd.pages_per_block가 있으면 그걸 사용
    - 없으면 blocks[0].pages 길이로 추정
    """
    if hasattr(ssd, "pages_per_block"):
        return int(getattr(ssd, "pages_per_block"))
    blocks = getattr(ssd, "blocks", []) or []
    if blocks and hasattr(blocks[0], "pages"):
        return len(getattr(blocks[0], "pages"))
    return 0


# ============================================================
# Simulator
# ============================================================

class Simulator:
    """
    Simulator는 SSD 모델의 주인공이 아니다.
    SSD는 실제 상태 전이(페이지 할당/무효화/GC 마이그레이션/erase)를 수행하고,
    Simulator는 언제 write/trim을 호출하고, 언제 GC를 시도할지,
    그리고 실행 과정을 어떤 형태로 기록할지를 책임진다.
    """

    def __init__(
        self,
        cfg_or_device: Any,
        policy_name: str = "greedy",
        cold_pool: bool = True,
        bg: Optional[BGSchedule] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        cfg_or_device:
            - SimConfig 유사 객체 (num_blocks, pages_per_block 등) 또는
            - 이미 생성된 SSD(Device) 인스턴스

        policy_name:
            외부에서 sim.gc_policy를 주입하지 않았을 때 사용할 기본 정책 이름

        bg:
            BG-GC 스케줄(주기). None이면 bg_gc_every로부터 기본값을 구성

        kwargs (하위호환 흡수)
        ----------------------
        enable_trace: bool
            True면 per-op trace를 기록한다.

        bg_gc_every: int
            gen pool에 대한 주기 기반 GC 시도.
            (프로젝트가 확장되면 hot/cold cadence도 분리 가능)

        그 외 키워드는 조용히 무시한다(호환성/드랍인 목적).
        """

        # ----------------------------------------------------
        # Trace 옵션
        # ----------------------------------------------------
        self.enable_trace: bool = bool(kwargs.pop("enable_trace", False))

        # run_sim.py가 기대하는 스키마를 그대로 유지
        self.trace: Dict[str, List[Any]] = {}
        if self.enable_trace:
            self.trace = {
                "step": [],
                "free_pages": [],
                "device_writes": [],
                "gc_count": [],
                "gc_event": [],   # per-op: "", bg_gen, bg_hot, bg_cold, low_free
                "gc_events": [],  # per-GC: dict snapshot (선택)
            }

        # ----------------------------------------------------
        # BG-GC 스케줄 구성
        # ----------------------------------------------------
        bg_every = int(kwargs.pop("bg_gc_every", 0) or 0)
        self.bg = bg or BGSchedule(
            every_hot=0,
            every_cold=max(bg_every * 4, 0) if bg_every else 0,
            every_gen=bg_every,
        )

        # ----------------------------------------------------
        # 정책: 기본 팩토리 + 외부 주입 우선 슬롯
        # ----------------------------------------------------
        self._policy_name = (policy_name or "greedy").lower()
        self._policy = get_gc_policy(self._policy_name)

        # 외부에서 sim.gc_policy에 함수를 주입하면 그걸 최우선 사용
        self.gc_policy: Optional[Callable[[List[Any]], Optional[int]]] = None

        # ----------------------------------------------------
        # cfg/device 수용
        # ----------------------------------------------------
        self.cfg = None
        self.ssd: Any = None

        # cfg처럼 보이면 (num_blocks/pages_per_block)로 판단
        if hasattr(cfg_or_device, "num_blocks") and hasattr(cfg_or_device, "pages_per_block"):
            self.cfg = cfg_or_device

            # cfg.prepare()가 있으면 먼저 호출 (latency preset/validation 등)
            try:
                if hasattr(self.cfg, "prepare"):
                    self.cfg.prepare()
            except Exception:
                # prepare 실패를 여기서 강제 종료할지 여부는 프로젝트 취향.
                pass

            nb = int(getattr(self.cfg, "num_blocks"))
            ppb = int(getattr(self.cfg, "pages_per_block"))
            seed = int(getattr(self.cfg, "rng_seed", 42))

            # models.SSD는 (num_blocks, pages_per_block, rng_seed=...) 시그니처를 기대
            self.ssd = Device(nb, ppb, rng_seed=seed)
        else:
            # 이미 SSD 인스턴스가 들어왔다고 가정
            self.ssd = cfg_or_device

        # 구버전 코드 호환 별칭
        self.dev = self.ssd

        # free ratio 임계치(있으면 cfg를 따름)
        self.gc_free_block_threshold = float(
            getattr(self.cfg, "gc_free_block_threshold", 0.0)
        ) if self.cfg is not None else 0.0

        # Simulator 기준 ops 카운터 (워크로드의 몇 번째 op인지)
        self.ops: int = 0

        # 남은 kwargs는 의도적으로 무시(하위호환 흡수)
        _ = kwargs

    # ========================================================
    # 내부: 정책 선택 / GC 수행
    # ========================================================

    def _choose_policy(self) -> Callable[[List[Any]], Optional[int]]:
        """
        외부 주입(sim.gc_policy)이 있으면 그것을 우선 사용한다.
        (run_sim.py가 정책을 래핑해서 넣는 구조와 잘 맞는다.)
        """
        return self.gc_policy or self._policy

    def _do_gc(self, cause: str = "bg") -> bool:
        """
        GC 한 번 실행.

        실제 마이그레이션/erase 로직은 SSD.collect_garbage에 있다.
        Simulator는 "언제 호출하는지"와 trace를 어떻게 남기는지만 다룬다.

        Returns
        -------
        True면 GC 호출을 시도했고, SSD가 처리했다고 간주.
        False면 SSD에 blocks가 없거나 collect_garbage가 없는 경우.
        """
        blocks = getattr(self.ssd, "blocks", []) or []
        if not blocks:
            return False

        policy = self._choose_policy()

        if hasattr(self.ssd, "collect_garbage"):
            try:
                # 최신 SSD 모델: collect_garbage(policy=..., cause=...)
                self.ssd.collect_garbage(policy=policy, cause=cause)
            except TypeError:
                # 구버전 호환: collect_garbage(policy) 형태
                self.ssd.collect_garbage(policy)

            # per-GC 스냅샷(선택)
            if self.enable_trace and "gc_events" in self.trace:
                self.trace["gc_events"].append({
                    "step": self.ops,
                    "cause": cause,
                    "free_blocks": _free_block_count(blocks),
                    "gc_count": int(getattr(self.ssd, "gc_count", 0)),
                })
            return True

        return False

    def _should_gc_by_free_ratio(self) -> bool:
        """
        cfg.gc_free_block_threshold(비율)가 설정된 경우,
        free block ratio가 임계치보다 낮으면 GC를 시도한다.
        """
        thr = float(self.gc_free_block_threshold or 0.0)
        if thr <= 0.0:
            return False

        blocks = getattr(self.ssd, "blocks", []) or []
        if not blocks:
            return False

        free = _free_block_count(blocks)
        ratio = free / max(1, len(blocks))
        return ratio < thr

    # ========================================================
    # Public: 워크로드 실행
    # ========================================================

    def run(self, workload: List[Any]) -> None:
        """
        workload 형식
        -------------
        1) [lpn, lpn, ...]
           - 각 항목은 write_lpn(lpn)으로 처리한다.

        2) [("write"|"trim", lpn), ...]
           - ("trim", lpn)은 trim_lpn(lpn)을 호출한다.
           - ("write", lpn)은 write_lpn(lpn)과 동일 취급.

        실행 중 수행하는 일
        -----------------
        - 각 op 수행(write/trim)
        - ops 카운터 증가
        - BG cadence 스케줄에 따라 GC 시도
        - free ratio 임계치 조건이면 GC 시도
        - enable_trace면 per-op trace 기록
        """
        # BG cadence 카운터
        t_hot = 0
        t_cold = 0
        t_gen = 0

        for op in workload:
            # -------------------------
            # 1) op decode
            # -------------------------
            if isinstance(op, (list, tuple)) and len(op) == 2:
                kind, lpn = op[0], int(op[1])
                if kind == "trim":
                    if hasattr(self.ssd, "trim_lpn"):
                        self.ssd.trim_lpn(lpn)
                else:
                    if hasattr(self.ssd, "write_lpn"):
                        self.ssd.write_lpn(lpn)
            else:
                lpn = int(op)
                if hasattr(self.ssd, "write_lpn"):
                    self.ssd.write_lpn(lpn)

            # -------------------------
            # 2) step update
            # -------------------------
            self.ops += 1
            t_hot += 1
            t_cold += 1
            t_gen += 1

            # -------------------------
            # 3) GC triggers
            # -------------------------
            did_gc = False
            gc_cause = ""

            # (a) cadence 기반 BG-GC
            if self.bg.every_gen and t_gen >= self.bg.every_gen:
                did_gc = self._do_gc(cause="bg_gen")
                if did_gc:
                    gc_cause = "bg_gen"
                t_gen = 0

            if (not did_gc) and self.bg.every_hot and t_hot >= self.bg.every_hot:
                did_gc = self._do_gc(cause="bg_hot")
                if did_gc:
                    gc_cause = "bg_hot"
                t_hot = 0

            if (not did_gc) and self.bg.every_cold and t_cold >= self.bg.every_cold:
                did_gc = self._do_gc(cause="bg_cold")
                if did_gc:
                    gc_cause = "bg_cold"
                t_cold = 0

            # (b) free ratio 임계 기반 GC
            if (not did_gc) and self._should_gc_by_free_ratio():
                did_gc2 = self._do_gc(cause="low_free")
                if did_gc2:
                    did_gc = True
                    gc_cause = "low_free"

            # -------------------------
            # 4) per-op trace
            # -------------------------
            if self.enable_trace and self.trace:
                self.trace["step"].append(int(self.ops))
                self.trace["free_pages"].append(int(getattr(self.ssd, "free_pages", 0)))
                self.trace["device_writes"].append(int(getattr(self.ssd, "device_write_pages", 0)))
                self.trace["gc_count"].append(int(getattr(self.ssd, "gc_count", 0)))
                self.trace["gc_event"].append(gc_cause)

    # ========================================================
    # Convenience properties
    # ========================================================

    @property
    def pages_per_block(self) -> int:
        return _pages_per_block(self.ssd)

    @property
    def free_blocks(self) -> int:
        blocks = getattr(self.ssd, "blocks", []) or []
        return _free_block_count(blocks)