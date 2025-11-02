# simulator.py
"""
Simulator — drop-in shim (backward-compatible)

- 하위호환:
  * __init__(..., enable_trace=..., bg_gc_every=...) 안전 흡수
  * sim.ssd 별칭 제공 (호출부가 sim.ssd를 기대)
  * sim.gc_policy (외부에서 주입 시 우선 사용)
  * sim.trace["gc_event"] 이벤트 로그 제공
- 기능:
  * cfg(SimConfig) 또는 SSD 인스턴스를 받아 초기화
  * 워크로드 실행(run): write/trim 지원
  * BG-GC: 간단한 cadence(주기) + free 비율 임계 연동
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Any, Tuple, Dict
import time

# -------------------------
# 안전한 의존성 로딩
# -------------------------
try:
    # 프로젝트 모델: SSD 이름으로 정의되어 있음
    from models import SSD as Device, Block, PageState
except Exception:
    # 타입 안전을 위해 최소 더미 제공
    Device = object  # type: ignore
    Block = object   # type: ignore
    class PageState:  # type: ignore
        FREE = 0; VALID = 1; INVALID = 2

try:
    # 정책 팩토리(함수형 정책: policy(blocks) -> victim_idx)
    from gc_algos import get_gc_policy
except Exception:
    def get_gc_policy(name: str) -> Callable[[List[Any]], Optional[int]]:
        # 최후방어: invalid_count가 최대인 블록
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

# -------------------------
# BG-GC cadence
# -------------------------
@dataclass
class BGSchedule:
    """pool별 GC 호출 주기(ops 기준). 0이면 주기 기반 off."""
    every_hot: int = 0
    every_cold: int = 0
    every_gen: int = 0

# -------------------------
# 유틸
# -------------------------
def _now_step() -> int:
    # 초 단위 step으로 충분 (상대 비교/로그용)
    return int(time.time())

def _free_block_count(blocks: List[Any]) -> int:
    cnt = 0
    for b in blocks:
        used = int(getattr(b, "valid_count", 0)) + int(getattr(b, "invalid_count", 0))
        if used == 0:
            cnt += 1
    return cnt

def _total_blocks(blocks: List[Any]) -> int:
    return len(blocks)

def _pages_per_block(ssd: Any) -> int:
    if hasattr(ssd, "pages_per_block"):
        return int(getattr(ssd, "pages_per_block"))
    # block 객체에 pages가 있다면 길이로 추정
    blocks = getattr(ssd, "blocks", []) or []
    if blocks and hasattr(blocks[0], "pages"):
        return len(getattr(blocks[0], "pages"))
    return 0

# -------------------------
# Simulator
# -------------------------
class Simulator:
    def __init__(
        self,
        cfg_or_device: Any,
        policy_name: str = "greedy",
        cold_pool: bool = True,
        bg: Optional[BGSchedule] = None,
        **kwargs
    ):
        """
        하위호환 키워드:
          - enable_trace: bool (기본 False)
          - bg_gc_every: int  (기본 0, gen pool에 대한 주기)
        """
        # ---- traces & opts
        self.trace: Dict[str, List[Any]] = {"gc_event": []}
        self.enable_trace: bool = bool(kwargs.pop("enable_trace", False))

        # BG cadence (기본값 + 인자 반영)
        bg_every = int(kwargs.pop("bg_gc_every", 0) or 0)
        self.bg = bg or BGSchedule(
            every_hot=0,
            every_cold=max(bg_every * 4, 0) if bg_every else 0,  # 기본적으로 cold는 더 드물게
            every_gen=bg_every,
        )

        # 정책
        self._policy_name = (policy_name or "greedy").lower()
        self._policy = get_gc_policy(self._policy_name)
        self.gc_policy: Optional[Callable[[List[Any]], Optional[int]]] = None  # 외부 주입 우선

        # ---- cfg/device 수용
        self.cfg = None
        self.ssd: Any = None

        # SimConfig 비슷한 객체가 들어왔는지 검사
        if hasattr(cfg_or_device, "num_blocks") and hasattr(cfg_or_device, "pages_per_block"):
            self.cfg = cfg_or_device
            # 사전 준비 훅(있으면)
            try:
                if hasattr(self.cfg, "prepare"):
                    self.cfg.prepare()
            except Exception:
                pass
            nb = int(getattr(self.cfg, "num_blocks"))
            ppb = int(getattr(self.cfg, "pages_per_block"))
            seed = int(getattr(self.cfg, "rng_seed", 42))
            self.ssd = Device(nb, ppb, rng_seed=seed)
        else:
            # 이미 SSD 인스턴스
            self.ssd = cfg_or_device

        # 별칭: 오래된 코드 호환
        self.dev = self.ssd

        # free threshold(비율) — cfg가 있으면 따라감
        self.gc_free_block_threshold = float(
            getattr(self.cfg, "gc_free_block_threshold", 0.0)
        ) if self.cfg is not None else 0.0

        # 내부 스텝 카운터 (ops 수)
        self.ops: int = 0

        # 기타 알 수 없는 키워드들은 조용히 무시 (하위호환 흡수)
        _ = kwargs  # noqa

    # --------------- 내부: GC 선택/실행 ---------------
    def _choose_policy(self) -> Callable[[List[Any]], Optional[int]]:
        """외부 주입(sim.gc_policy)이 있으면 그것을 우선 사용."""
        return self.gc_policy or self._policy

    def _do_gc(self, cause: str = "bg") -> bool:
        """한 번의 GC 실행. victim 선택 → SSD.collect_garbage 호출."""
        blocks = getattr(self.ssd, "blocks", []) or []
        if not blocks:
            return False
        policy = self._choose_policy()

        # SSD API: collect_garbage(policy, cause="...") 지원
        if hasattr(self.ssd, "collect_garbage"):
            before_inv = sum(int(getattr(b, "invalid_count", 0)) for b in blocks)
            try:
                self.ssd.collect_garbage(policy=policy, cause=cause)
            except TypeError:
                # 구버전: 인자명 다른 경우 대비
                self.ssd.collect_garbage(policy)
            after_inv = sum(int(getattr(b, "invalid_count", 0)) for b in blocks)
            moved = max(0, before_inv - after_inv)

            if self.enable_trace:
                self.trace["gc_event"].append({
                    "step": self.ops,
                    "cause": cause,
                    "moved_valid": int(getattr(self.ssd, "last_gc_moved_valid", 0)),
                    "free_blocks": _free_block_count(blocks),
                })
            return True
        return False

    def _should_gc_by_free_ratio(self) -> bool:
        """free 블록 비율이 cfg 임계치보다 낮으면 GC."""
        thr = float(self.gc_free_block_threshold or 0.0)
        if thr <= 0.0:
            return False
        blocks = getattr(self.ssd, "blocks", []) or []
        if not blocks:
            return False
        free = _free_block_count(blocks)
        ratio = free / max(1, len(blocks))
        return ratio < thr

    # --------------- 퍼블릭: 워크로드 실행 ---------------
    def run(self, workload: List[Any]) -> None:
        """
        workload 형식:
          - [lpn, lpn, ...]               (기본)
          - [("write"| "trim", lpn), ...] (enable_trim=True인 경우)
        """
        # 간단한 BG cadence 카운터
        t_hot = t_cold = t_gen = 0

        for op in workload:
            # --- op decode ---
            if isinstance(op, (list, tuple)) and len(op) == 2:
                kind, lpn = op[0], int(op[1])
                if kind == "trim":
                    if hasattr(self.ssd, "trim_lpn"):
                        self.ssd.trim_lpn(lpn)
                else:
                    # default write
                    if hasattr(self.ssd, "write_lpn"):
                        self.ssd.write_lpn(lpn)
            else:
                # 기본: 정수 = write LPN
                lpn = int(op)
                if hasattr(self.ssd, "write_lpn"):
                    self.ssd.write_lpn(lpn)

            self.ops += 1
            t_hot += 1; t_cold += 1; t_gen += 1

            # --- BG cadence 기반 GC (옵션) ---
            did = False
            if self.bg.every_gen and t_gen >= self.bg.every_gen:
                did = self._do_gc(cause="bg_gen")
                t_gen = 0
            # (필요 시 hot/cold도 사용)
            if not did and self.bg.every_hot and t_hot >= self.bg.every_hot:
                did = self._do_gc(cause="bg_hot")
                t_hot = 0
            if not did and self.bg.every_cold and t_cold >= self.bg.every_cold:
                did = self._do_gc(cause="bg_cold")
                t_cold = 0

            # --- free 비율 임계 기반 GC (cfg가 설정된 경우) ---
            if not did and self._should_gc_by_free_ratio():
                self._do_gc(cause="low_free")

    # --------------- 편의 프로퍼티 ---------------
    @property
    def pages_per_block(self) -> int:
        return _pages_per_block(self.ssd)

    @property
    def free_blocks(self) -> int:
        blocks = getattr(self.ssd, "blocks", []) or []
        return _free_block_count(blocks)