from __future__ import annotations

"""
models.py

이 파일은 SSD를 아주 단순화한 시뮬레이터용 물리 모델을 정의합니다.

핵심 목표
---------
1) 최소한의 NAND 제약을 재현
   - Page 단위 program(쓰기)은 가능하지만,
   - Block 단위 erase(소거)만 가능하다는 제약을 모델링합니다.
   - overwrite는 in-place가 아니라 out-of-place(write new + invalidate old)로 처리합니다.

2) GC 실험이 가능하도록 상태/카운터를 제공
   - 각 Block은 valid/invalid/free 페이지 수, erase 횟수(wear), 최근 활동(step) 등의
     정책 입력(feature)이 되는 값들을 유지합니다.
   - SSD는 mapping(LPN->PPN)과 reverse map(PPN->LPN)을 유지하여
     GC 시 “어떤 valid 페이지를 옮겨야 하는지”를 알 수 있습니다.

3) 재현성과 무결성
   - rng_seed 기반으로 free block 선택 등에서 재현 가능한 랜덤을 사용합니다.
   - “목적지 블록 보장(destination guarantee)” 로직을 포함해서
     GC 중간에 목적지가 없어 멈추는 상황을 최대한 막습니다.

용어
----
- LPN(Logical Page Number): 호스트가 보는 논리 주소
- PPN(Physical Page Number): (block_idx, page_idx)로 표현되는 물리 위치
- VALID: 현재 mapping이 가리키는 “유효 데이터”
- INVALID: overwrite/TRIM 등으로 더 이상 유효하지 않은 “쓰레기”
- FREE: 아직 한 번도 program되지 않은 페이지

주의(이 모델의 단순화)
----------------------
- 실제 SSD의 FTL, 채널/다이 병렬성, wear-leveling, read disturb, GC/host concurrency 등은 생략합니다.
- latency는 여기서 직접 계산하지 않고, 상위 simulator가 필요하면 별도로 모델링합니다.
- three_stream(옵션)은 개념 실험용이며, 실제 SSD의 stream 관리와 동일하다고 주장하지 않습니다.
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Callable, List, Any
import random
import time


# ============================================================
# Basic types
# ============================================================

class PageState(Enum):
    """페이지 상태(단순 3상태)."""
    FREE = 0
    VALID = 1
    INVALID = 2


class Block:
    """
    Physical erase-block.

    이 블록 객체는 GC 정책이 보고 싶어 하는 feature들을 최대한 한 곳에 모아 둡니다.

    보유 상태
    --------
    - pages: 페이지별 상태(PageState)
    - valid_count / invalid_count: 빠른 집계를 위한 캐시 카운터
    - erase_count: 마모(wear) proxy
    - last_prog_step / last_invalid_step: 최근 활동 시각(step)
    - inv_ewma: invalid 이벤트 기반 hotness proxy (0~1 근사)
    - age_counter: “마지막 erase 이후 상대 나이”를 단순 카운트로 유지(특허/정책 실험용)
    - trimmed_pages: TRIM으로 invalid 처리된 페이지 카운트(정책/분석용)
    - stream_id / pool: 선택적으로 stream 기반 라우팅을 실험할 때 사용
    """

    def __init__(self, pages_per_block: int):
        self.pages_per_block = int(pages_per_block)

        # 페이지 상태 배열
        self.pages: List[PageState] = [PageState.FREE] * self.pages_per_block

        # 카운터(빠른 집계용)
        self.valid_count = 0
        self.invalid_count = 0
        self.erase_count = 0

        # “최근 활동” (정책의 age/staleness proxy)
        self.last_invalid_step = 0
        self.last_prog_step = 0

        # invalid 이벤트 기반 hotness EWMA
        self.inv_ewma = 0.0

        # (단순화) age counter: 다른 블록이 erase될 때마다 +1 같은 방식으로 증가시키는 용도
        self.age_counter = 0

        # 스트림/풀 태그(옵션)
        self.stream_id = "user"   # 'user' | 'hot' | 'cold'
        self.pool = "gen"         # 'hot' | 'cold' | 'gen' (정책에서 cold-bias 등에 활용)

        # TRIM 이벤트 집계(옵션)
        self.trimmed_pages = 0

    # -------------------------
    # Derived helpers (정책에서 읽기 편하게)
    # -------------------------

    @property
    def free_count(self) -> int:
        """현재 FREE 페이지 수."""
        return self.pages_per_block - self.valid_count - self.invalid_count

    def invalid_ratio(self) -> float:
        """used(=valid+invalid) 중 invalid 비율."""
        used = self.valid_count + self.invalid_count
        return (self.invalid_count / used) if used > 0 else 0.0

    def wear_norm(self, max_erase_seen: int) -> float:
        """
        실험 내 상대 정규화 wear.
        - max_erase_seen는 현재 블록 집합에서의 최대 erase_count 같은 값을 넣습니다.
        """
        return (self.erase_count / max_erase_seen) if max_erase_seen > 0 else 0.0

    def last_activity(self) -> int:
        """최근 활동 시각(프로그램/무효화 중 더 최신)."""
        return max(int(self.last_prog_step), int(self.last_invalid_step))

    # -------------------------
    # Low-level ops
    # -------------------------

    def allocate_free_page(self) -> Optional[int]:
        """
        FREE 페이지를 하나 찾아 VALID로 바꾸고 페이지 인덱스를 반환합니다.
        - 없으면 None.

        정책/FTL 관점에서:
        - out-of-place write에서는 새 페이지 할당(program) 단계에 해당합니다.
        """
        for idx, st in enumerate(self.pages):
            if st == PageState.FREE:
                self.pages[idx] = PageState.VALID
                self.valid_count += 1
                return idx
        return None

    def invalidate_page(self, page_idx: int, step: int = 0, lam: float = 0.02) -> None:
        """
        VALID 페이지를 INVALID로 바꿉니다.
        - overwrite(기존 데이터 무효화), GC migration 후 원본 무효화 등에 사용됩니다.

        hotness(EWMA) 업데이트:
        - invalid 이벤트가 발생하면 inv_ewma를 조금 올립니다.
        - 시간이 지남에 따라(다른 이벤트가 없으면) 별도 디케이는 하지 않지만,
          invalid 이벤트가 없으면 “최근에 invalid를 잘 안 만드는 블록”으로 해석될 수 있습니다.
        """
        if 0 <= page_idx < self.pages_per_block and self.pages[page_idx] == PageState.VALID:
            self.pages[page_idx] = PageState.INVALID
            self.valid_count -= 1
            self.invalid_count += 1
            self.last_invalid_step = int(step)

            # invalid 이벤트 기반 hotness EWMA
            lam = float(lam)
            self.inv_ewma = (1.0 - lam) * self.inv_ewma + lam * 1.0

    def erase(self) -> None:
        """
        블록 erase(소거).
        - 모든 페이지를 FREE로 리셋
        - valid/invalid 카운터 리셋
        - erase_count 증가(wear++)
        - age/hotness/TRIM 등 부가 상태 리셋

        NOTE:
        - 이 함수는 블록 내부 상태 리셋만 담당합니다.
        - SSD 레벨에서 추가 훅(예: 다른 블록 age_counter 증가)을 적용하려면
          SSD.erase_block()을 통해 호출하는 것을 권장합니다.
        """
        self.pages = [PageState.FREE] * self.pages_per_block
        self.valid_count = 0
        self.invalid_count = 0

        self.erase_count += 1

        # 최근 활동/온도/부가 카운터 리셋
        self.last_invalid_step = 0
        self.last_prog_step = 0
        self.inv_ewma = 0.0

        self.trimmed_pages = 0
        self.age_counter = 0


# ============================================================
# SSD model
# ============================================================

class SSD:
    """
    Minimal SSD model.

    제공 API(핵심)
    ------------
    - write_lpn(lpn): 호스트 write (out-of-place)
    - trim_lpn(lpn): TRIM/삭제 (쓰기 없이 invalidate)
    - collect_garbage(policy): victim 선택 -> valid migration -> victim erase

    내부 구성요소
    ------------
    - blocks: Block 리스트
    - mapping: LPN -> (block_idx, page_idx)
    - reverse_map: (block_idx, page_idx) -> LPN

    재현성
    ------
    - rng_seed로 random.Random을 사용합니다.
      free block 선택 등에서 매번 같은 조건이면 같은 선택을 기대할 수 있습니다.

    안전장치(무결성)
    ---------------
    - destination guarantee:
      GC 중간에 목적지 페이지가 없어서 멈추지 않도록,
      미리 목적지 블록을 확보하거나 all-invalid 블록을 회수해 빈 블록을 만들 수 있습니다.
    """

    # 호스트에게 “완전 빈 블록”을 전부 내주면 GC가 스크래치 공간을 잃어버릴 수 있습니다.
    # 그래서 최소한 RESERVED_FREE_BLOCKS 개는 남겨두도록 제한합니다.
    RESERVED_FREE_BLOCKS = 2

    def __init__(self, num_blocks: int, pages_per_block: int, rng_seed: int = 42):
        # Geometry
        self.num_blocks = int(num_blocks)
        self.pages_per_block = int(pages_per_block)

        # Blocks
        self.blocks = [Block(self.pages_per_block) for _ in range(self.num_blocks)]

        # RNG (reproducible)
        self.rng = random.Random(int(rng_seed))

        # “시간(step)” - write/trim/GC 마이그레이션 등 이벤트마다 증가
        self._step = 0

        # hotness EWMA 파라미터
        self.ewma_lambda = 0.02

        # Metrics (write amplification 등)
        self.host_write_pages = 0
        self.device_write_pages = 0

        self.gc_count = 0
        self.gc_total_time = 0.0
        self.gc_durations: List[float] = []

        # GC 이벤트 로그(분석용)
        self.gc_event_log: List[Dict[str, Any]] = []

        # Mappings
        self.mapping: Dict[int, Tuple[int, int]] = {}            # LPN -> (b, p)
        self.reverse_map: Dict[Tuple[int, int], int] = {}        # (b, p) -> LPN

        # Write heads (single stream)
        self.active_block_idx: Optional[int] = None

        # Optional 3-stream mode
        self.three_stream = False
        self.stream_active: Dict[str, Optional[int]] = {"user": None, "hot": None, "cold": None}

        # Hotness classification (3-stream)
        self.hotness_mode = "recency"            # "recency" | "oracle"
        self.recency_tau = 200                   # 최근 업데이트가 tau 이내면 hot으로 간주
        self.oracle_hot_cut: Optional[int] = None
        self.lpn_last_write: Dict[int, int] = {} # LPN -> last host-write step

        # Optional debug: 정책 점수 스냅샷을 외부에서 주입해 기록할 수 있음
        # score_probe(blocks) -> dict (block_idx -> score_detail) 같은 형태를 기대
        self.score_probe: Optional[Callable[[List[Block]], Any]] = None

    # -------------------------
    # Derived
    # -------------------------

    @property
    def total_pages(self) -> int:
        """물리 총 페이지 수."""
        return self.num_blocks * self.pages_per_block

    @property
    def free_pages(self) -> int:
        """SSD 전체 FREE 페이지 수."""
        return sum(b.free_count for b in self.blocks)

    @property
    def free_blocks(self) -> int:
        """완전 빈 블록(FREE=pages_per_block) 개수."""
        return sum(1 for b in self.blocks if b.free_count == self.pages_per_block)

    # -------------------------
    # Erase wrapper (SSD-level hook)
    # -------------------------

    def erase_block(self, block_idx: int) -> None:
        """
        SSD 레벨 erase.

        왜 Block.erase를 직접 쓰지 않나?
        -------------------------------
        - 블록을 하나 지우는 순간, 다른 블록들은 상대적으로 더 늙는다와 같은
          실험용 나이(age_counter) 모델을 넣고 싶을 수 있습니다.
        - 이런 SSD 레벨 훅을 한 곳에서 통일하려고 SSD.erase_block을 둡니다.
        """
        block_idx = int(block_idx)

        # (실험용) 이 블록을 지우기 전에, 데이터(used)가 있는 나머지 블록 age를 +1
        for i, b in enumerate(self.blocks):
            if i == block_idx:
                continue
            used = b.valid_count + b.invalid_count
            if used > 0:
                b.age_counter += 1

        # 실제 erase 수행
        self.blocks[block_idx].erase()

        # mapping/reverse_map 정합성은:
        # - 일반적으로 GC에서는 migration 후 victim을 erase하므로 victim 안에는 reverse_map이 남지 않아야 합니다.
        # - 하지만 강제로 지우는 상황을 대비해, 필요하면 여기서 (block_idx, *) reverse_map cleanup을 할 수도 있습니다.
        #   (현재 구현은 GC 흐름에서 정리된다는 가정)

    # ============================================================
    # Destination guarantee helpers
    # ============================================================

    def _free_block_indices(self, exclude_idx: int | None = None) -> list[int]:
        """완전히 빈 블록 인덱스 목록(옵션: 특정 블록 제외)."""
        return [
            i for i, b in enumerate(self.blocks)
            if b.free_count == self.pages_per_block and i != exclude_idx
        ]

    def _find_free_block_index(self, exclude_idx: int | None = None, *, for_host: bool = False) -> int | None:
        """
        free 페이지를 가진 블록 하나를 찾습니다.

        정책
        ----
        - for_host=True: 호스트에게는 예약 블록(RESERVED_FREE_BLOCKS)을 내주지 않도록 제한합니다.
          즉, 완전 빈 블록이 너무 적으면(host가 가져가면 GC가 죽을 정도라면) 호스트 배정을 막습니다.
        - for_host=False: GC 목적지 확보는 예약 포함 허용(스크래치가 필요하므로).

        반환
        ----
        - 조건에 맞는 블록 인덱스 또는 None
        """
        empties = self._free_block_indices(exclude_idx)

        if for_host:
            # 완전 빈 블록이 (예약 포함) 충분히 있을 때만 그 중에서 배정
            if len(empties) > self.RESERVED_FREE_BLOCKS:
                usable = empties[:-self.RESERVED_FREE_BLOCKS]
                return self.rng.choice(usable) if usable else None

            # 완전 빈 블록이 예약밖에 없다면, “부분 free 블록”을 찾아서 호스트에게 배정
            for i, b in enumerate(self.blocks):
                if i == exclude_idx:
                    continue
                if 0 < b.free_count < self.pages_per_block:
                    return i
            return None

        # GC 목적지: 예약 포함 허용
        if empties:
            return self.rng.choice(empties)

        # 완전 빈 블록이 없으면, 부분 free 블록 중 하나
        for i, b in enumerate(self.blocks):
            if i == exclude_idx:
                continue
            if b.free_count > 0:
                return i
        return None

    def _ensure_active_block(self, exclude_idx: int | None = None, *, for_host: bool = False) -> int | None:
        """
        활성(쓰기 목적지) 블록을 보장합니다.

        사용처
        ----
        - host write: for_host=True (예약선 지키기)
        - GC migration: for_host=False (예약 포함 목적지 확보)

        절차
        ----
        1) 기존 active_block이 아직 free 페이지가 있으면 재사용
        2) 없으면 새 목적지 블록 탐색(_find_free_block_index)
        3) 그래도 없으면 all-invalid 블록을 찾아 erase하여 빈 블록을 만든 뒤 사용
        """
        cand = self.active_block_idx

        # 1) 기존 active 재사용
        if cand is not None and cand != exclude_idx and self.blocks[cand].free_count > 0:
            if for_host:
                # 호스트가 “완전 빈 블록”을 잡아먹는 게 예약선 침범인지 확인
                empties = self._free_block_indices(exclude_idx)
                if (
                    len(empties) <= self.RESERVED_FREE_BLOCKS
                    and self.blocks[cand].free_count == self.pages_per_block
                ):
                    cand = None
            if cand is not None:
                return cand

        # 2) 새 블록 찾기
        j = self._find_free_block_index(exclude_idx=exclude_idx, for_host=for_host)
        if j is not None:
            self.active_block_idx = j
            return j

        # 3) all-invalid 회수(erase해서 빈 블록 생성)
        for i, b in enumerate(self.blocks):
            if i == exclude_idx:
                continue
            if b.valid_count == 0 and b.invalid_count > 0:
                self.erase_block(i)
                self.active_block_idx = i

                # 호스트용이면, 방금 만든 빈 블록도 예약선에 걸릴 수 있음
                if for_host:
                    empties = self._free_block_indices(exclude_idx)
                    if len(empties) <= self.RESERVED_FREE_BLOCKS:
                        continue
                return i

        return None

    def _ensure_destination_block(self, victim_idx: int) -> int | None:
        """
        GC migration용 목적지 블록을 확보합니다.
        - GC는 예약 블록 사용을 허용합니다(for_host=False).

        목적
        ----
        - victim에서 VALID 페이지를 옮기기 시작했는데,
          목적지가 없어 중간에 멈추는 상황을 방지합니다.
        """
        j = self._find_free_block_index(exclude_idx=victim_idx, for_host=False)
        if j is not None:
            return j

        # all-invalid 회수
        for i, b in enumerate(self.blocks):
            if i == victim_idx:
                continue
            if b.valid_count == 0 and b.invalid_count > 0:
                self.erase_block(i)
                return i

        # 마지막: active 보장(예약 포함)
        return self._ensure_active_block(exclude_idx=victim_idx, for_host=False)

    def _alloc_block_for_migration(self, victim_idx: int, lpn: int) -> int | None:
        """
        migration 목적지 블록 선택.

        기본 정책
        --------
        - single-stream: active_block 사용(없으면 확보)
        - three-stream: victim의 stream_id를 따라가도록 시도(단순 버전)

        NOTE:
        - 실제 SSD의 stream 유지/재분류는 훨씬 복잡합니다.
          여기서는 “정책 아이디어 실험용” 정도로만 사용하세요.
        """
        if self.three_stream:
            victim_stream = getattr(self.blocks[victim_idx], "stream_id", "user")
            stream = victim_stream
            self._ensure_stream_block(stream, exclude_idx=victim_idx)
            dst = self.stream_active[stream]
            if dst is None:
                dst = self._find_free_block_index(exclude_idx=victim_idx)
            return dst

        return self._ensure_active_block(exclude_idx=victim_idx, for_host=False)

    # ============================================================
    # GC
    # ============================================================

    def collect_garbage(self, policy: Callable[[List[Block]], Optional[int]], cause: str = "manual") -> None:
        """
        Garbage Collection 실행.

        입력
        ----
        - policy(blocks) -> victim_idx (또는 None)
        - cause: 이벤트 로그에 남기는 문자열(예: "manual", "warmup", "low_free", ...)

        동작 흐름
        --------
        1) victim 선택
        2) victim이 all-invalid면 목적지 없이 즉시 erase
        3) 목적지 블록 선확보(destination guarantee)
        4) victim의 VALID 페이지를 하나씩 다른 블록으로 migration
        5) victim erase + 통계 기록

        무결성 포인트
        -------------
        - migration 시 mapping/reverse_map을 항상 같이 갱신합니다.
        - victim 내부 VALID를 INVALID로 바꾸고(reverse_map 제거),
          새 목적지에는 reverse_map을 등록합니다.
        """
        # 1) victim 선택
        victim_idx = policy(self.blocks)
        if victim_idx is None:
            # fallback: invalid_count가 큰 블록
            victim_idx = max(
                range(len(self.blocks)),
                key=lambda i: getattr(self.blocks[i], "invalid_count", 0),
                default=None,
            )
            if victim_idx is None:
                raise RuntimeError("No victim block available for GC")

        victim = self.blocks[victim_idx]

        # 2) all-invalid면 즉시 erase(목적지 불필요)
        if victim.valid_count == 0 and victim.invalid_count > 0:
            self.erase_block(victim_idx)
            return

        # 3) 목적지 블록 선확보
        _ = self._ensure_destination_block(victim_idx)

        # (옵션) 정책 점수 스냅샷: 디버깅/분석용
        probe_detail = None
        if self.score_probe is not None:
            try:
                snap = self.score_probe(self.blocks)
                if isinstance(snap, dict):
                    probe_detail = snap.get(victim_idx)
            except Exception:
                probe_detail = None

        # victim 상태 스냅샷(로그용)
        v_valid, v_invalid = victim.valid_count, victim.invalid_count
        v_ewma, v_erase = victim.inv_ewma, victim.erase_count

        t0 = time.perf_counter()
        moved_valid = 0

        # 4) VALID 페이지 migration
        for p_idx, st in enumerate(victim.pages):
            if st != PageState.VALID:
                continue

            lpn = self.reverse_map.get((victim_idx, p_idx))
            if lpn is None:
                # reverse_map이 없으면 정합성이 깨진 상태일 수 있음(하지만 안전하게 스킵)
                continue

            dst_idx = self._alloc_block_for_migration(victim_idx, lpn)
            if dst_idx is None:
                dst_idx = self._ensure_destination_block(victim_idx)
                if dst_idx is None:
                    raise RuntimeError("No destination block for migration")

            dst_p = self.blocks[dst_idx].allocate_free_page()
            if dst_p is None:
                # 목적지 블록이 꽉 찼으면 active를 리셋하고 다시 확보
                self.active_block_idx = None
                dst_idx = self._ensure_destination_block(victim_idx)
                if dst_idx is None:
                    raise RuntimeError("Allocator inconsistency during GC")
                dst_p = self.blocks[dst_idx].allocate_free_page()
                if dst_p is None:
                    raise RuntimeError("Allocator inconsistency during GC (second)")

            # 새 위치 기록 + 원본 무효화
            self.blocks[dst_idx].last_prog_step = self._step
            victim.invalidate_page(p_idx, step=self._step, lam=self.ewma_lambda)

            self.reverse_map.pop((victim_idx, p_idx), None)
            self.mapping[lpn] = (dst_idx, dst_p)
            self.reverse_map[(dst_idx, dst_p)] = lpn

            # migration은 device write로 카운트(호스트 write는 아님)
            self.device_write_pages += 1
            moved_valid += 1

        # 5) victim erase + 통계
        free_before = victim.free_count
        freed_pages = self.pages_per_block - free_before

        self.erase_block(victim_idx)  # SSD 레벨 훅(나이 증가 등) 포함
        self.gc_count += 1

        dt = time.perf_counter() - t0
        self.gc_total_time += dt
        self.gc_durations.append(dt)

        ev = {
            "step": self._step,
            "cause": cause,
            "victim": victim_idx,
            "moved_valid": moved_valid,
            "freed_pages": freed_pages,
            "gc_s": dt,
            "free_blocks_after": self.free_blocks,

            # victim 상태 스냅샷
            "v_valid": v_valid,
            "v_invalid": v_invalid,
            "v_inv_ewma": v_ewma,
            "v_erase": v_erase,

            # 선택: 정책 점수 스냅샷
            **({"score_detail": probe_detail} if probe_detail is not None else {}),
        }
        self.gc_event_log.append(ev)

    # ============================================================
    # Hotness / stream helpers (3-stream 옵션)
    # ============================================================

    def _is_hot_lpn(self, lpn: int) -> bool:
        """
        LPN이 hot인지 판정.

        - oracle 모드: LPN 범위를 기준으로 hot/cold를 정답처럼 구분(실험용)
        - recency 모드: 최근에 다시 쓰였는지(temporal locality)로 hot을 추정
        """
        if self.hotness_mode == "oracle" and self.oracle_hot_cut is not None:
            return lpn < self.oracle_hot_cut

        last = self.lpn_last_write.get(lpn, -10**12)
        return (self._step - last) <= int(self.recency_tau)

    def _find_block_with_free(self, exclude_idx: Optional[int] = None) -> Optional[int]:
        """free 페이지가 있는 블록 중 하나를 랜덤 선택(옵션: 특정 블록 제외)."""
        cands = [i for i, b in enumerate(self.blocks) if b.free_count > 0 and i != exclude_idx]
        return self.rng.choice(cands) if cands else None

    def _ensure_stream_block(self, stream: str, exclude_idx: Optional[int] = None) -> None:
        """
        3-stream 모드에서 stream별 활성 블록을 확보합니다.

        호스트 write에서는 예약 블록을 침범하지 않도록,
        - 완전 빈 블록이 충분하면 예약 제외 영역에서 선택
        - 아니면 부분 free 블록을 허용(실험 편의)
        """
        idx = self.stream_active.get(stream)
        if idx is not None and self.blocks[idx].free_count > 0:
            empties = self._free_block_indices(exclude_idx)
            if len(empties) <= self.RESERVED_FREE_BLOCKS and self.blocks[idx].free_count == self.pages_per_block:
                idx = None
            else:
                return

        empties = self._free_block_indices(exclude_idx)
        if len(empties) > self.RESERVED_FREE_BLOCKS:
            usable = empties[:-self.RESERVED_FREE_BLOCKS]
            chosen = self.rng.choice(usable) if usable else None
        else:
            chosen = None
            for i, b in enumerate(self.blocks):
                if i == exclude_idx:
                    continue
                if 0 < b.free_count < self.pages_per_block:
                    chosen = i
                    break

        self.stream_active[stream] = chosen
        if chosen is not None:
            self.blocks[chosen].stream_id = stream

    # ============================================================
    # TRIM
    # ============================================================

    def trim_lpn(self, lpn: int) -> None:
        """
        TRIM(논리 삭제).

        효과
        ----
        - mapping에서 LPN을 제거하고,
        - 해당 PPN의 페이지를 VALID -> INVALID로 바꿉니다.
        - 쓰기(host/device write)는 증가하지 않습니다.
        - trimmed_pages 카운터를 증가시켜 TRIM 이벤트를 추적합니다.

        NOTE:
        - 실제 SSD에서는 TRIM이 즉시 물리 invalid로 반영되기도 하고,
          내부적으로 지연되기도 하지만, 여기서는 단순화를 위해 즉시 invalid 처리합니다.
        """
        self._step += 1

        pos = self.mapping.pop(lpn, None)
        if pos is None:
            return

        b, p = pos
        blk = self.blocks[b]

        if blk.pages[p] == PageState.VALID:
            blk.pages[p] = PageState.INVALID
            blk.valid_count -= 1
            blk.invalid_count += 1

            blk.last_invalid_step = self._step
            blk.inv_ewma = (1.0 - self.ewma_lambda) * blk.inv_ewma + self.ewma_lambda * 1.0

            blk.trimmed_pages += 1

        self.reverse_map.pop((b, p), None)

    # ============================================================
    # Host write (out-of-place)
    # ============================================================

    def write_lpn(self, lpn: int) -> None:
        """
        호스트 write 1회(out-of-place).

        흐름
        ----
        1) step 증가
        2) 기존 mapping이 있으면 그 PPN을 invalidate (overwrite 효과)
        3) 목적지 블록을 선택(단일/스트림)
        4) FREE 페이지를 allocate하여 VALID로 만들고 mapping 갱신
        5) host_write_pages, device_write_pages 증가

        무결성 포인트
        -------------
        - mapping과 reverse_map은 항상 함께 갱신합니다.
        - invalidate 시 reverse_map에서 기존 PPN을 제거합니다.
        """
        self._step += 1

        # 1) overwrite 처리: 기존 mapping이 있으면 old PPN invalidate
        if lpn in self.mapping:
            b, p = self.mapping[lpn]
            self.blocks[b].invalidate_page(p, step=self._step, lam=self.ewma_lambda)
            self.reverse_map.pop((b, p), None)

        # 2) 목적지 블록 선택
        if self.three_stream:
            stream = "hot" if self._is_hot_lpn(lpn) else "user"
            self._ensure_stream_block(stream)
            b_idx = self.stream_active[stream]
        else:
            self._ensure_active_block(for_host=True)
            b_idx = self.active_block_idx

        if b_idx is None:
            # 마지막 방어: 어떤 free 블록이라도 찾아보기(여기서도 없으면 정말 위험한 상태)
            b_idx = self._find_free_block_index()
            if b_idx is None:
                raise RuntimeError("No free page before GC")

        # 3) 페이지 allocate
        p_idx = self.blocks[b_idx].allocate_free_page()
        if p_idx is None:
            # 목적지 블록이 찼으면 “회전” (새 목적지 확보 후 재시도)
            if self.three_stream:
                self._ensure_stream_block(stream)
                b_idx = self.stream_active[stream]
            else:
                self._ensure_active_block(for_host=True)
                b_idx = self.active_block_idx

            if b_idx is None:
                b_idx = self._find_free_block_index()
            if b_idx is None:
                raise RuntimeError("No free page after rotate")

            p_idx = self.blocks[b_idx].allocate_free_page()
            if p_idx is None:
                raise RuntimeError("Allocator inconsistency")

        # 4) mapping/reverse_map 갱신 + 통계
        self.lpn_last_write[lpn] = self._step
        self.mapping[lpn] = (b_idx, p_idx)
        self.reverse_map[(b_idx, p_idx)] = lpn

        self.host_write_pages += 1
        self.device_write_pages += 1

        self.blocks[b_idx].last_prog_step = self._step