"""
workload.py

이 프로젝트에서 호스트가 어떤 논리 주소(LPN)에 어떤 순서로 접근했는가를 생성하는 모듈.

핵심 역할
---------
SSD GC 시뮬레이터는 결국 워크로드(workload) 에 의해 행동이 결정된다.
- 같은 GC 정책이라도, update 비율이 달라지면 invalid가 쌓이는 방식이 달라지고,
- hot 데이터 비율/가중치가 달라지면 특정 영역에 덮어쓰기가 몰리는 현상이 달라지며,
- TRIM을 넣으면 “호스트가 삭제한 데이터”가 GC 효율에 영향을 주게 된다.

따라서 workload.py는 실험 재현성의 출발점이다.
- 입력 파라미터(n_ops, update_ratio, hot_ratio, hot_weight, seed, trim 옵션)가 같으면
  항상 동일한 워크로드를 생성한다.

반환 형식(Contract)
-------------------
make_workload는 두 가지 모드를 지원한다.

1) enable_trim=False (기본)
   - 기존 인터페이스 그대로:  [lpn, lpn, lpn, ...]

2) enable_trim=True
   - 이벤트를 명시: [("write"|"trim", lpn), ...]

Simulator.run()은 이 두 형식을 모두 수용하도록 구현되어 있어야 한다.
(네 프로젝트에서는 simulator.py가 이를 처리한다.)

모델링 의도 / 단순화
-------------------
- LPN은 0..ssd_total_pages-1 범위에서만 생성한다.
- 신규 쓰기(new write)는 아직 한 번도 할당되지 않은 next_lpn을 순차 배정한다.
- 업데이트(update)는 이미 live 상태인 LPN 중 하나를 다시 선택한다.
- hot/cold는 oracle-style로 단순 분리:
    lpn < hot_cut  => hot
    그 외          => cold
  (hot_cut = int(ssd_total_pages * hot_ratio))

성능(생성기 내부 최적화)
-----------------------
워크로드 생성은 실험 반복에서 자주 호출되므로, update/trim 대상 선택이 느리면 전체가 느려진다.
이 파일은 live LPN 집합을 list 선형 탐색 대신,
- add/remove/choice가 평균 O(1)인 _IndexList로 관리한다.

흔한 함정(중요)
---------------
1) update_ratio 해석
   - update_ratio가 높아질수록 overwrite가 많아져 invalid가 잘 생기고,
     결과적으로 WAF/GC 부담이 커지는 경향이 나오는 게 자연스럽다.
   - 단, 시스템이 아직 비어있을 때는 update를 할 대상이 없으므로
     자동으로 new write로 fallback한다.

2) ssd_total_pages를 초과하지 않기
   - next_lpn이 ssd_total_pages에 도달하면 더 이상 new write를 만들 수 없고,
     이후는 update(또는 trim) 이벤트만 발생한다.
   - 이 파일은 next_lpn이 범위를 넘지 않도록 가드한다.

3) TRIM과 live-set
   - TRIM은 “현재 live인 LPN”에서만 발생한다.
   - TRIM이 발생하면 live-set에서 제거되어, 이후 update 후보에서 빠진다.
   - (즉, TRIM은 워크로드 레벨에서 삭제된 데이터는 다시 덮어쓰지 않는다를 의미한다.)
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Union
import random


# ------------------------------------------------------------
# 내부 유틸: 인덱스드 리스트 (O(1) add/remove/choice)
# ------------------------------------------------------------

class _IndexList:
    """
    O(1)에 가까운 성능으로 "집합처럼" 쓰기 위한 컨테이너.

    왜 필요한가?
    - live LPN 중 하나를 랜덤 선택(choice)해야 한다.
    - TRIM이 발생하면 LPN을 빠르게 제거(remove)해야 한다.
    - 단순 list로 remove를 하면 O(n)이라 sweep 반복에서 비용이 커진다.

    구현 개요:
    - _arr: 값들을 담는 리스트
    - _pos: 값 -> _arr 인덱스 매핑
    - remove 시 제거 대상 자리에 마지막 원소를 swap하여 O(1) 유지
    """

    def __init__(self):
        self._arr: List[int] = []
        self._pos: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._arr)

    def add(self, x: int) -> None:
        """중복 없이 추가."""
        if x in self._pos:
            return
        self._pos[x] = len(self._arr)
        self._arr.append(x)

    def remove(self, x: int) -> None:
        """존재하면 제거. 없으면 무시."""
        i = self._pos.pop(x, None)
        if i is None:
            return
        last = self._arr.pop()
        if i < len(self._arr):
            self._arr[i] = last
            self._pos[last] = i

    def choice(self, rng: random.Random) -> int:
        """랜덤 선택. 비어 있으면 예외."""
        if not self._arr:
            raise IndexError("empty _IndexList")
        return self._arr[rng.randrange(len(self._arr))]

    def to_list(self) -> List[int]:
        """디버깅/검증용."""
        return list(self._arr)


# ------------------------------------------------------------
# 메인 워크로드
# ------------------------------------------------------------

Workload = Union[List[int], List[Tuple[str, int]]]


def make_workload(
    n_ops: int,
    update_ratio: float,
    ssd_total_pages: int,
    rng_seed: int = 42,
    hot_ratio: float = 0.2,
    hot_weight: float = 0.7,
    # 신규 옵션 (기본 False라 기존 호출엔 영향 없음)
    enable_trim: bool = False,
    trim_ratio: float = 0.0,  # enable_trim=True일 때만 사용
) -> Workload:
    """
    워크로드 생성기.

    Parameters
    ----------
    n_ops:
        총 이벤트 개수(기본적으로 페이지 단위 write/trim 이벤트를 n_ops번 생성)
    update_ratio:
        (0~1) update(덮어쓰기) 비율. 높을수록 이미 존재하는 LPN을 더 자주 다시 선택한다.
    ssd_total_pages:
        LPN 공간 크기(논리 페이지 수). 생성되는 LPN은 0..ssd_total_pages-1 범위로 제한된다.
    rng_seed:
        재현성을 위한 시드. 동일 시드면 동일 workload가 생성된다.
    hot_ratio:
        hot 영역 크기 비율. hot_cut = int(ssd_total_pages * hot_ratio)
    hot_weight:
        update/trim에서 hot 풀을 우선 선택할 확률(0~1). hot 풀 비면 자동으로 cold로 fallback.
    enable_trim:
        True면 ("write"/"trim", lpn) 형식으로 반환하여 trim 이벤트를 포함한다.
    trim_ratio:
        enable_trim=True일 때, 각 op에서 trim 이벤트가 발생할 확률(0~1).

    Returns
    -------
    enable_trim=False:
        List[int]  (기존 인터페이스 유지)
    enable_trim=True:
        List[Tuple[str,int]]  (명시 이벤트)
    """
    rng = random.Random(rng_seed)

    # ---- 파라미터 방어적 클램프 ----
    update_ratio = max(0.0, min(float(update_ratio), 1.0))
    hot_ratio = max(0.0, min(float(hot_ratio), 1.0))
    hot_weight = max(0.0, min(float(hot_weight), 1.0))
    trim_ratio = max(0.0, min(float(trim_ratio), 1.0))

    # LPN 공간이 0이면 의미 있는 workload를 만들 수 없다.
    # (실험 입력 자체가 잘못된 것에 가깝다)
    if ssd_total_pages <= 0:
        if not enable_trim:
            return [0] * max(0, int(n_ops))
        return [("write", 0)] * max(0, int(n_ops))

    # hotset 경계 (oracle 스타일: lpn < hot_cut → hot)
    hot_cut = max(1, int(ssd_total_pages * hot_ratio))

    # live LPN 컨테이너(빠른 샘플링/삭제용)
    live_hot = _IndexList()
    live_cold = _IndexList()

    def _is_hot(lpn: int) -> bool:
        return lpn < hot_cut

    def _add_live(lpn: int) -> None:
        (live_hot if _is_hot(lpn) else live_cold).add(lpn)

    def _remove_live(lpn: int) -> None:
        (live_hot if _is_hot(lpn) else live_cold).remove(lpn)

    def _have_live() -> bool:
        return (len(live_hot) + len(live_cold)) > 0

    def _pick_update_lpn() -> int:
        """
        update 대상 LPN을 선택한다.

        규칙:
        - hot_weight 확률로 hot 풀을 우선 선택
        - 해당 풀이 비어 있으면 다른 풀로 fallback
        - 둘 다 비어 있으면 0 반환(이 경우는 거의 new_write로 흡수되는 편)
        """
        if rng.random() < hot_weight:
            if len(live_hot) > 0:
                return live_hot.choice(rng)
            if len(live_cold) > 0:
                return live_cold.choice(rng)
        else:
            if len(live_cold) > 0:
                return live_cold.choice(rng)
            if len(live_hot) > 0:
                return live_hot.choice(rng)
        return 0

    next_lpn = 0

    # --------------------------------------------------------
    # enable_trim=False: 기존 반환 형식 유지 (List[int])
    # --------------------------------------------------------
    if not enable_trim:
        ops: List[int] = []

        for _ in range(int(n_ops)):
            # new_write = "새로운 LPN에 처음 쓰기"
            # - 시스템이 비어있으면 강제로 new_write
            # - 아니면 update_ratio를 기준으로 new vs update 결정
            new_write = (not _have_live()) or (rng.random() >= update_ratio)

            if new_write and next_lpn < ssd_total_pages:
                # 신규 할당(write)
                lpn = next_lpn
                next_lpn += 1
                _add_live(lpn)
            else:
                # update 또는 이미 용량을 다 채운 상태
                if _have_live():
                    lpn = _pick_update_lpn()
                else:
                    # 이론상 거의 안 나와야 하지만, 안전 fallback
                    lpn = min(next_lpn, ssd_total_pages - 1)
                    if next_lpn < ssd_total_pages:
                        next_lpn += 1
                        _add_live(lpn)

            ops.append(lpn)

        return ops

    # --------------------------------------------------------
    # enable_trim=True: 이벤트 명시 (List[("write"/"trim", lpn)])
    # --------------------------------------------------------
    ops2: List[Tuple[str, int]] = []

    for _ in range(int(n_ops)):
        # 1) TRIM 이벤트 시도
        # - live가 존재해야 trim 대상이 있다.
        if trim_ratio > 0.0 and _have_live() and (rng.random() < trim_ratio):
            # trim도 hot_weight로 풀 선택 편향을 준다.
            if rng.random() < hot_weight and len(live_hot) > 0:
                lpn = live_hot.choice(rng)
            elif len(live_cold) > 0:
                lpn = live_cold.choice(rng)
            elif len(live_hot) > 0:
                lpn = live_hot.choice(rng)
            else:
                lpn = 0

            ops2.append(("trim", lpn))
            _remove_live(lpn)
            continue

        # 2) WRITE 이벤트 (신규/업데이트)
        new_write = (not _have_live()) or (rng.random() >= update_ratio)

        if new_write and next_lpn < ssd_total_pages:
            lpn = next_lpn
            next_lpn += 1
            _add_live(lpn)
        else:
            if _have_live():
                lpn = _pick_update_lpn()
            else:
                lpn = min(next_lpn, ssd_total_pages - 1)
                if next_lpn < ssd_total_pages:
                    next_lpn += 1
                    _add_live(lpn)

        ops2.append(("write", lpn))

    return ops2


# ------------------------------------------------------------
# 멀티 페이즈 (그대로 유지, 내부적으로 make_workload 호출)
# ------------------------------------------------------------

def make_phased_workload(phases, ssd_total_pages: int, base_seed: int = 42) -> Workload:
    """
    여러 phase를 이어 붙인 워크로드 생성기.

    phases 형식 예:
      [
        {"n_ops":..., "update_ratio":..., "hot_ratio":..., "hot_weight":...,
         "trim_ratio":..., "enable_trim":..., "seed":...},
        ...
      ]

    반환 정책:
    - 모든 phase가 enable_trim=False(또는 미지정) → List[int]
    - 하나라도 enable_trim=True → List[("write"/"trim", lpn)]
      (혼합을 피하려고, 정수 항목은 ("write", lpn)으로 승격한다)
    """
    out: list = []
    made_tuple = False

    for i, p in enumerate(phases):
        chunk = make_workload(
            n_ops=p["n_ops"],
            update_ratio=p.get("update_ratio", 0.8),
            ssd_total_pages=ssd_total_pages,
            rng_seed=p.get("seed", base_seed + i),
            hot_ratio=p.get("hot_ratio", 0.2),
            hot_weight=p.get("hot_weight", 0.85),
            enable_trim=p.get("enable_trim", False),
            trim_ratio=p.get("trim_ratio", 0.0),
        )

        if chunk and isinstance(chunk[0], tuple):
            made_tuple = True

        out.extend(chunk)

    # 혼합을 피하기 위해, 한 번이라도 튜플이 나오면 전체를 튜플로 통일
    if made_tuple:
        norm: List[Tuple[str, int]] = []
        for x in out:
            if isinstance(x, tuple):
                norm.append(x)
            else:
                norm.append(("write", int(x)))
        return norm

    return out


# ------------------------------------------------------------
# 보조 유틸
# ------------------------------------------------------------

def only_writes(seq: Workload) -> List[int]:
    """
    튜플 워크로드에서 write만 뽑아 정수 LPN 리스트로 변환.
    - ("trim", lpn)은 버린다.
    """
    out: List[int] = []
    for x in seq:
        if isinstance(x, tuple):
            op, lpn = x
            if op == "write":
                out.append(int(lpn))
        else:
            out.append(int(x))
    return out


def trim_count(seq: Workload) -> int:
    # 생성된 시퀀스에서 TRIM 개수 확인.
    c = 0
    for x in seq:
        if isinstance(x, tuple) and x[0] == "trim":
            c += 1
    return c


def make_rocksdb_like_phases(user_pages: int, base_seed: int = 500) -> list:
    """
    아주 단순화된 LSM/rocksdb 느낌의 phase 시나리오 생성기.

    의도:
    - 초기 bulk-load 구간: update_ratio 낮음(대부분 신규 write)
    - 이후 burst 구간 반복: update-heavy 구간(덮어쓰기) + 그보다 덜한 구간
      → “한동안 업데이트가 몰렸다가, 좀 풀리는” 패턴을 흉내낸다.

    NOTE:
    - 이건 정확한 RocksDB 모델이 아니라, 실험용 자극(패턴)을 만드는 헬퍼다.
    """
    bulk = int(user_pages * 0.8)
    burst = int(user_pages * 0.2)

    phases = [{
        "n_ops": bulk,
        "update_ratio": 0.2,
        "hot_ratio": 0.2,
        "hot_weight": 0.85,
        "seed": base_seed,
    }]

    for i in range(3):
        phases.append({
            "n_ops": burst,
            "update_ratio": 0.9,
            "hot_ratio": 0.2,
            "hot_weight": 0.9,
            "seed": base_seed + i + 1,
        })
        phases.append({
            "n_ops": burst,
            "update_ratio": 0.7,
            "hot_ratio": 0.2,
            "hot_weight": 0.85,
            "seed": base_seed + i + 10,
        })

    return phases