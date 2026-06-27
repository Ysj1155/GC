from __future__ import annotations

"""
config.py

SSD GC 시뮬레이터 프로젝트의 실험 설정(Experiment Config) 모듈입니다.

이 파일은 시뮬레이터의 동작을 결정하는 입력 파라미터들을 한 곳에 모아,
- 같은 설정이면 같은 조건의 실험을 재현할 수 있고(재현성),
- 실행 전에 잘못된 설정을 조기에 잡아내며(fail fast),
- 파생 값(총 페이지 수, 임계치 절대값 등)을 일관되게 계산하도록 합니다.

계약(Contract)
--------------
- 시뮬레이션을 시작하기 전에 반드시 `SimConfig.prepare()`를 호출하는 것을 권장합니다.
  prepare()는
  1) io_profile 프리셋을 적용하고,
  2) validate()로 값 범위를 검증합니다.

재현성(Reproducibility)에서 중요한 노브(knob)
--------------------------------------------
- `rng_seed`: 워크로드/시뮬레이터가 난수를 사용한다면 결과 재현의 핵심입니다.
- Geometry(`num_blocks`, `pages_per_block`) + 용량 비율(`user_capacity_ratio`):
  총 용량과 오버프로비저닝(OP) 유사 효과를 결정합니다.
- `gc_free_block_threshold`: GC가 얼마나 민감하게(빨리/자주) 트리거되는지를 결정합니다.

가정/단순화(Assumptions / Simplifications)
-----------------------------------------
- 지연시간(latency)은 마이크로초(μs) 단위의 “상수 비용”으로 모델링합니다.
  실제 SSD는 더 복잡하지만, 비교 실험을 위한 통제된 추상화로 봅니다.
- `migrate_read_prog_us`는 GC가 유효 페이지를 옮길 때의 비용을
  (read + program) 정도로 근사한 값입니다. (실제는 더 복잡할 수 있음)

자주 생기는 실수(Common pitfalls)
---------------------------------
- 임계치의 “비율”과 “절대 개수”를 혼동하기:
  * `gc_free_block_threshold`는 비율(0~1)
  * `free_block_threshold_abs`는 이를 블록 개수로 환산한 파생값
- `io_profile` 오타:
  알 수 없는 io_profile이면 아무 변화 없이 기존 값이 유지됩니다(조용히 무시).
  오타가 있으면 기대와 다르게 default처럼 보일 수 있으니 주의하세요.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class SimConfig:
    """
    실험 설정 컨테이너.

    이 클래스의 필드들은 시뮬레이션에 들어가는 “입력”입니다.
    크게 3 묶음으로 구성됩니다.

    1) Geometry(장치 형태): 블록/페이지 구성, 유저에게 보이는 용량 비율
    2) GC 트리거 + RNG: 언제 GC를 돌릴지, 랜덤이 있다면 어떻게 재현할지
    3) Latency 모델: 시간/지연 계산에 쓰는 비용 가정(μs)

    중요 포인트
    ----------
    - `validate()`는 값 범위를 강제합니다.
    - `apply_io_profile()`는 프리셋으로 latency 값을 덮어쓸 수 있습니다.
    - `prepare()`는 프리셋 적용 + 검증을 한 번에 수행하는 “권장 진입점”입니다.
    """

    # ---------------------------------------------------------------------
    # 1) Geometry (장치 형태)
    # ---------------------------------------------------------------------
    # 장치 모델의 erase block 개수
    # - 총 용량에 직접 영향
    # - 보통 장치가 작을수록 GC가 더 빨리/자주 발생하는 경향이 있습니다.
    num_blocks: int = 256

    # 블록 당 페이지 수(장치의 세부 그레뉼러리티)
    pages_per_block: int = 64

    # 유저에게 보이는 논리 용량 비율 (0~1]
    # - (1 - ratio)는 오버프로비저닝(OP)과 유사한 효과
    # - ratio가 낮아지면 여유 공간이 늘어 GC 부담이 줄어드는 경향
    user_capacity_ratio: float = 0.9

    # ---------------------------------------------------------------------
    # 2) GC 트리거 / 난수
    # ---------------------------------------------------------------------
    # free block 비율 임계치 (0~1)
    # 예: 0.12면 free block이 대략 num_blocks의 12% 아래로 내려가면 GC 트리거
    gc_free_block_threshold: float = 0.12

    # 난수 시드(워크로드/시뮬레이터가 랜덤을 쓴다면 재현성 핵심)
    rng_seed: int = 42

    # ---------------------------------------------------------------------
    # 3) Latency profile (마이크로초, μs)
    # ---------------------------------------------------------------------
    # 아래 값들은 “상수 비용” 기반의 latency 모델입니다.
    # 실제 SSD 지연은 더 복잡하지만, 비교 실험을 위한 베이스라인으로 사용합니다.
    host_prog_us: int = 100         # page program 지연(μs)
    host_read_us: int = 50          # page read 지연(μs)
    erase_us: int = 1500            # block erase 지연(μs)

    # GC 이동(유효 페이지 migration) 비용 근사값
    # 기본은 host_read_us + host_prog_us에 맞춰 쓰는 것이 일반적입니다.
    migrate_read_prog_us: int = 150

    # latency 프리셋 이름
    # - 지정하면 `apply_io_profile()`에서 latency 필드들을 덮어씁니다.
    # - 알려진 프리셋: default | fast | slow | qos_lowlat
    io_profile: str = "default"

    # 내부 플래그: validate()가 성공적으로 끝났는지 표시
    # 실험 정의의 일부가 아니라 “내부 상태”로 취급합니다.
    _validated: bool = field(default=False, init=False, repr=False)

    # ---------------------------------------------------------------------
    # 파생값(입력으로부터 계산되는 값)
    # ---------------------------------------------------------------------

    @property
    def total_pages(self) -> int:
        """
        장치의 총 물리 페이지 수.

        계산식:
            total_pages = num_blocks * pages_per_block
        """
        return max(0, int(self.num_blocks) * int(self.pages_per_block))

    @property
    def user_total_pages(self) -> int:
        """
        유저에게 보이는 논리 페이지 수.

        계산식:
            user_total_pages = total_pages * clamp(user_capacity_ratio, 0..1)

        NOTE:
        - validate()에서 이미 user_capacity_ratio를 (0,1]로 강제하지만,
          혹시 외부에서 잘못 넣었을 때를 대비해 여기서도 클램프합니다(방어적).
        """
        r = min(max(float(self.user_capacity_ratio), 0.0), 1.0)
        return int(self.total_pages * r)

    @property
    def free_block_threshold_abs(self) -> int:
        """
        free block 임계치를 블록 개수(절대값)로 환산한 값.

        계산식:
            abs_threshold = round(num_blocks * clamp(gc_free_block_threshold, 0..1))

        NOTE:
        - round를 사용하므로 경계에서 ±1 블록 차이가 날 수 있습니다.
          ‘엄밀한 의미’를 원하면 floor/ceil로 고정하는 방법도 있습니다.
        """
        r = min(max(float(self.gc_free_block_threshold), 0.0), 1.0)
        return int(round(self.num_blocks * r))

    # ---------------------------------------------------------------------
    # 검증 / 준비
    # ---------------------------------------------------------------------

    def validate(self) -> None:
        """
        설정 값의 범위와 기본 일관성을 검증합니다.

        Raises
        ------
        ValueError:
            값이 허용 범위를 벗어나면 예외를 발생시킵니다.

        Guarantees
        ----------
        - 성공하면 `_validated=True`가 됩니다.
        - ratio가 기대 범위 안에 들어옵니다.
        - geometry/latency가 양수임을 보장합니다.
        """
        # geometry
        if self.num_blocks <= 0 or self.pages_per_block <= 0:
            raise ValueError("num_blocks/pages_per_block 는 양수여야 합니다")

        # ratio 범위
        if not (0.0 < self.user_capacity_ratio <= 1.0):
            raise ValueError("user_capacity_ratio 는 (0,1] 범위여야 합니다")
        if not (0.0 <= self.gc_free_block_threshold < 1.0):
            raise ValueError("gc_free_block_threshold 는 [0,1) 범위여야 합니다")

        # latency 양수 체크
        for k in ("host_prog_us", "host_read_us", "erase_us", "migrate_read_prog_us"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} 는 양수여야 합니다")

        self._validated = True

    def apply_io_profile(self) -> None:
        """
        io_profile 프리셋으로 latency 필드들을 덮어씁니다.

        목적
        ----
        - 여러 실험에서 시간 비용 가정을 표준화하고 싶을 때 유용합니다.
          (예: fast/slow flash 환경을 비교하는 느낌)

        동작(Behavior)
        ------------
        - 알려진 io_profile이면 host_read_us/host_prog_us/erase_us를 설정하고,
          migrate_read_prog_us를 (read+prog)로 갱신합니다.
        - 알 수 없는 io_profile이면 아무것도 하지 않습니다(기존 값 유지).
          => 오타가 조용히 무시될 수 있으므로 주의하세요.
        """
        p = (self.io_profile or "default").lower()

        if p == "default":
            self.host_read_us = 50
            self.host_prog_us = 100
            self.erase_us = 1500
            self.migrate_read_prog_us = self.host_read_us + self.host_prog_us

        elif p == "fast":
            self.host_read_us = 30
            self.host_prog_us = 70
            self.erase_us = 1000
            self.migrate_read_prog_us = self.host_read_us + self.host_prog_us

        elif p == "slow":
            self.host_read_us = 80
            self.host_prog_us = 160
            self.erase_us = 2500
            self.migrate_read_prog_us = self.host_read_us + self.host_prog_us

        elif p == "qos_lowlat":
            # 읽기 지연을 더 줄이는 프로파일(erase는 유지)
            self.host_read_us = 25
            self.host_prog_us = 90
            self.erase_us = 1500
            self.migrate_read_prog_us = self.host_read_us + self.host_prog_us

        else:
            # 알 수 없는 프로파일: 기존 값 유지(무시)
            # 엄격 모드를 원한다면 여기서 경고 출력/예외 처리로 바꿀 수 있습니다.
            pass

    # ---------------------------------------------------------------------
    # 직렬화(실험 메타데이터 저장용)
    # ---------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """
        설정을 dict로 변환합니다(로그/메타데이터 저장용).

        NOTE:
        - dataclass의 asdict는 내부 필드까지 포함하므로,
          `_validated` 같은 내부 상태는 제거하여 실험 정의만 남깁니다.
        """
        d = asdict(self)
        d.pop("_validated", None)  # 내부 상태 제거
        return d

    def prepare(self) -> None:
        """
        실행 전 권장 초기화 함수.

        이 함수 하나만 호출하면:
        1) io_profile이 적용되고,
        2) validate()로 설정이 검증됩니다.

        즉, 시뮬레이터는 prepare() 이후의 config를 정상/일관된 설정으로 가정할 수 있습니다.
        """
        self.apply_io_profile()
        self.validate()