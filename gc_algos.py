from __future__ import annotations

"""
gc_algos.py

GC victim selection policies (victim picker) 모음입니다.

이 프로젝트에서 GC 정책은 아래 형태의 함수로 정의됩니다.

    policy(blocks) -> victim_index | None

- blocks: SSD 내부의 블록 객체 리스트(또는 iterable)
- victim_index: GC 대상으로 선택된 블록의 인덱스
- None: 선택할 블록이 없는 경우(예: 전부 free block)

이 파일의 목적
--------------
1) 정책 구현의 분리(Separation of concerns)
   - simulator/ssd 로직과 어떤 블록을 지울지 고르는 로직을 분리합니다.
2) 비교 실험의 단순화(Comparative experiments)
   - greedy / cost-benefit / bsgc / cota / atcb / re50315 등 정책을 동일 인터페이스로 제공합니다.
3) 재현 가능한 점수 정의(Scoring transparency)
   - 각 정책이 어떤 feature(무효비율, hotness, age, wear)를 보고 점수를 만드는지 주석으로 명확히 합니다.

입력 블록 객체가 제공해야 하는 속성(Contract)
--------------------------------------------
정책들은 블록 객체에서 아래 속성들을 getattr로 읽습니다. (없으면 기본값 사용)

필수에 가까운 것(없으면 정책이 의미가 약해짐)
- valid_count   : 유효 페이지 수
- invalid_count : 무효 페이지 수
- erase_count   : erase 횟수(wear 지표)

COTA/ATCB/RE50315에서 해석에 영향을 주는 것
- inv_ewma          : hotness(“뜨거움”) 추정치. 없으면 0.0(차가움)으로 간주
- last_prog_step    : 마지막 program 시각(step)
- last_invalid_step : 마지막 invalidation 시각(step)
- age_counter       : (특허 단순화 정책) block age counter

주의(Assumptions / Pitfalls)
----------------------------
- 모든 정책은 “현재 상태의 block list”를 입력으로 받으며,
  이 block list가 simulator/ssd 내부에서 최신 상태로 갱신된다는 전제가 필요합니다.
- hotness(inv_ewma)가 0~1 범위로 정규화되어 있다고 가정하는 정책이 있습니다.
  만약 범위가 다르면 가중치(beta/eta) 해석이 달라질 수 있습니다.
- wear_norm, age_norm는 실험 내 상대 정규화입니다(최대값/범위 기반).
  따라서 실험 조건(디바이스 크기/ops/워크로드)에 따라 스케일이 달라질 수 있습니다.

확장 포인트
-----------
- 새로운 정책을 추가할 때는:
  1) policy(blocks) -> index 형태를 유지하고,
  2) get_gc_policy에 이름 매핑을 추가하면 됩니다.
"""


# ------------------------------------------------------------
# 공통 헬퍼: 블록 상태(feature) 읽기
# ------------------------------------------------------------

def _block_used(b) -> int:
    """
    블록에 할당된(사용 중인) 페이지 수 = valid + invalid

    - used == 0이면 free block으로 간주할 수 있습니다.
    - policy들은 보통 free block은 victim 후보에서 제외합니다.
    """
    return int(getattr(b, "valid_count", 0)) + int(getattr(b, "invalid_count", 0))


def _hotness(b) -> float:
    """
    블록의 "뜨거움(hotness)" 추정치.

    - 프로젝트에서는 inv_ewma 같은 지표로 최근 invalidation 빈도를 추정하는 경우가 많습니다.
    - 속성이 없으면 0.0으로 처리하여 차가운 블록으로 간주합니다.

    NOTE:
    - 이 값이 0~1 범위로 정규화되어 있다는 가정이 있으면,
      (1 - hotness)가 coldness처럼 작동합니다.
    """
    return float(getattr(b, "inv_ewma", 0.0))


def _last_activity(b) -> int:
    """
    블록의 마지막 활동 시각(step) proxy.

    - last_prog_step: 마지막 program이 발생한 step
    - last_invalid_step: 마지막 invalidation이 발생한 step

    둘 중 큰 값을 최근 활동으로 보고 사용합니다.
    (block이 최근에 쓰였거나 무효화되었다면 뜨거운/젊은 블록일 가능성이 큼)
    """
    lp = int(getattr(b, "last_prog_step", 0))
    li = int(getattr(b, "last_invalid_step", 0))
    return max(lp, li)


def _wear(b) -> int:
    """
    블록 wear(마모) proxy: erase_count.

    - erase_count는 블록이 얼마나 많이 지워졌는지의 누적 값입니다.
    - 정책에서 wear_norm(정규화 마모)을 계산해 wear-leveling 관점의 bias를 줄 수 있습니다.
    """
    return int(getattr(b, "erase_count", 0))


# ------------------------------------------------------------
# 1) 기본 정책들 (baseline)
# ------------------------------------------------------------

def greedy_policy(blocks):
    """
    Greedy 정책: invalid 페이지가 가장 많은 블록을 victim으로 선택.

    직관
    ----
    - invalid가 많으면 옮길 valid가 적다는 의미이므로,
      GC 비용(copy)이 상대적으로 적고 free page를 많이 회수할 수 있습니다.

    계약
    ----
    - blocks 내 각 블록은 valid_count/invalid_count를 제공하는 것이 바람직합니다.
    - used == 0 (free block)은 후보에서 제외합니다.

    반환
    ----
    - victim 블록 인덱스 또는 None
    """
    best_idx, best_invalid = None, -1
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue
        inv = int(getattr(b, "invalid_count", 0))
        if inv > best_invalid:
            best_invalid = inv
            best_idx = i
    return best_idx


def cb_policy(blocks):
    """
    Cost-Benefit 정책의 단순 근사(Simplified CB).

    score = (1 - u) * age_proxy
    - u = valid_ratio = valid_count / used
    - (1 - u) = invalid_ratio
    - age_proxy = 1 / (1 + erase_count)  (단순화된 젊음/수명 여유 가중)

    해석
    ----
    - invalid_ratio가 큰 블록(회수 효율↑)을 선호하면서,
    - erase_count가 낮은 블록(마모 여유↑)에 추가 점수를 줍니다.

    주의
    ----
    - 원래 Cost-Benefit은 age(시간)를 다루는 형태가 많지만,
      여기서는 wear를 age_proxy로 사용한 *근사 버전*입니다.
    """
    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue
        u = (getattr(b, "valid_count", 0) / used)
        age_proxy = 1.0 / (1.0 + _wear(b))
        s = (1.0 - u) * age_proxy
        if s > best_score:
            best_score, best_idx = s, i
    return best_idx


def bsgc_policy(blocks, alpha=0.7, beta=0.3):
    """
    BSGC(균형형) 정책: invalid 회수 효율 + wear-leveling을 함께 고려.

    score = alpha * invalid_ratio + beta * (1 - wear_norm)

    - invalid_ratio: invalid_count / used
    - wear_norm: erase_count / max_erase (실험 내 상대 정규화)

    해석
    ----
    - invalid_ratio가 높을수록 점수↑ (GC 효율↑)
    - wear_norm이 낮을수록 (1 - wear_norm)↑ => 마모가 덜 된 블록을 더 지우게 되는 경향
      (wear를 평준화하려면 가중치 튜닝이 중요)

    주의
    ----
    - wear_norm은 현재 block set 내 최대 erase_count로 정규화합니다.
      실험 조건에 따라 스케일이 달라질 수 있습니다.
    """
    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue
        invalid_ratio = getattr(b, "invalid_count", 0) / used
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0
        s = alpha * invalid_ratio + beta * (1.0 - wear_norm)
        if s > best_score:
            best_score, best_idx = s, i
    return best_idx


def age_stale_policy(blocks, K=50):
    """
    'age & staleness' 기반 정책(특허 아이디어 단순화 버전).

    rank(B) = staleness(B) * (age(B) + K)

    - staleness: invalid 페이지 수를 staleness의 proxy로 사용(최소 1)
    - age: block.age_counter (마지막 erase 이후 상대적 나이로 가정)
    - K: 상수 오프셋(완전히 새 블록도 최소 점수를 갖게 하는 형태)

    목적/해석
    --------
    - invalid가 많고(age도 큰) 오래되고 무효가 많은 블록을 우선 victim으로 선택합니다.

    주의
    ----
    - 이 정책은 block 객체에 age_counter가 있어야 의미가 있습니다.
      (없으면 기본값 0으로만 동작)
    """
    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        staleness = getattr(b, "invalid_count", 0)
        if staleness < 1:
            staleness = 1

        age = getattr(b, "age_counter", 0)
        score = staleness * (age + K)

        if score > best_score:
            best_score, best_idx = score, i

    return best_idx


# ------------------------------------------------------------
# 2) COTA (Cost-over-Temperature-and-Age)
# ------------------------------------------------------------

def cota_policy(blocks, alpha=0.55, beta=0.25, gamma=0.15, delta=0.05):
    """
    COTA 정책: invalid 회수 효율 + coldness + age + wear를 함께 반영한 점수식.

    score = α*invalid_ratio
          + β*(1 - hotness)
          + γ*age_norm
          + δ*(1 - wear_norm)

    feature 정의(이 구현에서)
    ------------------------
    - invalid_ratio = invalid_count / used
    - hotness       = inv_ewma (없으면 0.0으로 간주)
    - age_norm      = (last_max - last_activity) / (last_max - last_min)
                      => 최근 활동이 오래전일수록 age_norm↑ (더 “오래된” 블록)
    - wear_norm     = erase_count / max_erase (실험 내 상대 정규화)

    해석
    ----
    - invalid_ratio↑: GC 비용↓/회수량↑
    - (1-hotness)↑ : “차가운(cold)” 블록 선호 (hot data는 자주 바뀌니 건드리기 싫다)
    - age_norm↑    : 오래된 블록 선호 (최근 활동 없는 블록)
    - (1-wear_norm)↑: 마모가 덜 된 블록 선호(가중치 δ에 의해 정도 조절)

    주의
    ----
    - hotness(inv_ewma)가 0~1로 정규화되어 있다는 전제에서 (1-hotness)가 의미를 갖습니다.
    - age_norm은 현재 blocks의 last_activity 분포로 정규화하므로, 실험 규모/조건에 따라 달라집니다.
    """
    lasts = [_last_activity(b) for b in blocks]
    if not lasts:
        return None

    last_max, last_min = max(lasts), min(lasts)
    age_den = max(1, last_max - last_min)

    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        invalid_ratio = getattr(b, "invalid_count", 0) / used
        hotness = _hotness(b)

        age_norm = (last_max - _last_activity(b)) / age_den
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0

        s = (
            alpha * invalid_ratio
            + beta * (1.0 - hotness)
            + gamma * age_norm
            + delta * (1.0 - wear_norm)
        )
        if s > best_score:
            best_score, best_idx = s, i

    return best_idx


# ------------------------------------------------------------
# 3) ATCB / RE50315 (경량 비교용)
# ------------------------------------------------------------

def atcb_policy(blocks, alpha=0.5, beta=0.3, gamma=0.1, eta=0.1, now_step=None):
    """
    ATCB 정책(경량 비교용): invalid, wear, age, coldness를 섞은 점수식.

    score = α*(1-u) + β*(1 - wear_norm) + γ*age_norm + η*(1 - hotness)

    - u        = valid_ratio = valid_count / used
    - (1-u)    = invalid_ratio
    - wear_norm: erase_count / max_erase
    - age_norm : (now_step - last_activity) / (now_step - last_min)
    - hotness  : inv_ewma

    now_step
    --------
    - 시뮬레이터 내부 step(시간)을 정책에 전달하기 위한 인자입니다.
    - 외부에서 주입하는 형태가 많으므로(run_sim/experiments에서 wrapper로 전달),
      여기서는 None이면 blocks의 마지막 활동 기반으로 추정합니다.
    """
    used_list = [_block_used(b) for b in blocks]
    if not any(used_list):
        return None

    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    lasts = [_last_activity(b) for b in blocks]
    last_min = min(lasts) if lasts else 0
    if now_step is None:
        now_step = max(lasts) if lasts else 0
    age_den = max(1, now_step - last_min)

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        u = getattr(b, "valid_count", 0) / used
        inv = 1.0 - u

        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0
        age_norm = (now_step - _last_activity(b)) / age_den

        hot = _hotness(b)

        s = alpha * inv + beta * (1.0 - wear_norm) + gamma * age_norm + eta * (1.0 - hot)
        if s > best_score:
            best_score, best_idx = s, i

    return best_idx


def re50315_policy(blocks, K=1.0, now_step=None):
    """
    RE50315 정책(경량 비교용): invalid + age + wear 기반.

    score = (1-u) + K*age_norm + (1 - wear_norm)

    - (1-u)      : invalid_ratio
    - age_norm   : (now_step - last_activity) / (now_step - last_min)
    - wear_norm  : erase_count / max_erase
    - K          : age 가중치

    주의
    ----
    - 이 구현은 특허 레퍼런스 아이디어를 실험용으로 단순화한 버전입니다.
    - now_step 주입 방식은 ATCB와 동일합니다.
    """
    used_list = [_block_used(b) for b in blocks]
    if not any(used_list):
        return None

    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    lasts = [_last_activity(b) for b in blocks]
    last_min = min(lasts) if lasts else 0
    if now_step is None:
        now_step = max(lasts) if lasts else 0
    age_den = max(1, now_step - last_min)

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        u = getattr(b, "valid_count", 0) / used
        inv = 1.0 - u

        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0
        age_norm = (now_step - _last_activity(b)) / age_den

        s = inv + K * age_norm + (1.0 - wear_norm)
        if s > best_score:
            best_score, best_idx = s, i

    return best_idx


# ------------------------------------------------------------
# 4) 정책 팩토리(이름 -> 함수)
# ------------------------------------------------------------

def get_gc_policy(name: str):
    """
    문자열 이름으로 정책 함수를 얻는 팩토리.

    목적
    ----
    - CLI/실험 설정에서 policy 이름만 받아도,
      대응되는 함수를 쉽게 주입할 수 있도록 합니다.

    NOTE
    ----
    - ATCB/RE50315는 now_step/하이퍼파라미터 주입이 필요할 수 있으므로,
      기본 반환은 기본 파라미터 버전입니다.
      실험 스크립트(run_sim/experiments)에서 wrapper로 확장하는 방식을 권장합니다.
    """
    n = (name or "").lower()

    if n == "greedy":
        return greedy_policy
    if n in ("cb", "cost_benefit"):
        return cb_policy
    if n == "bsgc":
        return bsgc_policy
    if n == "cota":
        return cota_policy
    if n == "age_stale":
        return age_stale_policy

    if n in ("atcb", "atcb_policy"):
        return lambda blocks: atcb_policy(blocks)

    if n in ("re50315", "re50315_policy"):
        return lambda blocks: re50315_policy(blocks)

    raise ValueError(f"unknown policy: {name}")