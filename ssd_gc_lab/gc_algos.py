from __future__ import annotations

"""
gc_algos.py

GC victim selection policies (victim picker) 모음입니다.

이 프로젝트에서 GC 정책은 아래 형태의 함수로 정의됩니다.

    policy(blocks) -> victim_index | None

- blocks: SSD 내부의 블록 객체 리스트(또는 iterable)
- victim_index: GC 대상으로 선택된 블록의 인덱스
- None: 선택할 블록이 없는 경우(예: 전부 free block)

이 파일을 읽는 가장 쉬운 순서
----------------------------
1) 먼저 _block_used(), _hotness(), _last_activity(), _wear()를 봅니다.
   - 정책들이 공통으로 사용하는 "블록 특징(feature)"을 읽는 함수입니다.
2) greedy_policy()를 봅니다.
   - 가장 단순한 기준입니다. invalid가 가장 많은 블록을 지웁니다.
3) cb_policy(), bsgc_policy()를 봅니다.
   - invalid만 보지 않고 wear(마모)까지 조금씩 섞습니다.
4) cota_policy()를 봅니다.
   - 이 프로젝트의 핵심 아이디어입니다.
   - invalid_ratio, coldness, age, wear를 하나의 score로 합칩니다.
5) get_gc_policy()를 봅니다.
   - CLI에서 받은 문자열 이름을 실제 정책 함수로 바꿔주는 연결부입니다.

SSD GC를 top-down으로 보면
--------------------------
SSD는 page 단위로 program(write)할 수 있지만, 지울 때는 block 단위로 erase해야 합니다.
그래서 어떤 block을 erase할지 고르는 선택이 GC 정책의 핵심입니다.

GC victim 선택에서 보통 보는 질문은 네 가지입니다.
- 이 블록을 지우면 얼마나 많이 회수되는가?      -> invalid_ratio
- 이 블록 안의 데이터가 자주 바뀌는가?          -> hotness / coldness
- 이 블록이 최근에 사용되었는가, 오래 방치됐는가? -> age
- 이 블록은 이미 많이 지워져서 낡았는가?        -> wear

이 파일의 모든 정책은 위 질문들에 서로 다른 가중치를 주는 방식이라고 보면 됩니다.

용어를 아주 쉽게 정리하면
------------------------
- valid page   : 현재 LPN mapping이 가리키는 살아있는 데이터
- invalid page : overwrite/TRIM 때문에 더 이상 필요 없는 쓰레기 데이터
- free page    : 아직 program되지 않은 빈 공간
- used page    : valid + invalid. 즉, 한 번이라도 사용된 페이지
- victim block : GC가 이번에 지우기로 고른 블록
- score        : "이 블록을 victim으로 고르면 얼마나 좋은가?"를 숫자로 표현한 값

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

    왜 free block을 victim 후보에서 빼는가?
    -------------------------------
    free block은 이미 비어 있으므로 erase할 필요가 없습니다.
    GC는 "사용된 적이 있는 블록"을 지워서 다시 free block으로 만드는 작업입니다.
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

    SSD 직관
    --------
    hot한 데이터는 가까운 미래에 다시 overwrite될 가능성이 큽니다.
    이런 데이터를 GC가 굳이 옮기면, 곧 또 무효화될 데이터를 복사한 셈이 되어
    write amplification이 나빠질 수 있습니다.
    """
    return float(getattr(b, "inv_ewma", 0.0))


def _last_activity(b) -> int:
    """
    블록의 마지막 활동 시각(step) proxy.

    - last_prog_step: 마지막 program이 발생한 step
    - last_invalid_step: 마지막 invalidation이 발생한 step

    둘 중 큰 값을 최근 활동으로 보고 사용합니다.
    (block이 최근에 쓰였거나 무효화되었다면 뜨거운/젊은 블록일 가능성이 큼)

    age 계산에서의 의미
    ------------------
    last_activity가 작다  -> 오래전에 마지막으로 건드린 블록 -> age가 큼
    last_activity가 크다  -> 최근에 건드린 블록             -> age가 작음
    """
    lp = int(getattr(b, "last_prog_step", 0))
    li = int(getattr(b, "last_invalid_step", 0))
    return max(lp, li)


def _wear(b) -> int:
    """
    블록 wear(마모) proxy: erase_count.

    - erase_count는 블록이 얼마나 많이 지워졌는지의 누적 값입니다.
    - 정책에서 wear_norm(정규화 마모)을 계산해 wear-leveling 관점의 bias를 줄 수 있습니다.

    SSD 직관
    --------
    NAND block은 erase 가능 횟수가 제한되어 있습니다.
    특정 block만 계속 지우면 그 block이 먼저 닳기 때문에,
    정책에 따라 wear를 함께 고려해 마모 균형을 맞추려 합니다.
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

    장단점
    ------
    - 장점: 이해하기 쉽고 GC 1회당 회수 효율이 좋습니다.
    - 단점: hot/cold, age, wear를 전혀 보지 않아서 장기적인 마모 균형은 약할 수 있습니다.
    """
    best_idx, best_invalid = None, -1
    for i, b in enumerate(blocks):
        # valid도 invalid도 없는 완전 빈 블록은 이미 free block입니다.
        # 지울 이유가 없으므로 victim 후보에서 제외합니다.
        used = _block_used(b)
        if used == 0:
            continue

        # greedy는 이름 그대로 invalid_count 하나만 봅니다.
        # invalid가 많을수록 erase 후 회수되는 공간이 많다고 판단합니다.
        inv = int(getattr(b, "invalid_count", 0))

        # 지금까지 본 블록 중 invalid가 가장 많으면 best를 갱신합니다.
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

    이 구현의 핵심 감각
    ------------------
    invalid_ratio는 "지우면 얼마나 이득인가"이고,
    age_proxy는 "이 블록을 지워도 wear 관점에서 부담이 작은가"에 가깝습니다.
    """
    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        # valid ratio가 높으면 아직 살아있는 데이터가 많다는 뜻입니다.
        # GC가 이 블록을 지우려면 valid page를 다른 곳으로 많이 옮겨야 합니다.
        u = (getattr(b, "valid_count", 0) / used)

        # erase_count가 높을수록 age_proxy는 작아집니다.
        # 즉, 이미 많이 지운 블록은 또 지우지 않도록 점수를 낮춥니다.
        age_proxy = 1.0 / (1.0 + _wear(b))

        # (1-u)는 invalid_ratio입니다.
        # invalid가 많고, wear 부담이 낮을수록 score가 커집니다.
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

    alpha/beta 해석
    ---------------
    - alpha를 키우면 성능/회수 효율 쪽으로 더 기울어집니다.
    - beta를 키우면 wear를 덜 먹은 블록을 더 선호합니다.
    """
    # wear_norm을 만들려면 기준점이 필요합니다.
    # 여기서는 현재 블록들 중 가장 큰 erase_count를 1.0으로 놓고 상대 비교합니다.
    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        # invalid_ratio가 높으면 GC 회수 효율이 좋습니다.
        invalid_ratio = getattr(b, "invalid_count", 0) / used

        # max_erase가 0이면 아직 아무 블록도 erase되지 않은 상태입니다.
        # 이때는 모든 블록의 wear_norm을 0으로 봅니다.
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0

        # 두 feature를 선형 결합합니다.
        # 선형 결합은 단순하지만, 실험에서 가중치별 trade-off를 보기 좋습니다.
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

    공부 포인트
    -----------
    이 정책은 COTA처럼 여러 feature를 정규화해서 더하는 방식이 아니라,
    invalid page 수와 age를 곱해서 오래되고 stale한 블록을 강하게 밀어줍니다.
    """
    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        # staleness는 "이 블록에 죽은 데이터가 얼마나 쌓였는가"입니다.
        # invalid가 0이어도 최소 1을 줘서 age만으로도 비교가 가능하게 합니다.
        staleness = getattr(b, "invalid_count", 0)
        if staleness < 1:
            staleness = 1

        # age_counter는 SSD.erase_block() 쪽에서 다른 블록이 erase될 때 증가합니다.
        # 값이 클수록 오래 살아남은 블록으로 해석합니다.
        age = getattr(b, "age_counter", 0)

        # 오래됐고(stale/age), invalid도 많으면 score가 크게 튑니다.
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

    가중치 해석
    ----------
    - alpha: GC 회수 효율을 얼마나 중요하게 볼지
    - beta : cold data를 얼마나 선호할지
    - gamma: 오래 방치된 블록을 얼마나 선호할지
    - delta: wear가 낮은 블록을 얼마나 선호할지

    포트폴리오 관점에서 중요한 점
    ---------------------------
    COTA는 "성능(WAF)만 보는 정책"이 아니라,
    회수 효율, 데이터 온도, 시간, 마모를 한 점수식으로 묶어
    성능-마모 균형을 실험하려는 정책입니다.
    """
    # age_norm 계산을 위해 각 블록의 마지막 활동 시각을 모읍니다.
    # 활동 시각의 범위 안에서 "얼마나 오래됐는지"를 상대값으로 만들 것입니다.
    lasts = [_last_activity(b) for b in blocks]
    if not lasts:
        return None

    # 가장 최근 활동(last_max)과 가장 오래된 활동(last_min)을 기준으로
    # age_norm의 분모를 만듭니다.
    last_max, last_min = max(lasts), min(lasts)
    age_den = max(1, last_max - last_min)

    # wear도 현재 실험 안에서 상대 정규화합니다.
    # max_erase가 0이면 아직 wear 차이가 없다는 뜻입니다.
    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    best_idx, best_score = None, float("-inf")
    for i, b in enumerate(blocks):
        used = _block_used(b)
        if used == 0:
            continue

        # 1) invalid_ratio:
        #    invalid가 많을수록 이 블록을 지웠을 때 회수되는 공간이 큽니다.
        invalid_ratio = getattr(b, "invalid_count", 0) / used

        # 2) hotness:
        #    값이 높을수록 최근에 자주 바뀐 블록으로 봅니다.
        #    COTA는 hotness 자체가 아니라 (1 - hotness), 즉 coldness를 씁니다.
        hotness = _hotness(b)

        # 3) age_norm:
        #    last_activity가 오래전일수록 last_max - last_activity가 커집니다.
        #    따라서 오래 방치된 블록일수록 age_norm이 큽니다.
        age_norm = (last_max - _last_activity(b)) / age_den

        # 4) wear_norm:
        #    erase_count가 현재 최대치에 가까우면 1.0에 가까워집니다.
        #    COTA는 (1 - wear_norm)을 쓰므로, 덜 마모된 블록에 점수를 더 줍니다.
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0

        # 최종 score:
        # 서로 단위가 다른 feature들을 0~1 근처 값으로 맞춘 뒤,
        # 가중합(weighted sum)으로 하나의 숫자를 만듭니다.
        # score가 가장 큰 블록이 이번 GC victim이 됩니다.
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

    COTA와의 차이
    ------------
    COTA는 age를 last_max 기준으로 "오래된 정도"로 계산하고,
    ATCB는 now_step 기준으로 "현재 시점에서 얼마나 오래 지났는지"에 가깝게 계산합니다.
    """
    used_list = [_block_used(b) for b in blocks]
    if not any(used_list):
        return None

    # wear 정규화를 위한 기준값입니다.
    wears = [_wear(b) for b in blocks]
    max_erase = max(wears) if wears else 0

    # now_step이 주어지지 않으면 현재 블록 상태에서 추정합니다.
    # 실험 스크립트에서 실제 step을 넣어주면 더 자연스럽습니다.
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

        # u는 valid_ratio입니다.
        # valid_ratio가 낮을수록 invalid_ratio(inv)가 높아지고 GC 효율이 좋아집니다.
        u = getattr(b, "valid_count", 0) / used
        inv = 1.0 - u

        # wear, age, hotness를 각각 0~1 근처 feature로 바꿉니다.
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0
        age_norm = (now_step - _last_activity(b)) / age_den

        hot = _hotness(b)

        # ATCB도 결국 feature들의 가중합입니다.
        # inv, low-wear, old-age, coldness를 동시에 보겠다는 뜻입니다.
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

    COTA/ATCB보다 단순한 점
    ----------------------
    RE50315는 hotness를 직접 쓰지 않습니다.
    invalid, age, wear만으로 victim을 고릅니다.
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

        # invalid_ratio를 얻기 위해 valid_ratio를 먼저 계산합니다.
        u = getattr(b, "valid_count", 0) / used
        inv = 1.0 - u

        # age와 wear를 상대값으로 바꿉니다.
        wear_norm = (_wear(b) / max_erase) if max_erase > 0 else 0.0
        age_norm = (now_step - _last_activity(b)) / age_den

        # invalid 회수 효율 + age 보너스 + low-wear 보너스.
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

    왜 팩토리가 필요한가?
    --------------------
    CLI에서는 사용자가 --gc_policy cota 같은 문자열을 입력합니다.
    하지만 simulator는 문자열이 아니라 실제 함수가 필요합니다.
    이 함수가 그 둘을 연결해줍니다.
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
