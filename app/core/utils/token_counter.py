"""
tiktoken 기반 토큰 추정 및 대화 히스토리 트런케이션.

cl100k_base 인코딩을 사용합니다. Claude 전용 토크나이저가 공개되지 않아
근사치로 사용하며, 특히 한국어는 영어 대비 토큰 수가 과대 추정될 수 있습니다
(한국어 한 글자 ≈ 1~2 토큰, 실제 Claude는 이보다 효율적일 수 있음).
확인 필요: 실 서비스 데이터로 오차 범위(≤10%) 검증 권장.

Turn 1 보호 정책:
- 히스토리의 첫 번째 턴(user + assistant 쌍)이 전체 예산의 30% 이하일 때만 보호합니다.
- 보호된 Turn 1은 최신 우선 역순 탐색에서 잘리더라도 강제로 포함됩니다.
- Turn 1이 매우 긴 경우(예: 코드 블록 대량 포함) 예산 초과 위험이 있으므로
  30% 조건을 초과하면 보호하지 않습니다.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

_FIRST_TURN_PROTECTION_RATIO = 0.3


def estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수를 추정합니다 (cl100k_base 기준)."""
    return len(_enc.encode(text))


def truncate_history(
    history: list[dict],
    budget: int = 22000,
) -> tuple[list[dict], bool]:
    """대화 히스토리를 토큰 예산에 맞게 트런케이션합니다.

    Args:
        history: [{"role": "user"/"assistant", "content": "..."}, ...] 형식의 대화 목록.
                 conversation_history (이전 턴들)만 전달합니다. 현재 쿼리는 포함하지 않습니다.
        budget:  히스토리에 허용되는 최대 토큰 수.

    Returns:
        (truncated_history, context_truncated):
        - truncated_history: 예산 내 포함 가능한 메시지 목록 (시간 순).
        - context_truncated: 하나라도 잘린 경우 True.
    """
    if not history:
        return [], False

    msg_tokens = [estimate_tokens(m.get("content", "")) for m in history]

    # Turn 1 보호 여부 결정
    first_turn_end = min(2, len(history))
    first_turn_tokens = sum(msg_tokens[:first_turn_end])
    protect_first = first_turn_tokens <= budget * _FIRST_TURN_PROTECTION_RATIO

    # 최신 턴부터 역순으로 포함 (연속된 최근 컨텍스트 우선)
    included = set()
    remaining = budget
    for i in range(len(history) - 1, -1, -1):
        if msg_tokens[i] <= remaining:
            included.add(i)
            remaining -= msg_tokens[i]
        else:
            break

    # Turn 1 보호 적용: 포함되지 않은 경우 강제 삽입
    if protect_first:
        for i in range(first_turn_end):
            included.add(i)

    context_truncated = len(included) < len(history)
    truncated = [history[i] for i in sorted(included)]
    return truncated, context_truncated
