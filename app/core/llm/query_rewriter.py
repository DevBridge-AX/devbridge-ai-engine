"""
멀티턴 쿼리 재구성.

- turn 1: 재구성 없이 사용자 쿼리를 그대로 검색에 사용 (이 모듈을 거치지 않음)
- turn 2+: conversation_history를 참고하여 검색에 적합한 독립적인 쿼리로 재구성

내부 LLM 호출은 core.llm.provider.LLMProvider를 통해서만 수행합니다 (현재 placeholder).

TODO:
- rewrite_query(query, conversation_history) -> str 구현
- 재구성 프롬프트 템플릿 설계
"""


async def rewrite_query(query: str, conversation_history: list[dict[str, str]]) -> str:
    """turn 2+ 쿼리를 검색용으로 재구성. TODO: 구현."""
    raise NotImplementedError
