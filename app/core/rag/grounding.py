"""
신뢰도/그라운딩 판정.

retriever의 유사도 점수(1차 필터)와 LLM structured output의 is_groundable/confidence
(최종 판단)를 결합하여 최종 신뢰도를 산출합니다.

주의:
- 임계치 비교 및 알림 트리거(UC-04)는 Spring 백엔드의 책임입니다.
  이 모듈은 is_groundable, confidence 값을 산출하여 반환만 합니다.

TODO:
- assess_grounding(similarity_scores, answer, retrieved_context) -> GroundingResult 구현
- LLM structured output 호출은 core.llm.provider.LLMProvider.generate_structured 사용
"""


async def assess_grounding(similarity_scores: list[float], answer: str, retrieved_context: str):
    """유사도 점수 + LLM 자기평가를 결합해 is_groundable/confidence 산출. TODO: 구현."""
    raise NotImplementedError
