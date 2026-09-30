"""
app/core/llm/provider.py 라이브 검증 (실 GMS API 호출, 과금 발생).

RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live로 실행합니다
(docs/ai-api-usage.md "라이브 검증 실행법" 참고). grounding/embedding/call_structured가
실제 GMS 응답을 기대한 형식으로 파싱하는지만 확인하며, `live_env` fixture로 이미
설정/캐시가 초기화된 상태를 전제합니다(워크스페이스 시딩은 불필요).
"""

import pytest

from app.core.embeddings.embedder import embed_texts
from app.core.llm import provider as llm

pytestmark = pytest.mark.live


async def test_grounding_model_returns_structured_json(live_env):
    """call_grounding 결과 dict에 is_groundable(bool)/confidence(float)가 존재."""
    prompt = (
        "질문: 배포는 어떤 전략으로 진행되나요?\n\n"
        "검색된 컨텍스트:\n"
        "[1] (document) 배포 절차\n"
        "배포는 blue-green 전략으로 진행되며 ./deploy.sh --env production --strategy "
        "blue-green 명령을 사용합니다."
    )

    result, usage = await llm.call_grounding(prompt)

    assert "is_groundable" in result
    assert "confidence" in result
    assert isinstance(result["is_groundable"], bool)
    assert isinstance(result["confidence"], (int, float))
    assert usage.prompt_tokens > 0


async def test_embedding_dim_consistent(live_env):
    """embed_texts가 여러 건에 대해 동일한 차원의 임베딩을 반환."""
    result = await embed_texts(["결제 API 문서", "배포 절차 안내"])

    assert len(result.embeddings) == 2
    dim = len(result.embeddings[0])
    assert dim > 0
    assert all(len(embedding) == dim for embedding in result.embeddings)


async def test_call_structured_parses_json(live_env):
    """call_structured가 system_prompt 지시대로 JSON dict를 파싱해 반환."""
    system_prompt = '반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.\n{"greeting": "..."}'
    messages = [
        {"role": "user", "content": "안녕이라는 뜻의 한국어 인사말을 greeting 필드에 담아 JSON으로 답해줘."}
    ]

    result = await llm.call_structured(messages, system_prompt, purpose="live_smoke")

    assert isinstance(result, dict)
    assert "greeting" in result
