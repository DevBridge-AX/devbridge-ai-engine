"""
멀티턴 RAG 챗봇 SSE 스트리밍 엔드포인트.

Spring 백엔드가 매 요청마다 대화 히스토리, workspace_id, user_id, role을 포함한
stateless payload를 전달하면, StreamingResponse(SSE)로 토큰 단위 답변을 스트리밍하고
최종적으로 {answer_stream, citations[], similarity_scores[], is_groundable, confidence,
suggested_owner_id, token_usage}를 구성해 반환합니다.

처리 흐름 (구현 예정):
1. turn 1: query_rewriter를 거치지 않고 사용자 쿼리로 바로 검색
   turn 2+: core.llm.query_rewriter로 쿼리 재구성 후 검색
2. core.rag.retriever로 document_chunks 벡터 검색 (유사도 점수 포함)
3. core.llm.persona_prompts로 role(직무)별 시스템 프롬프트 구성
   (동적 변수: retrieved_context, conversation_history, user_preference_summary)
4. core.llm.provider로 LLM 호출, 토큰 스트리밍
5. core.rag.grounding으로 유사도 점수 + LLM structured output(is_groundable/confidence) 결합 판정
6. 응답 payload 조립 (citations, similarity_scores, suggested_owner_id, token_usage 등)
   - 임계치 비교/알림 트리거(UC-04)는 Spring 책임 — 이 엔드포인트는 값만 반환

TODO:
- POST /chat 엔드포인트 (StreamingResponse) 구현
- 요청 스키마: app.schemas.chat.ChatRequest 참조
- user_preference_summary는 (user_id, workspace_id) 복합키로 격리 조회
"""

from fastapi import APIRouter

router = APIRouter()

# TODO: SSE 스트리밍 챗 엔드포인트 구현
# @router.post("/")
# async def chat_stream(request: ChatRequest) -> StreamingResponse:
#     ...
