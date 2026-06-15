"""
/chat 엔드포인트 요청/응답 Pydantic 스키마.

요청은 Spring이 매 호출마다 전달하는 stateless payload(대화 히스토리, workspace_id,
user_id, role 등)를 담습니다. 응답은 SSE 스트림과 함께 최종적으로
{citations, similarity_scores, is_groundable, confidence, suggested_owner_id,
token_usage}를 포함해야 합니다.

TODO:
- ChatMessage, ChatRequest, Citation, ChatResponseMeta 필드 정의
- role: core.llm.persona_prompts.PersonaRole 값과 매칭
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """채팅 요청 스키마. TODO: 필드 정의 (messages, workspace_id, user_id, role 등)."""


class Citation(BaseModel):
    """답변 근거 인용 정보 (document_id, chunk_id, similarity_score, source_type 등).

    MESSAGE_CITATIONS는 Spring 소유 테이블이며, 이 스키마는 응답 payload로만 전달되는
    citation 관련 값을 표현합니다. TODO: 필드 정의.
    """


class ChatResponseMeta(BaseModel):
    """채팅 응답 메타데이터.

    TODO: citations, similarity_scores, is_groundable, confidence,
    suggested_owner_id, token_usage 필드 정의.

    임계치 비교/알림 트리거(UC-04)는 Spring 책임이며, 이 스키마는 값만 전달합니다.
    """
