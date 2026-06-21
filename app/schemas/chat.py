"""
/chat 엔드포인트 요청/응답 Pydantic 스키마.

요청(ChatRequest)은 Spring이 매 호출마다 전달하는 stateless payload(대화 히스토리,
workspace_id, user_id, role 등)를 담습니다.

응답은 SSE로 두 종류의 이벤트를 보냅니다.
- event: token (반복) — ChatTokenEvent
- event: done  (1회)  — ChatDoneEvent
"""

from typing import Literal, Optional

from pydantic import BaseModel

from app.core.llm.persona_prompts import PersonaRole


class ChatTokenEvent(BaseModel):
    """event: token 의 data. 스트리밍 중 답변 텍스트 조각."""

    text: str


class ConversationMessage(BaseModel):
    """conversation_history의 개별 메시지."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Spring -> FastAPI /chat 요청 스키마 (stateless payload)."""

    session_id: str
    content: str
    conversation_history: list[ConversationMessage]
    workspace_id: str
    user_id: str
    role: PersonaRole


class Citation(BaseModel):
    """답변 근거 인용 정보.

    MESSAGE_CITATIONS(Spring 소유)의 source_type/source_id/vector_chunk_id/
    similarity_score에 1:1 매핑됩니다. source_id는 source_type에 따라
    KNOWLEDGE_DOCUMENTS / GIT_COMMITS / DATABASE_SCHEMAS 중 해당 테이블의 PK이며,
    chunk_id는 document_chunks.id입니다.
    """

    source_type: Literal["document", "git_commit", "db_schema"]
    source_id: str
    chunk_id: int
    title: str
    similarity_score: float


class TokenUsageDetail(BaseModel):
    """모델별 raw 토큰 사용량. Credit 단가 적용은 Spring의 책임입니다."""

    model: str
    prompt_tokens: int
    completion_tokens: int


class TokenUsage(BaseModel):
    """/chat 호출의 토큰 사용량.

    turn 1에서는 rewrite가 호출되지 않으므로 rewrite는 None입니다.
    """

    main: TokenUsageDetail
    rewrite: Optional[TokenUsageDetail] = None
    context_truncated: bool


class ChatDoneEvent(BaseModel):
    """event: done 의 data. 스트림 종료 시 1회 전송되는 메타데이터.

    is_groundable/confidence의 임계치 비교 및 알림 트리거(UC-04)는 Spring의
    책임이며, 이 레포는 값만 반환합니다.
    """

    citations: list[Citation]
    is_groundable: bool
    confidence: float
    suggested_owner_id: Optional[str] = None  # GIT_COMMITS.author_id (USERS.id UUID String)
    prompt_version: str
    token_usage: TokenUsage
