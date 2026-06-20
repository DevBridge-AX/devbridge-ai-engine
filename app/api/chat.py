"""
멀티턴 RAG 챗봇 SSE 스트리밍 엔드포인트.

POST /chat 는 StreamingResponse(text/event-stream)로 두 종류의 이벤트를 반환합니다.
  event: token  — 답변 텍스트 청크 (반복)
  event: done   — 메타데이터 (스트림 종료 시 1회)
  event: error  — 파이프라인 예외 발생 시

X-Internal-Api-Key 헤더 검증이 모든 요청에 적용됩니다.
"""

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core import chat_pipeline
from app.core.security import verify_internal_api_key
from app.db.session import get_db
from app.schemas.chat import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> StreamingResponse:
    """RAG 기반 멀티턴 챗봇 SSE 스트리밍 엔드포인트."""

    async def generate():
        try:
            async for event in chat_pipeline.run(request, db):
                payload = json.dumps(event.data, ensure_ascii=False)
                yield f"event: {event.event}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.exception("chat_pipeline 처리 중 오류 발생")
            error_payload = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
