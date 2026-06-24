"""
FastAPI application entrypoint.

- RAG 기반 질의응답(/chat), 문서/Git 인덱싱(/ingestion), 사용량(/usage) 라우터를 등록합니다.
- Spring Backend에서 호출할 문서 분석 API(/analysis)를 등록합니다.
- 이 서비스는 stateless 추론 엔진으로 동작하며, session_id/workspace_id/user_id/role 등의
  상태값은 요청마다 Spring Backend가 payload로 전달합니다.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import analysis, chat, ingestion, usage
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title="DevBridge AI Engine",
    description="RAG 기반 프로젝트 이해 지원 AI 백엔드",
    version="0.1.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 발생 시 raw body 로깅"""
    body = await request.body()
    logger.error(
        "=== 422 VALIDATION ERROR ===\n"
        "url=%s\n"
        "content_type=%s\n"
        "raw_body_length=%d\n"
        "raw_body=%r\n"
        "errors=%s",
        request.url,
        request.headers.get("content-type"),
        len(body),
        body[:2000],
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

app.include_router(ingestion.router, prefix="/api/ingestion", tags=["ingestion"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(usage.router, prefix="/api/usage", tags=["usage"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """서비스 상태 확인용 헬스체크 엔드포인트입니다."""
    return {"status": "ok", "env": settings.app_env}