"""
FastAPI 애플리케이션 엔트리포인트.

- RAG 기반 질의응답과 문서/Git 인덱싱 라우터를 등록합니다.
- Day 2에서는 Spring Backend에서 호출할 문서 분석 API를 추가합니다.
"""

from fastapi import FastAPI

from app.api import analysis, ingestion
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="DevBridge AI Engine",
    description="RAG 기반 프로젝트 이해 지원 챗봇 AI 백엔드",
    version="0.1.0",
)

app.include_router(ingestion.router, prefix="/api/ingestion", tags=["ingestion"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

# TODO: chat 라우터 구현 후 활성화
# from app.api import chat
# app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """서비스 상태 확인용 헬스체크 엔드포인트."""
    return {"status": "ok", "env": settings.app_env}