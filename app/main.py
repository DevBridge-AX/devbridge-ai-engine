"""
FastAPI 애플리케이션 엔트리포인트.

- RAG 기반 질의응답(/chat), 문서/Git 인덱싱(/ingestion), 임베딩 사용량(/usage) 라우터를 등록합니다.
- 이 서비스는 stateless 추론 엔진으로, 대화 히스토리/workspace_id/user_id/role 등의
  상태는 매 요청마다 Spring 백엔드(backend 레포)가 payload로 전달합니다.

TODO:
- 전역 예외 핸들러 / 로깅 미들웨어 추가
- CORS 설정 (Spring 백엔드 도메인만 허용)
"""

from fastapi import FastAPI

from app.api import chat, ingestion, usage
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="DevBridge AI Engine",
    description="RAG 기반 프로젝트 이해 지원 챗봇 AI 백엔드",
    version="0.1.0",
)

app.include_router(ingestion.router, prefix="/api/ingestion", tags=["ingestion"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(usage.router, prefix="/api/usage", tags=["usage"])


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """서비스 상태 확인용 헬스체크 엔드포인트."""
    return {"status": "ok", "env": settings.app_env}
