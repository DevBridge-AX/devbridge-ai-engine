"""
Internal API Key 검증.

Spring -> FastAPI의 모든 요청에는 X-Internal-Api-Key 헤더가 포함되어야 하며,
.env의 INTERNAL_API_KEY와 일치하지 않으면 401을 반환합니다. 이 레포는 최종
사용자 인증을 직접 수행하지 않으며, FastAPI는 내부망에서만 접근 가능해야 합니다.
"""

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_internal_api_key(
    x_internal_api_key: str = Header(..., alias="X-Internal-Api-Key"),
) -> None:
    """X-Internal-Api-Key 헤더를 검증하는 FastAPI dependency."""
    settings = get_settings()
    if x_internal_api_key != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Internal-Api-Key header",
        )
