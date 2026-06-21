"""
워크스페이스별 임베딩 토큰 사용량 집계 API.

GET /usage/summary?workspace_id={id}
  — usage_logs 테이블에서 workspace_id 기준 일 단위 집계값을 반환합니다.

이 레포는 토큰 수치만 반환합니다. Credit 단가 적용 및 비용 계산은 Spring의 책임입니다.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import verify_internal_api_key
from app.db.models import UsageLog
from app.db.session import get_db

router = APIRouter()


@router.get("/summary")
async def usage_summary(
    workspace_id: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> dict:
    """workspace_id 기준 일 단위 임베딩 토큰 사용량을 반환합니다."""
    rows = db.execute(
        select(UsageLog)
        .where(UsageLog.workspace_id == workspace_id)
        .order_by(UsageLog.date)
    ).scalars().all()

    return {
        "workspace_id": workspace_id,
        "records": [
            {
                "date": row.date.isoformat(),
                "embedding_model": row.embedding_model,
                "embedding_tokens": row.embedding_tokens,
            }
            for row in rows
        ],
    }
