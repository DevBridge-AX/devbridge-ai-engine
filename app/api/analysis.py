"""
문서 분석 및 워크스페이스 요약 API 엔드포인트.

Spring Backend가 AI Engine에 문서 분석 및 대시보드 요약을 요청합니다.
"""

import logging

from fastapi import APIRouter, Depends, Request

from app.core.security import verify_internal_api_key
from app.pipelines.document_analysis import analyze_document
from app.pipelines.workspace_summary import generate_workspace_summary
from app.schemas.analysis import DocumentAnalysisRequest, DocumentAnalysisResponse
from app.schemas.workspace_summary import (
    WorkspaceSummaryRequest,
    WorkspaceSummaryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/document", response_model=DocumentAnalysisResponse)
async def analyze_document_endpoint(
    request: DocumentAnalysisRequest,
    _: None = Depends(verify_internal_api_key),
) -> DocumentAnalysisResponse:
    return await analyze_document(request)


@router.post("/document/debug")
async def analyze_document_debug(request: Request):
    """디버그 엔드포인트 — raw body 수신 여부 확인"""
    body = await request.body()
    headers = dict(request.headers)
    logger.info("=== DEBUG raw body (%d bytes) ===", len(body))
    logger.info("Headers: %s", headers)
    logger.info("Body repr: %r", body[:2000])
    return {
        "body_length": len(body),
        "body_preview": body[:500].decode("utf-8", errors="replace"),
    }


@router.post("/workspace-summary", response_model=WorkspaceSummaryResponse)
async def workspace_summary_endpoint(
    request: WorkspaceSummaryRequest,
    _: None = Depends(verify_internal_api_key),
) -> WorkspaceSummaryResponse:
    """워크스페이스 대시보드 메트릭 기반 AI 요약을 생성합니다."""
    return await generate_workspace_summary(request)
