"""
문서 분석 API.

Spring Backend가 저장한 KnowledgeDocument의 파일 경로를 전달하면,
AI Engine은 분석 결과를 반환합니다. DB 저장은 Spring Backend가 담당합니다.
"""

from fastapi import APIRouter, Depends

from app.core.security import verify_internal_api_key
from app.pipelines.document_analysis import analyze_document
from app.schemas.analysis import DocumentAnalysisRequest, DocumentAnalysisResponse

router = APIRouter()


@router.post("/document", response_model=DocumentAnalysisResponse)
async def analyze_document_endpoint(
    request: DocumentAnalysisRequest,
    _: None = Depends(verify_internal_api_key),
) -> DocumentAnalysisResponse:
    return analyze_document(request)