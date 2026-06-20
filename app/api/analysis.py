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
    return await analyze_document(request)