"""
LoRA 학습데이터 export API.

POST /training-data/export
  — Spring이 전달한 레코드를 PII 스크러빙 후 workspace별 JSONL로 저장합니다.
    이 레포는 Spring 소유 테이블(CHAT_MESSAGES 등)을 조회하지 않습니다.
"""

from fastapi import APIRouter, Depends

from app.core.security import verify_internal_api_key
from app.pipelines.export_training_data import export_training_data
from app.schemas.training_data import (
    TrainingDataExportRequest,
    TrainingDataExportResponse,
)

router = APIRouter()


@router.post("/export", response_model=TrainingDataExportResponse)
def export(
    request: TrainingDataExportRequest,
    _: None = Depends(verify_internal_api_key),
) -> TrainingDataExportResponse:
    # 동기 함수: 파일 쓰기가 있어 FastAPI 스레드풀에서 실행되도록 def로 선언
    return export_training_data(request)
