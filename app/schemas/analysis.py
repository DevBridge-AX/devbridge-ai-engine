"""
문서 분석 API 요청/응답 스키마.

Spring Backend가 이미 문서 메타데이터와 파일 경로를 저장한 뒤,
AI Engine은 해당 파일을 분석하고 요약 결과만 반환합니다.
"""

from pydantic import BaseModel, Field


class DocumentAnalysisRequest(BaseModel):
    document_id: str = Field(..., description="Spring Backend KnowledgeDocument ID")
    workspace_id: str = Field(..., description="Workspace ID")
    task_id: str | None = Field(default=None, description="Task ID")
    title: str = Field(..., description="문서 제목")
    file_path: str = Field(..., description="Spring Backend가 저장한 파일 경로")
    doc_type: str | None = Field(default=None, description="문서 유형")


class DocumentAnalysisResponse(BaseModel):
    document_id: str
    workspace_id: str
    task_id: str | None = None

    summary: str
    keywords: str
    risk_level: str
    next_action: str

    model: str
    mode: str