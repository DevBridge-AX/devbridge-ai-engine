"""
워크스페이스 대시보드 AI 요약 스키마.

Spring Backend가 대시보드 메트릭을 전송하면 AI Engine이 한글 요약을 생성합니다.
"""

from pydantic import BaseModel, Field


class WorkspaceSummaryRequest(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID")
    workspace_name: str = Field(..., description="워크스페이스 이름")

    total_task_count: int = Field(..., description="전체 업무 수")
    assigned_task_count: int = Field(..., description="배정 업무 수")
    in_progress_task_count: int = Field(..., description="진행 중 업무 수")
    done_task_count: int = Field(..., description="완료 업무 수")
    delayed_task_count: int = Field(..., description="지연 업무 수")
    progress_rate: int = Field(..., description="완료율 (0-100)")
    member_count: int = Field(..., description="멤버 수")

    recent_task_count: int = Field(..., description="최근 업무 건수")
    recent_git_commit_count: int = Field(..., description="최근 Git Commit 건수")
    recent_document_count: int = Field(..., description="최근 문서 건수")


class WorkspaceSummaryResponse(BaseModel):
    workspace_id: str
    summary: str
    model: str
    mode: str
