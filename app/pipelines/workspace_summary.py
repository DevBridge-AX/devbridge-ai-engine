"""
워크스페이스 대시보드 AI 요약 파이프라인.

지원 모드:
- fallback: 메트릭 기반 규칙 생성 한글 요약
- llm: call_structured()를 사용해 LLM이 생성한 한글 요약
"""

from __future__ import annotations

from app.config import get_settings
from app.core.llm.provider import call_structured
from app.schemas.workspace_summary import (
    WorkspaceSummaryRequest,
    WorkspaceSummaryResponse,
)


async def generate_workspace_summary(
    request: WorkspaceSummaryRequest,
) -> WorkspaceSummaryResponse:
    """워크스페이스 대시보드 요약을 생성합니다."""
    settings = get_settings()
    mode = settings.ai_analysis_mode.lower().strip()

    if mode == "llm" and settings.gms_api_key:
        return await _llm_summary(request, settings)
    return _fallback_summary(request, settings)


async def _llm_summary(
    request: WorkspaceSummaryRequest,
    settings,
) -> WorkspaceSummaryResponse:
    """LLM provider를 사용해 워크스페이스 상태 요약을 생성합니다."""

    system_prompt = """
You are DevBridge AX, a Korean workspace-project intelligence assistant.

Your task: Generate a natural Korean paragraph that summarizes the workspace status based on the given numeric metrics.

Return ONLY valid JSON — no markdown fences, no extra keys:
{
  "summary": string
}

Rules:
- The summary MUST be written entirely in Korean (한국어).
- The summary must be 3-5 sentences long.
- Mention the workspace name.
- Include the overall task count and completion rate.
- Highlight in-progress and delayed tasks if significant.
- Comment on recent activity (commits, documents).
- End with a practical observation or recommendation.
- Do NOT include any field other than "summary".
""".strip()

    metrics_text = (
        f"Workspace: {request.workspace_name} (ID: {request.workspace_id})\n"
        f"Total tasks: {request.total_task_count}\n"
        f"Assigned: {request.assigned_task_count}\n"
        f"In progress: {request.in_progress_task_count}\n"
        f"Done: {request.done_task_count}\n"
        f"Delayed: {request.delayed_task_count}\n"
        f"Progress rate: {request.progress_rate}%\n"
        f"Members: {request.member_count}\n"
        f"Recent tasks: {request.recent_task_count}\n"
        f"Recent git commits: {request.recent_git_commit_count}\n"
        f"Recent documents: {request.recent_document_count}"
    )

    result = await call_structured(
        messages=[{"role": "user", "content": metrics_text}],
        system_prompt=system_prompt,
        max_tokens=800,
    )

    summary = str(result.get("summary", "")).strip()
    if not summary:
        summary = _build_fallback_summary(request)

    return WorkspaceSummaryResponse(
        workspace_id=request.workspace_id,
        summary=summary,
        model=settings.main_model,
        mode="llm",
    )


def _fallback_summary(
    request: WorkspaceSummaryRequest,
    settings,
) -> WorkspaceSummaryResponse:
    """메트릭 기반 규칙 템플릿으로 한글 요약을 생성합니다."""
    summary = _build_fallback_summary(request)

    return WorkspaceSummaryResponse(
        workspace_id=request.workspace_id,
        summary=summary,
        model=settings.document_analysis_model,
        mode="fallback",
    )


def _build_fallback_summary(request: WorkspaceSummaryRequest) -> str:
    """실제 메트릭을 기반으로 한글 요약 문장을 조합합니다."""
    lines: list[str] = []

    lines.append(
        f"📊 {request.workspace_name} 워크스페이스에는 현재 총 "
        f"{request.total_task_count}개의 업무가 등록되어 있으며, "
        f"이 중 {request.done_task_count}건이 완료"
        f"(완료율 {request.progress_rate}%), "
        f"{request.in_progress_task_count}건이 진행 중입니다."
    )

    if request.delayed_task_count > 0:
        lines.append(
            f"⚠️ 지연 업무가 {request.delayed_task_count}건 있어 "
            f"우선 확인이 필요합니다."
        )
    else:
        lines.append("✅ 현재 지연 업무가 없어 일정 리스크가 낮습니다.")

    if request.recent_git_commit_count > 0:
        lines.append(
            f"🔄 최근 {request.recent_git_commit_count}건의 Git Commit이 "
            f"있어 활발한 개발이 진행 중입니다."
        )

    if request.recent_document_count > 0:
        lines.append(
            f"📄 최근 {request.recent_document_count}건의 문서가 "
            f"업데이트되었습니다."
        )

    if request.progress_rate >= 70:
        lines.append(
            "💡 완료율이 높은 편이므로 마무리 작업과 산출물 검토에 "
            "집중하면 좋습니다."
        )
    elif request.progress_rate >= 40:
        lines.append(
            "💡 진행 중인 업무의 병목 여부를 확인하고 "
            "우선순위를 조정하는 것이 좋습니다."
        )
    else:
        lines.append(
            "💡 초기 단계로 핵심 업무의 우선순위를 재정리하고 "
            "단기 목표를 설정하는 것이 좋습니다."
        )

    return "\n\n".join(lines)
