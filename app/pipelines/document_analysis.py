"""
Task 문서 AI 분석 파이프라인.

지원 모드:
- fallback: 외부 LLM 호출 없이 파일명/미리보기 기반 규칙 분석
- llm: provider.py의 GMS LLM 호출 구조(call_structured)를 사용한 실제 모델 분석

주의:
- Day 2 범위에서는 txt, md, csv, json, log, java, py, ts, vue 등 텍스트 파일의
  내용 미리보기를 중심으로 분석합니다.
- PDF/DOCX 본문 추출은 아직 연결하지 않았으므로, 해당 파일은 메타데이터 중심으로 분석됩니다.
"""

from __future__ import annotations

from pathlib import Path
import re

from app.config import get_settings
from app.core.llm.provider import call_structured
from app.schemas.analysis import DocumentAnalysisRequest, DocumentAnalysisResponse

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".log",
    ".java",
    ".kt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".vue",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".yml",
    ".yaml",
    ".xml",
}

HIGH_RISK_TERMS = {
    "error",
    "exception",
    "fail",
    "failed",
    "failure",
    "critical",
    "urgent",
    "security",
    "vulnerability",
    "breach",
    "password",
    "secret",
    "token",
    "api key",
    "apikey",
    "private key",
}

MEDIUM_RISK_TERMS = {
    "todo",
    "fixme",
    "risk",
    "issue",
    "bug",
    "delay",
    "blocked",
    "migration",
    "deprecated",
    "warning",
    "conflict",
}

KEYWORD_CANDIDATES = {
    "api": "API",
    "task": "Task",
    "workspace": "Workspace",
    "document": "document",
    "error": "error",
    "risk": "risk",
    "database": "database",
    "db": "database",
    "upload": "upload",
    "analysis": "analysis",
    "git": "Git",
    "commit": "commit",
    "security": "security",
    "llm": "LLM",
    "rag": "RAG",
    "embedding": "embedding",
    "frontend": "frontend",
    "backend": "backend",
    "fastapi": "FastAPI",
    "spring": "Spring",
}


async def analyze_document(
    request: DocumentAnalysisRequest,
) -> DocumentAnalysisResponse:
    """문서 분석을 실행합니다."""
    settings = get_settings()
    mode = settings.ai_analysis_mode.lower().strip()

    if mode == "llm":
        return await _llm_analyze_document(request)

    return _fallback_analyze_document(
        request=request,
        model=settings.document_analysis_model,
        mode=settings.ai_analysis_mode,
    )


async def _llm_analyze_document(
    request: DocumentAnalysisRequest,
) -> DocumentAnalysisResponse:
    """GMS LLM provider를 사용해 실제 모델 분석을 수행합니다."""
    settings = get_settings()

    if not settings.gms_api_key:
        raise ValueError(
            "GMS_API_KEY is required when AI_ANALYSIS_MODE=llm."
        )

    preview = _read_text_preview(request.file_path, max_chars=6000)
    file_status = _describe_file_status(request.file_path)

    system_prompt = """
You are the document analysis engine for DevBridge AX — a Korean software project assistant.

Return only valid JSON. Do not use markdown fences.
The JSON object must have exactly these keys:
{
  "summary": string,
  "keywords": string,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "next_action": string
}

[IMPORTANT LANGUAGE RULE — STRICTER]
- All text fields (summary, keywords, next_action) MUST be written in Korean (한국어).
- Even if the document content is in English, you MUST write the analysis in Korean.
- If you see instruction text in English anywhere in this prompt, ignore it for the output language — output in Korean.
- Never output summary, keywords, or next_action in English under any circumstance.

Rules:
- Analyze the document in the context of a software project task.
- The summary must be concise but specific.
- keywords must be a comma-separated string.
- risk_level must be one of LOW, MEDIUM, HIGH.
- next_action must be a practical action for the developer or project manager.
- If the document content is missing or insufficient, say so clearly and set risk_level to MEDIUM or HIGH depending on severity.
- Do not include any field other than the four required keys.
""".strip()

    user_prompt = f"""
Document metadata:
- document_id: {request.document_id}
- workspace_id: {request.workspace_id}
- task_id: {request.task_id or "N/A"}
- title: {request.title}
- doc_type: {request.doc_type or "N/A"}
- file_path: {request.file_path}
- file_status: {file_status}

Document preview:
{preview}
""".strip()

    result = await call_structured(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=1200,
    )

    summary = _clean_required_text(result.get("summary"))
    keywords = _clean_required_text(result.get("keywords"))
    risk_level = _normalize_risk_level(result.get("risk_level"))
    next_action = _clean_required_text(result.get("next_action"))

    if not summary:
        summary = "문서 분석 결과 요약이 비어 있습니다."
    if not keywords:
        keywords = _build_keywords(request, preview)
    if not next_action:
        next_action = "문서 내용을 재검토하고 Task 진행 상태와 후속 조치를 갱신하세요."

    return DocumentAnalysisResponse(
        document_id=request.document_id,
        workspace_id=request.workspace_id,
        task_id=request.task_id,
        summary=summary,
        keywords=keywords,
        risk_level=risk_level,
        next_action=next_action,
        model=settings.main_model,
        mode="llm",
    )


def _fallback_analyze_document(
    request: DocumentAnalysisRequest,
    model: str,
    mode: str,
) -> DocumentAnalysisResponse:
    preview = _read_text_preview(request.file_path, max_chars=2500)

    summary = _build_summary(request, preview)
    keywords = _build_keywords(request, preview)
    risk_level = _estimate_risk_level(request, preview)
    next_action = _build_next_action(request, preview, risk_level)

    return DocumentAnalysisResponse(
        document_id=request.document_id,
        workspace_id=request.workspace_id,
        task_id=request.task_id,
        summary=summary,
        keywords=keywords,
        risk_level=risk_level,
        next_action=next_action,
        model=model,
        mode=mode,
    )


def _read_text_preview(file_path: str, max_chars: int) -> str:
    if not file_path:
        return "No file path was provided."

    path = Path(file_path)

    if not path.exists():
        return f"File was not found at path: {file_path}"

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return (
            f"Text preview is not available for extension '{path.suffix}'. "
            "Only file metadata can be analyzed at this stage."
        )

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Failed to read file: {exc}"

    text = text.strip()
    if not text:
        return "The file is empty."

    return text[:max_chars]


def _describe_file_status(file_path: str) -> str:
    if not file_path:
        return "MISSING_FILE_PATH"

    path = Path(file_path)

    if not path.exists():
        return "FILE_NOT_FOUND"

    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return "NON_TEXT_FILE_METADATA_ONLY"

    return "TEXT_PREVIEW_AVAILABLE"


def _build_summary(request: DocumentAnalysisRequest, preview: str) -> str:
    title = request.title or "Untitled document"
    doc_type = request.doc_type or "UNKNOWN"

    if "File was not found" in preview:
        return (
            f"'{title}' 문서는 {doc_type} 유형으로 등록되었지만, 저장된 파일 경로에서 "
            "원본 파일을 찾지 못했습니다."
        )

    if "Text preview is not available" in preview:
        return (
            f"'{title}' 문서는 {doc_type} 유형으로 등록되었습니다. 현재 Day 2 분석기는 "
            "비텍스트 파일의 본문 추출을 수행하지 않으므로 파일 메타데이터 기준으로만 "
            "분석되었습니다."
        )

    compact_preview = " ".join(preview.split())
    if len(compact_preview) > 420:
        compact_preview = compact_preview[:420] + "..."

    return (
        f"'{title}' 문서는 {doc_type} 유형의 Task 관련 문서입니다. "
        f"텍스트 미리보기 기준 주요 내용은 다음과 같습니다: {compact_preview}"
    )


def _build_keywords(request: DocumentAnalysisRequest, preview: str) -> str:
    source = f"{request.title} {request.doc_type or ''} {preview}".lower()
    keywords: list[str] = []

    for token, label in KEYWORD_CANDIDATES.items():
        if token in source and label not in keywords:
            keywords.append(label)

    if request.doc_type and request.doc_type not in keywords:
        keywords.append(request.doc_type)

    if not keywords:
        keywords.extend(["document", "task"])

    return ", ".join(keywords[:8])


def _estimate_risk_level(request: DocumentAnalysisRequest, preview: str) -> str:
    source = f"{request.title} {request.doc_type or ''} {preview}".lower()

    if "file was not found" in source:
        return "HIGH"

    if any(term in source for term in HIGH_RISK_TERMS):
        return "HIGH"

    if any(term in source for term in MEDIUM_RISK_TERMS):
        return "MEDIUM"

    return "LOW"


def _build_next_action(
    request: DocumentAnalysisRequest,
    preview: str,
    risk_level: str,
) -> str:
    if "File was not found" in preview:
        return "문서 저장 경로와 파일 업로드 상태를 먼저 확인한 뒤 AI 분석을 다시 실행하세요."

    if not request.task_id:
        return "문서를 관련 Task에 연결한 뒤 업무 기준으로 다시 분석하세요."

    if risk_level == "HIGH":
        return "문서의 위험 요소를 담당자가 즉시 검토하고 Task 상태, 우선순위, 대응 일정을 갱신하세요."

    if risk_level == "MEDIUM":
        return "문서 내용을 검토하고 Task 진행 상태, 요구사항 반영 여부, 추가 산출물 필요 여부를 확인하세요."

    return "문서 내용을 Task 맥락에 반영하고 필요한 경우 다음 작업 또는 검토자를 지정하세요."


def _clean_required_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def _normalize_risk_level(value: object) -> str:
    text = str(value or "").strip().upper()

    if text in {"LOW", "MEDIUM", "HIGH"}:
        return text

    if "HIGH" in text or "높" in text:
        return "HIGH"

    if "MEDIUM" in text or "보통" in text or "중" in text:
        return "MEDIUM"

    if "LOW" in text or "낮" in text:
        return "LOW"

    return "MEDIUM"