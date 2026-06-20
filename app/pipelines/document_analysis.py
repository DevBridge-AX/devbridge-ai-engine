"""
문서 분석 파이프라인.

Day 2 MVP에서는 외부 LLM 호출 없이 fallback 분석을 먼저 제공합니다.
이 구조를 먼저 고정한 뒤, 이후 ai_analysis_mode에 따라
OpenAI, Anthropic, 사내 모델 등으로 교체할 수 있습니다.
"""

from pathlib import Path

from app.config import get_settings
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
    ".vue",
    ".html",
    ".css",
    ".sql",
    ".yml",
    ".yaml",
    ".xml",
}


def analyze_document(request: DocumentAnalysisRequest) -> DocumentAnalysisResponse:
    settings = get_settings()

    return _fallback_analyze_document(
        request=request,
        model=settings.document_analysis_model,
        mode=settings.ai_analysis_mode,
    )


def _fallback_analyze_document(
    request: DocumentAnalysisRequest,
    model: str,
    mode: str,
) -> DocumentAnalysisResponse:
    file_path = Path(request.file_path)
    file_name = request.title or file_path.name
    doc_type = request.doc_type or "ETC"

    text_preview = _read_text_preview(file_path)
    file_exists = file_path.exists()

    keywords = _build_keywords(
        file_name=file_name,
        doc_type=doc_type,
        text_preview=text_preview,
    )
    risk_level = _estimate_risk_level(
        file_exists=file_exists,
        text_preview=text_preview,
    )
    summary = _build_summary(
        file_name=file_name,
        doc_type=doc_type,
        file_exists=file_exists,
        text_preview=text_preview,
    )
    next_action = _build_next_action(
        file_exists=file_exists,
        text_preview=text_preview,
        task_id=request.task_id,
    )

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


def _read_text_preview(file_path: Path, max_chars: int = 2500) -> str:
    if not file_path.exists() or not file_path.is_file():
        return ""

    if file_path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""

    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    return " ".join(text.split())[:max_chars]


def _build_keywords(file_name: str, doc_type: str, text_preview: str) -> str:
    base_keywords = ["document", doc_type.lower()]
    lower_text = text_preview.lower()

    candidates = [
        ("api", "API"),
        ("task", "Task"),
        ("workspace", "Workspace"),
        ("document", "Document"),
        ("error", "Error"),
        ("risk", "Risk"),
        ("database", "Database"),
        ("upload", "Upload"),
        ("analysis", "Analysis"),
        ("git", "Git"),
        ("commit", "Commit"),
        ("security", "Security"),
    ]

    detected = [label for token, label in candidates if token in lower_text]

    file_tokens = [
        token.strip("._-()[]{} ").lower()
        for token in file_name.replace(".", " ").replace("_", " ").replace("-", " ").split()
        if len(token.strip("._-()[]{} ")) >= 3
    ]

    keywords = base_keywords + detected + file_tokens[:5]

    unique_keywords = []
    for keyword in keywords:
        if keyword and keyword not in unique_keywords:
            unique_keywords.append(keyword)

    return ", ".join(unique_keywords[:10])


def _estimate_risk_level(file_exists: bool, text_preview: str) -> str:
    if not file_exists:
        return "HIGH"

    lower_text = text_preview.lower()

    high_risk_tokens = [
        "critical",
        "blocker",
        "security",
        "leak",
        "failure",
        "fatal",
    ]
    medium_risk_tokens = [
        "todo",
        "fixme",
        "error",
        "warning",
        "risk",
        "delay",
    ]

    if any(token in lower_text for token in high_risk_tokens):
        return "HIGH"

    if any(token in lower_text for token in medium_risk_tokens):
        return "MEDIUM"

    return "LOW"


def _build_summary(
    file_name: str,
    doc_type: str,
    file_exists: bool,
    text_preview: str,
) -> str:
    if not file_exists:
        return (
            f"'{file_name}' 파일 경로를 찾을 수 없어 문서 내용을 분석하지 못했습니다. "
            "Spring Backend의 file_path 저장 값과 AI Engine 실행 환경의 파일 접근 경로를 확인해야 합니다."
        )

    if text_preview:
        short_preview = text_preview[:450]
        return (
            f"'{file_name}' 문서는 {doc_type} 유형의 업무 관련 문서입니다. "
            f"텍스트 일부를 기준으로 보면 다음 내용이 포함되어 있습니다: {short_preview}"
        )

    return (
        f"'{file_name}' 문서는 {doc_type} 유형의 업무 관련 파일입니다. "
        "현재 Day 2 MVP 분석기는 PDF/DOCX 등 비텍스트 파일의 본문 추출을 아직 수행하지 않으므로, "
        "파일 메타데이터 기준으로 분석 결과를 생성했습니다."
    )


def _build_next_action(
    file_exists: bool,
    text_preview: str,
    task_id: str | None,
) -> str:
    if not file_exists:
        return "문서 파일 경로를 확인한 뒤 다시 AI 분석을 실행하세요."

    if not task_id:
        return "문서를 관련 Task에 연결한 뒤 업무 맥락 기준으로 다시 분석하세요."

    if text_preview:
        return "문서 내용을 검토하고 Task 진행 상태, 요구사항 반영 여부, 추가 산출물 필요 여부를 확인하세요."

    return "비텍스트 파일의 경우 PDF/DOCX 본문 추출 기능을 추가한 뒤 상세 분석을 다시 실행하세요."