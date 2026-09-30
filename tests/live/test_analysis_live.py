"""
AI_ANALYSIS_MODE=llm 활성화 라이브 계약 검증 (실 GMS API 호출, 과금 발생).

RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live/test_analysis_live.py -s로
실행합니다(docs/ai-api-usage.md "라이브 검증 실행법" 참고). 이 모듈 안에서만
AI_ANALYSIS_MODE=llm으로 임시 전환하고(get_settings.cache_clear() 포함), 세션 종료
시 원래 값으로 복원합니다 — 운영 기본값(fallback)은 코드 어디서도 바뀌지 않습니다.

문서 분석(md/py/비텍스트 pdf) 3종, 커밋 분석(가상 CommitData) 3종, 워크스페이스 요약
1종에 대해 llm 모드 결과가 계약(4필드 비어있지 않음/risk_level 허용값/한국어 포함
등)을 만족하는지 검증합니다. 품질 판정은 하지 않고, llm 결과와 fallback 결과를
나란히 `live_run_recorder["analysis"]`에 저장해 수동 비교합니다(risk_level 일치
여부는 참고 지표일 뿐 하드 assert하지 않음). 이 값은 세션 종료 시
`tests/live/conftest.py::pytest_sessionfinish`가 만드는 리포트(data/live_runs/*.json)의
`persona_answers_preview`에 함께 실립니다.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import get_settings
from app.pipelines import document_analysis, git_ingestion, workspace_summary
from app.schemas.analysis import DocumentAnalysisRequest
from app.schemas.ingestion import CommitData, GitChangedFileData
from app.schemas.workspace_summary import WorkspaceSummaryRequest

pytestmark = pytest.mark.live

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "analysis"
_KOREAN_RE = re.compile(r"[가-힣]")
_ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


@pytest.fixture(scope="module", autouse=True)
def llm_analysis_mode(live_env):
    """이 모듈 동안만 AI_ANALYSIS_MODE=llm으로 전환하고 종료 후 원복합니다."""
    original = os.environ.get("AI_ANALYSIS_MODE")
    os.environ["AI_ANALYSIS_MODE"] = "llm"
    get_settings.cache_clear()

    yield

    if original is None:
        os.environ.pop("AI_ANALYSIS_MODE", None)
    else:
        os.environ["AI_ANALYSIS_MODE"] = original
    get_settings.cache_clear()


def _count_sentences(text: str) -> int:
    """마침표/느낌표/물음표 뒤 공백 또는 문자열 끝을 문장 경계로 어림잡아 센다."""
    candidates = re.split(r"(?<=[.!?])\s+", text.strip())
    return len([c for c in candidates if c.strip()])


# ---------------------------------------------------------------------------
# 문서 분석 3종 (md / py / 비텍스트 pdf 메타만)
# ---------------------------------------------------------------------------

_DOCUMENT_CASES = [
    ("deploy-checklist.md", "spec"),
    ("retry_policy.py", "code"),
    ("incident-report.pdf", "report"),
]


@pytest.mark.parametrize("filename,doc_type", _DOCUMENT_CASES)
async def test_document_analysis_llm_contract(filename, doc_type, live_run_recorder):
    settings = get_settings()
    assert settings.ai_analysis_mode == "llm"

    request = DocumentAnalysisRequest(
        document_id=f"live-doc-{filename}",
        workspace_id="live-analysis-ws",
        task_id=None,
        title=f"라이브 검증 문서 - {filename}",
        file_path=str(_FIXTURES_DIR / filename),
        doc_type=doc_type,
    )

    llm_result = await document_analysis.analyze_document(request)

    assert llm_result.mode in {"llm", "llm_fallback"}
    assert llm_result.summary.strip() != ""
    assert llm_result.keywords.strip() != ""
    assert llm_result.risk_level in _ALLOWED_RISK_LEVELS
    assert llm_result.next_action.strip() != ""
    assert _KOREAN_RE.search(llm_result.summary), "summary에 한글이 포함되어야 합니다"

    fallback_result = document_analysis._fallback_analyze_document(
        request=request, model=settings.document_analysis_model, mode="fallback"
    )

    analysis_log = live_run_recorder.setdefault("analysis", {})
    doc_log = analysis_log.setdefault("document_analysis", {})
    doc_log[filename] = {
        "llm": {
            "mode": llm_result.mode,
            "risk_level": llm_result.risk_level,
            "summary_preview": llm_result.summary[:200],
        },
        "fallback": {
            "risk_level": fallback_result.risk_level,
            "summary_preview": fallback_result.summary[:200],
        },
        "risk_level_agreement": llm_result.risk_level == fallback_result.risk_level,
    }


# ---------------------------------------------------------------------------
# 커밋 분석 3종 (가상 CommitData, 소형 diff)
# ---------------------------------------------------------------------------


def _commit(hash_suffix: str, message: str, changed_files: list[GitChangedFileData]) -> CommitData:
    return CommitData(
        commit_hash=f"live-hash-{hash_suffix}",
        short_hash=hash_suffix[:7],
        author_name="라이브검증봇",
        author_email="live-bot@example.com",
        message=message,
        committed_at=datetime.now(timezone.utc),
        branch_name="main",
        changed_files=changed_files,
    )


_COMMIT_CASES = [
    _commit(
        "aaa1111",
        "feat: 로그인 폼 입력값 검증 로직 추가",
        [
            GitChangedFileData(
                file_path="src/pages/Login.tsx",
                change_type="MODIFIED",
                additions=14,
                deletions=3,
                diff_summary="이메일/비밀번호 형식 검증 함수를 추가했습니다.",
                patch=(
                    "+ function validateEmail(email: string) {\n"
                    "+   return /.+@.+\\..+/.test(email);\n"
                    "+ }\n"
                    "- // TODO: validate\n"
                ),
            )
        ],
    ),
    _commit(
        "bbb2222",
        "fix: 결제 실패 시 재시도 횟수 제한 적용",
        [
            GitChangedFileData(
                file_path="src/services/PaymentRetryService.java",
                change_type="MODIFIED",
                additions=9,
                deletions=2,
                diff_summary="무한 재시도를 막기 위해 MAX_RETRY_COUNT 상한을 추가했습니다.",
                patch=(
                    "+ private static final int MAX_RETRY_COUNT = 3;\n"
                    "+ if (attempt >= MAX_RETRY_COUNT) { throw new PaymentRetryException(); }\n"
                ),
            )
        ],
    ),
    _commit(
        "ccc3333",
        "chore: Dockerfile 베이스 이미지 버전 업데이트",
        [
            GitChangedFileData(
                file_path="Dockerfile",
                change_type="MODIFIED",
                additions=1,
                deletions=1,
                diff_summary="베이스 이미지를 python:3.11-slim으로 갱신했습니다.",
                patch="- FROM python:3.10-slim\n+ FROM python:3.11-slim\n",
            )
        ],
    ),
]


@pytest.mark.parametrize("commit", _COMMIT_CASES, ids=[c.commit_hash for c in _COMMIT_CASES])
async def test_commit_analysis_llm_contract(commit, live_run_recorder):
    settings = get_settings()
    assert settings.ai_analysis_mode == "llm"

    commit_text = git_ingestion._build_commit_text(commit)

    llm_result = await git_ingestion._analyze_commit(commit=commit, commit_text=commit_text)

    assert llm_result["summary"].strip() != ""
    assert llm_result["impact_area"].strip() != ""
    assert llm_result["risk_level"] in _ALLOWED_RISK_LEVELS
    assert llm_result["next_action"].strip() != ""
    assert _KOREAN_RE.search(llm_result["summary"]), "summary에 한글이 포함되어야 합니다"

    fallback_result = git_ingestion._fallback_analyze_commit(commit, commit_text)

    analysis_log = live_run_recorder.setdefault("analysis", {})
    commit_log = analysis_log.setdefault("commit_analysis", {})
    commit_log[commit.commit_hash] = {
        "llm": {
            "risk_level": llm_result["risk_level"],
            "impact_area": llm_result["impact_area"],
            "summary_preview": llm_result["summary"][:200],
        },
        "fallback": {
            "risk_level": fallback_result["risk_level"],
            "impact_area": fallback_result["impact_area"],
            "summary_preview": fallback_result["summary"][:200],
        },
        "risk_level_agreement": llm_result["risk_level"] == fallback_result["risk_level"],
    }


# ---------------------------------------------------------------------------
# 워크스페이스 요약 1종
# ---------------------------------------------------------------------------


async def test_workspace_summary_llm_contract(live_run_recorder):
    settings = get_settings()
    assert settings.ai_analysis_mode == "llm"

    request = WorkspaceSummaryRequest(
        workspace_id="live-analysis-ws",
        workspace_name="DevBridge 라이브검증 워크스페이스",
        total_task_count=24,
        assigned_task_count=18,
        in_progress_task_count=7,
        done_task_count=12,
        delayed_task_count=2,
        progress_rate=50,
        member_count=6,
        recent_task_count=4,
        recent_git_commit_count=5,
        recent_document_count=3,
    )

    llm_result = await workspace_summary.generate_workspace_summary(request)

    assert llm_result.mode in {"llm", "llm_fallback"}
    assert llm_result.summary.strip() != ""
    assert _KOREAN_RE.search(llm_result.summary), "summary에 한글이 포함되어야 합니다"
    assert request.workspace_name in llm_result.summary
    sentence_count = _count_sentences(llm_result.summary)
    assert 3 <= sentence_count <= 5, f"3~5문장이어야 하는데 {sentence_count}문장입니다"

    fallback_result = workspace_summary._fallback_summary(request, settings)

    analysis_log = live_run_recorder.setdefault("analysis", {})
    analysis_log["workspace_summary"] = {
        "llm": {"mode": llm_result.mode, "summary_preview": llm_result.summary[:300]},
        "fallback": {"summary_preview": fallback_result.summary[:300]},
    }
