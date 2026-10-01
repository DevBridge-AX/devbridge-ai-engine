"""
app/pipelines/git_ingestion.py 커밋 분석 LLM 프롬프트 유닛 테스트 (API 호출 없음).

_llm_analyze_commit()이 call_structured()에 넘기는 system/user 프롬프트에
author_email이 전혀 포함되지 않는지(PII 최소화), 한국어 출력 규칙이 system prompt에
포함되는지를 확인합니다. commit_text는 임베딩/RAG 청크용 _build_commit_text() 원본을
그대로 사용하되(그 안에는 author_email이 들어있음), LLM 프롬프트에 넣기 직전에만
제거되는지를 검증합니다.
"""

import asyncio
from datetime import datetime, timezone

from app.pipelines import git_ingestion
from app.schemas.ingestion import CommitData


def _commit() -> CommitData:
    return CommitData(
        commit_hash="hash-abc123",
        short_hash="abc123",
        author_name="테스터",
        author_email="tester-secret@example.com",
        message="fix: 로그인 오류 수정",
        committed_at=datetime.now(timezone.utc),
        branch_name="main",
    )


def test_llm_analyze_commit_prompt_excludes_author_email_and_has_korean_rule(monkeypatch):
    commit = _commit()
    # 임베딩/RAG 청크에 쓰이는 원본 commit_text에는 author_email이 포함되어 있다.
    commit_text = git_ingestion._build_commit_text(commit)
    assert commit.author_email in commit_text

    captured = {}

    async def fake_call_structured(*, messages, system_prompt, max_tokens, purpose):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = messages[0]["content"]
        captured["purpose"] = purpose
        return {
            "summary": "로그인 오류를 수정했습니다.",
            "impact_area": "backend",
            "risk_level": "LOW",
            "next_action": "관련 테스트를 실행하세요.",
        }

    monkeypatch.setattr(git_ingestion, "call_structured", fake_call_structured)

    result = asyncio.run(git_ingestion._llm_analyze_commit(commit, commit_text))

    # author_email 값과 필드명 모두 LLM 프롬프트 입력에 남지 않아야 한다.
    assert commit.author_email not in captured["user_prompt"]
    assert commit.author_email not in captured["system_prompt"]
    assert "author_email" not in captured["user_prompt"]
    assert "author_email" not in captured["system_prompt"]

    # 한국어 출력 규칙이 system prompt에 존재해야 한다.
    assert "한국어" in captured["system_prompt"]
    assert "IMPORTANT LANGUAGE RULE" in captured["system_prompt"]

    assert captured["purpose"] == "commit_analysis"
    assert result["summary"] == "로그인 오류를 수정했습니다."


def test_strip_author_email_line_keeps_other_lines():
    commit = _commit()
    commit_text = git_ingestion._build_commit_text(commit)

    stripped = git_ingestion._strip_author_email_line(commit_text)

    assert commit.author_email not in stripped
    assert commit.commit_hash in stripped
    assert commit.author_name in stripped
    assert commit.message in stripped
