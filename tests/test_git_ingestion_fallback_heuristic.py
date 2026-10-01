"""
커밋 분석 fallback 휴리스틱 유닛 테스트.

commit_text 메타데이터 헤더(author_name/author_email)의 "author"가 security 키워드
"auth"에 매칭되어 모든 커밋이 security로 분류되던 회귀를 검증합니다.
"""

from datetime import datetime, timezone

from app.pipelines.git_ingestion import (
    _build_commit_text,
    _fallback_analyze_commit,
)
from app.schemas.ingestion import CommitData


def _commit(message: str, diff: str) -> CommitData:
    return CommitData(
        commit_hash="a" * 40,
        short_hash="aaaaaaa",
        branch_name="main",
        author_name="홍길동",
        author_email="hong@example.com",
        committed_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        message=message,
        diff=diff,
    )


def _analyze(commit: CommitData) -> dict:
    return _fallback_analyze_commit(commit, _build_commit_text(commit))


def test_header_author_does_not_force_security():
    commit = _commit("chore: Dockerfile 베이스 이미지 버전 업데이트", "FROM python:3.11-slim")
    assert _analyze(commit)["impact_area"] == "infra"


def test_frontend_commit_classified_by_body():
    commit = _commit("feat: 로그인 폼 검증 추가", "src/pages/Login.tsx component")
    assert _analyze(commit)["impact_area"] == "frontend"


def test_security_keyword_in_body_still_detected():
    commit = _commit("fix: JWT 만료 검증 누락 수정", "jwt.verify(token)")
    assert _analyze(commit)["impact_area"] == "security"
