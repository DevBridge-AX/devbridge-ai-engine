"""
/chat 라이브 E2E 검증 (실 GMS API 호출, 과금 발생).

RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live로 실행합니다
(docs/ai-api-usage.md "라이브 검증 실행법" 참고). tests/live/fixtures/corpus로 시드된
워크스페이스를 대상으로 SSE 계약, 그라운딩 판정, 멀티턴 rewrite, 페르소나 6종을
검증합니다. 품질(답변 내용) 자체는 판정하지 않고, 페르소나 답변 앞 200자를
`live_run_recorder`를 통해 세션 종료 리포트(data/live_runs/*.json)에 남겨 수동
검토합니다.
"""

import pytest

from app.config import get_settings
from app.core import chat_pipeline
from app.core.llm.persona_prompts import PersonaRole
from app.schemas.chat import ChatDoneEvent
from tests.live.sse import parse_sse

pytestmark = pytest.mark.live


def _request_body(*, workspace_id: str, **overrides) -> dict:
    payload = {
        "session_id": "live-session",
        "content": "배포 절차 알려줘",
        "conversation_history": [],
        "workspace_id": workspace_id,
        "user_id": "live-user",
        "role": "developer",
        **overrides,
    }
    return payload


def _tokens_and_done(events: list[tuple[str, dict]]) -> tuple[str, dict]:
    """events에서 token 텍스트를 이어붙인 answer와 마지막 done payload를 반환합니다."""
    answer = "".join(data.get("text", "") for name, data in events if name == "token")
    assert events, "SSE 이벤트가 비어 있습니다"
    assert events[-1][0] == "done"
    return answer, events[-1][1]


def test_turn1_sse_contract(client, seeded_workspace):
    """turn 1: token >= 1개 → done 1회, error 0회. done이 ChatDoneEvent 계약을 만족."""
    _, workspace_id = seeded_workspace

    resp = client.post(
        "/api/chat/", json=_request_body(workspace_id=workspace_id, content="배포 절차 알려줘")
    )
    assert resp.status_code == 200

    events = parse_sse(resp.text)
    answer, done_data = _tokens_and_done(events)

    assert answer != ""
    assert sum(1 for name, _ in events if name == "done") == 1
    assert sum(1 for name, _ in events if name == "error") == 0

    done = ChatDoneEvent.model_validate(done_data)
    assert done.is_groundable is True
    assert done.prompt_version == chat_pipeline.PROMPT_VERSION

    for citation in done.citations:
        assert citation.source_type in ("document", "git_commit", "db_schema")
        assert isinstance(citation.chunk_id, int)
        assert 0.0 <= citation.similarity_score <= 1.0

    assert done.token_usage.rewrite is None
    assert done.token_usage.main.prompt_tokens > 0
    assert done.token_usage.main.completion_tokens > 0


def test_out_of_corpus_question_not_groundable(client, seeded_workspace):
    """코퍼스에 없는 사내 고유 질문 → token 0개, is_groundable=False, main 토큰 0/0."""
    _, workspace_id = seeded_workspace

    resp = client.post(
        "/api/chat/",
        json=_request_body(
            workspace_id=workspace_id,
            content="우리 회사 급여 정산 시스템의 4대보험 공제율 계산 로직은 어떻게 되어있어?",
        ),
    )
    assert resp.status_code == 200

    events = parse_sse(resp.text)
    assert sum(1 for name, _ in events if name == "token") == 0
    assert events[-1][0] == "done"

    done = ChatDoneEvent.model_validate(events[-1][1])
    assert done.is_groundable is False
    assert done.token_usage.main.prompt_tokens == 0
    assert done.token_usage.main.completion_tokens == 0


def test_multiturn_rewrite_path(client, seeded_workspace):
    """turn 2+ "그거 소스는 어디 있어?" → rewrite 호출됨 + 결제 API 문서가 인용됨."""
    _, workspace_id = seeded_workspace
    history = [
        {"role": "user", "content": "결제 API 문서 보여줘"},
        {"role": "assistant", "content": "결제 API는 TossPayments 게이트웨이를 사용합니다."},
    ]

    resp = client.post(
        "/api/chat/",
        json=_request_body(
            workspace_id=workspace_id,
            content="그거 소스는 어디 있어?",
            conversation_history=history,
        ),
    )
    assert resp.status_code == 200

    events = parse_sse(resp.text)
    _, done_data = _tokens_and_done(events)
    done = ChatDoneEvent.model_validate(done_data)

    assert done.token_usage.rewrite is not None
    assert done.token_usage.rewrite.model == get_settings().rewrite_model
    assert any(c.source_type == "document" for c in done.citations)


@pytest.mark.parametrize("role", list(PersonaRole))
def test_persona_smoke(client, seeded_workspace, live_run_recorder, role):
    """6종 페르소나 모두 SSE 계약을 통과하고 답변이 비어 있지 않은지 확인(품질 판정 없음)."""
    _, workspace_id = seeded_workspace

    resp = client.post(
        "/api/chat/",
        json=_request_body(workspace_id=workspace_id, content="배포 절차 알려줘", role=role.value),
    )
    assert resp.status_code == 200

    events = parse_sse(resp.text)
    answer, done_data = _tokens_and_done(events)
    assert sum(1 for name, _ in events if name == "error") == 0

    ChatDoneEvent.model_validate(done_data)
    assert answer.strip() != ""

    live_run_recorder[role.value] = answer[:200]
