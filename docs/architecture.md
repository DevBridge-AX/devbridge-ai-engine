# Architecture

`devbridge-ai-engine`은 사내 프로젝트 이해 지원 챗봇 시스템의 AI 백엔드(FastAPI)입니다.
이 문서는 `backend`(Spring Boot)와의 경계, 통신 방식, 핵심 데이터/처리 설계 원칙을
사람이 읽기 좋은 형태로 정리합니다. 규칙으로서의 요약은 [`CLAUDE.md`](../CLAUDE.md)와
[`.claude/rules/db-boundary.md`](../.claude/rules/db-boundary.md)를 참고하세요.

## 1. 전체 그림

```
                 SSE / REST
 [Spring Boot backend] <----------------> [devbridge-ai-engine (FastAPI)]
        |                                          |
        |  USERS, WORKSPACES, TASKS,               |  KNOWLEDGE_DOCUMENTS, DATA_SOURCES,
        |  CHAT_SESSIONS, CHAT_MESSAGES,            |  DATABASE_SCHEMAS, GIT_COMMITS,
        |  NOTIFICATIONS, MESSAGE_CITATIONS,        |  document_chunks
        |  OWNER_CONFIRMATIONS ...                  |
        v                                          v
   [Spring 도메인 MySQL 스키마]            [AI 도메인 MySQL 스키마] + [벡터스토어(Chroma/FAISS, 파일)]
```

- 두 서비스는 같은 MySQL 인스턴스를 공유할 수 있지만, **스키마/계정이 분리**되어
  있고 테이블 소유권은 위와 같이 명확히 나뉩니다.
- 벡터스토어는 별도 서버 없이 파일 기반으로 동작하며, 온디바이스 전환 시 해당
  경로(`VECTOR_STORE_PATH`)를 그대로 이동하면 됩니다.

## 2. 테이블 소유권 경계

### FastAPI(이 레포)가 소유

| 테이블 | 설명 |
| --- | --- |
| `KNOWLEDGE_DOCUMENTS` | 인덱싱 대상 문서 메타데이터 |
| `DATA_SOURCES` | 인덱싱 데이터 소스 |
| `DATABASE_SCHEMAS` | 인덱싱된 DB 스키마 정보 |
| `GIT_COMMITS` | Git 커밋 메타데이터 (`author_id`는 Spring `USERS.id`로 사전 매핑됨) |
| `document_chunks` | RAG 청크 (원문 + 임베딩 + 모델 버전) |

### Spring(`backend` 레포)이 소유 — FastAPI는 접근 금지

`USERS`, `WORKSPACES`, `TASKS`, `CHAT_SESSIONS`, `CHAT_MESSAGES`, `NOTIFICATIONS`,
`MESSAGE_CITATIONS`, `OWNER_CONFIRMATIONS` 및 그 외 모든 비즈니스 도메인 테이블.

이 경계는 코드 리뷰 시 가장 우선적으로 확인해야 하는 항목입니다. 자세한 체크리스트는
[`.claude/rules/db-boundary.md`](../.claude/rules/db-boundary.md) 참고.

## 3. 통신 원칙

### Stateless 추론 엔진

FastAPI는 자체적으로 대화 상태를 들고 있지 않습니다. `/chat` 호출마다 Spring이
다음을 payload로 전달합니다:

- 대화 히스토리 (이전 턴들의 user/assistant 메시지)
- `workspace_id`
- `user_id`
- `role` (직무 — 페르소나 선택에 사용)

### SSE 스트리밍

`/chat` 류 엔드포인트는 `StreamingResponse`를 사용해 토큰 단위로 답변을
스트리밍합니다.

### 응답 payload

```json
{
  "answer_stream": "...",
  "citations": [
    {"document_id": 1, "chunk_id": 10, "similarity_score": 0.82, "source_type": "code"}
  ],
  "similarity_scores": [0.82, 0.77],
  "is_groundable": true,
  "confidence": 0.91,
  "suggested_owner_id": 42,
  "token_usage": {"prompt_tokens": 512, "completion_tokens": 128}
}
```

Spring은 이 응답을 받아:

- `CHAT_MESSAGES`, `MESSAGE_CITATIONS`에 저장
- `is_groundable` / `confidence`를 임계치와 비교해 UC-04 알림(`NOTIFICATIONS`)
  트리거 여부 결정

**FastAPI는 값만 계산해서 반환하고, 저장/임계치 판단/알림은 모두 Spring의
책임입니다.**

### GIT_COMMITS.author_id 매핑

```
[인덱싱 시점]
  Git push 이벤트 -> git_ingestion.py
    -> Spring 사용자조회 API 호출 (Git 계정 -> USERS.id)
    -> GIT_COMMITS.author_id = USERS.id 로 저장

[채팅 시점]
  retriever가 GIT_COMMITS를 citation으로 반환
    -> author_id를 추가 조회 없이 그대로 응답에 포함
```

## 4. 핵심 데이터/처리 설계

### 4.1 document_chunks: 원문과 임베딩의 분리

```
document_chunks
├── content                   (원문, 불변, source of truth)
├── embedding                 (벡터, derived — 재생성 가능)
├── embedding_model            (예: placeholder-embedding-model)
└── embedding_model_version    (예: v1)
```

임베딩 모델을 교체/업그레이드할 때는:

1. `content`는 그대로 유지
2. 새 모델로 `embedding`을 재생성
3. `embedding_model` / `embedding_model_version`을 갱신

이를 통해 임베딩 모델 변경이 원문 데이터 무결성에 영향을 주지 않습니다.

### 4.2 멀티턴 처리

```
turn 1   : user_query ────────────────────────► retriever
turn 2+  : user_query + conversation_history ─► query_rewriter ─► rewritten_query ─► retriever
```

- `query_rewriter`는 turn 1에는 호출되지 않습니다.
- turn 2+에서는 대화 히스토리를 참고해 검색에 적합한 독립 쿼리로 재구성합니다.
- query_rewriter를 포함한 모든 LLM 호출은 `provider.py`의 추상 인터페이스를
  통해서만 이루어지며, 구체 모델은 현재 placeholder입니다.

### 4.3 페르소나 변환 (6종)

대상 역할: 기획자, 개발자, QA, 디자이너, 운영자, 신규투입자

각 역할별 시스템 프롬프트는 다음 동적 변수를 채워 렌더링됩니다:

- `retrieved_context` — RAG로 검색된 문서/코드/Git 컨텍스트
- `conversation_history` — Spring이 전달한 대화 히스토리
- `user_preference_summary` — 아래 4.4 참고

### 4.4 user_preference_summary 격리

```
key = (user_id, workspace_id)
```

한 사용자가 여러 워크스페이스에 속할 수 있으므로, preference summary는 반드시
`(user_id, workspace_id)` 복합키로 조회/저장되어야 하며, 워크스페이스 간 데이터
혼입이 발생하면 안 됩니다.

### 4.5 신뢰도/그라운딩 판정

```
similarity_scores (retriever, 1차 필터)
        +
is_groundable / confidence (LLM structured output, 최종 판단)
        ↓
   grounding.assess_grounding()
        ↓
   { is_groundable, confidence }  -> 응답 payload로 반환
```

임계치 비교 및 UC-04 알림 트리거는 Spring의 책임이며, 이 레포는 산출된 값만
반환합니다.

### 4.6 LoRA 학습데이터 export

`pipelines/export_training_data.py`는 워크스페이스 단위로 격리된 JSONL을
생성합니다.

```json
{
  "original_question": "...",
  "target_role": "developer",
  "final_answer": "...",
  "source": "...",
  "prompt_version": "...",
  "is_faq": false,
  "owner_verified": true
}
```

- PII 스크러빙 포함
- `dataset_version` 기록
- `is_faq` / `owner_verified` 등 `OWNER_CONFIRMATIONS` 관련 값은 Spring이 전달한
  데이터를 기준으로 채우며, 이 레포가 해당 테이블을 직접 조회하지 않습니다.

## 5. 1차 구현 범위

**포함**

- RAG 기반 문서/코드/Git 검색 질의응답
- 멀티턴 컨텍스트 처리
- 직무별 페르소나 변환 (6종)
- 신뢰도/그라운딩 판정
- 문서·Git 인덱싱 파이프라인
- LoRA용 학습데이터 누적 로깅

**제외 (2차 이후)**

- NL2SQL (자연어 → SQL 조회)
- 자체 모델 서빙 / LoRA 적용
