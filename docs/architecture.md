# Architecture

`devbridge-ai-engine`은 사내 프로젝트 이해 지원 챗봇 시스템의 AI 백엔드(FastAPI)입니다.
이 문서는 `backend`(Spring Boot)와의 경계, 통신 방식, 핵심 데이터/처리 설계 원칙을
사람이 읽기 좋은 형태로 정리합니다. 규칙으로서의 요약은 [`CLAUDE.md`](../CLAUDE.md)와
[`.claude/rules/db-boundary.md`](../.claude/rules/db-boundary.md)를 참고하세요.

> **워크스페이스 내부 문서/데이터소스 단위 접근 제어**(task 기반 스코핑, 민감도
> 플래그 등, 아직 미구현)는 [`access-control.md`](access-control.md)에 별도
> 설계 문서로 정리되어 있습니다.

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
- Spring → FastAPI 요청은 `X-Internal-Api-Key` 헤더로 인증되며, FastAPI는 외부에
  노출되지 않는 내부망에서만 접근 가능합니다 (3장 참고).

## 2. 테이블 소유권 경계

### FastAPI(이 레포)가 소유

| 테이블 | 설명 |
| --- | --- |
| `KNOWLEDGE_DOCUMENTS` | 인덱싱 대상 문서 메타데이터 |
| `DATA_SOURCES` | 인덱싱 데이터 소스 |
| `DATABASE_SCHEMAS` | 인덱싱된 DB 스키마 정보 |
| `GIT_COMMITS` | Git 커밋 메타데이터 (`author_id`는 Spring `USERS.id`로 사전 매핑됨) |
| `document_chunks` | RAG 청크 (원문 + 임베딩 + 모델 버전) |
| `usage_logs` | 워크스페이스별 일 단위 임베딩 토큰 사용량 집계 |

### Spring(`backend` 레포)이 소유 — FastAPI는 접근 금지

`USERS`, `WORKSPACES`, `TASKS`, `CHAT_SESSIONS`, `CHAT_MESSAGES`, `NOTIFICATIONS`,
`MESSAGE_CITATIONS`, `OWNER_CONFIRMATIONS` 및 그 외 모든 비즈니스 도메인 테이블.

이 경계는 코드 리뷰 시 가장 우선적으로 확인해야 하는 항목입니다. 자세한 체크리스트는
[`.claude/rules/db-boundary.md`](../.claude/rules/db-boundary.md) 참고.

## 3. 통신 원칙

### 인증

Spring → FastAPI의 모든 요청에는 `X-Internal-Api-Key` 헤더가 포함되어야 하며,
`.env`의 `INTERNAL_API_KEY`와 일치하지 않으면 401을 반환합니다
(`app/core/security.py`). 이 레포는 최종 사용자 인증을 직접 수행하지 않으며,
Spring이 이미 인증/인가를 마친 뒤 전달하는 `workspace_id` / `user_id` / `role`을
신뢰합니다. FastAPI는 외부에 직접 노출되지 않는 내부망(Docker 네트워크 등)에서만
접근 가능해야 합니다.

### Stateless 추론 엔진

FastAPI는 자체적으로 대화 상태를 들고 있지 않습니다. `/chat` 호출마다 Spring이
다음을 payload로 전달합니다:

- 대화 히스토리 (이전 턴들의 user/assistant 메시지)
- `workspace_id`
- `user_id`
- `role` (직무 — 페르소나 선택에 사용)

### SSE 스트리밍 — 이벤트 2종

`/chat` 류 엔드포인트는 `StreamingResponse`로 **두 종류의 이벤트**를 보냅니다.
답변 텍스트(스트리밍)와 메타데이터(스트림 종료 시 1회)를 분리합니다 —
`answer_stream`처럼 답변 전체를 하나의 JSON 필드로 묶어 보내지 않습니다.

```
event: token   (반복)
data: {"text": "..."}

event: done    (스트림 종료 시 1회)
data: { ... 아래 "응답 payload" 참고 ... }
```

### 응답 payload (`done` 이벤트)

```json
{
  "citations": [
    {
      "source_type": "document",
      "source_id": 1,
      "chunk_id": 10,
      "title": "결제 API 문서",
      "similarity_score": 0.82
    }
  ],
  "is_groundable": true,
  "confidence": 0.91,
  "suggested_owner_id": 42,
  "prompt_version": "persona-v1",
  "token_usage": {
    "main": {"model": "claude-sonnet-4-6", "prompt_tokens": 512, "completion_tokens": 128},
    "rewrite": {"model": "claude-haiku-4-5-20251001", "prompt_tokens": 80, "completion_tokens": 20},
    "context_truncated": false
  }
}
```

- `citations[].source_type`은 `document` / `git_commit` / `db_schema` 중 하나이며,
  `source_id`는 해당 테이블(`KNOWLEDGE_DOCUMENTS`/`GIT_COMMITS`/`DATABASE_SCHEMAS`)의
  PK, `chunk_id`는 `document_chunks.id`입니다. 이 구조는 `MESSAGE_CITATIONS`의
  `source_type`/`source_id`/`vector_chunk_id`/`similarity_score`에 1:1
  매핑됩니다. `title`은 Spring/프론트 표시용 라벨로, FastAPI가 인덱싱 시점에
  확보한 정보를 이용해 응답 시점에 채워줍니다.
- `prompt_version`은 사용된 persona 템플릿 버전이며 `CHAT_MESSAGES.prompt_version`에
  저장됩니다.
- `token_usage.context_truncated`는 입력 토큰이 한도에 근접해 RAG 컨텍스트/히스토리를
  줄였는지를 나타내는 모니터링용 플래그입니다.

Spring은 `done` payload를 받아:

- `CHAT_MESSAGES`, `MESSAGE_CITATIONS`에 저장
- `is_groundable` / `confidence`를 임계치와 비교해 UC-04 알림(`NOTIFICATIONS`)
  트리거 여부 결정
- `token_usage`의 모델별 raw 토큰 수치에 자체 Credit 단가표를 적용해
  `estimated_cost`를 계산·저장

**FastAPI는 값(및 raw 토큰 수치)만 계산해서 반환하고, 저장/임계치 판단/알림/단가
적용은 모두 Spring의 책임입니다.** turn 1에서는 `rewrite`가 호출되지 않으므로
`token_usage.rewrite`는 `null`입니다.

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
├── source_type                (document / git_commit / db_schema)
├── source_id                  (위 source_type에 해당하는 테이블의 PK)
├── content                    (원문, 불변, source of truth)
├── embedding                  (벡터, derived — 재생성 가능)
├── embedding_model            (현재 기본값: text-embedding-3-large)
├── embedding_model_version    (예: v1)
├── vector_id                  (Chroma/FAISS 내 실제 벡터 참조)
└── workspace_id               (워크스페이스별 격리)
```

`source_type` + `source_id`는 단일 `document_id` FK가 아니라, 이 청크가
`KNOWLEDGE_DOCUMENTS` / `GIT_COMMITS` / `DATABASE_SCHEMAS` 중 어디에서 왔는지를
표현합니다. `/chat` 응답의 `citations[]`도 동일한 `source_type`/`source_id`
구조를 사용합니다 (3장 참고).

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
  통해서만 이루어집니다. 현재 기본 설정(`.env`)은 메인 답변 생성/페르소나
  변환/그라운딩 판정에 `claude-sonnet-4-6`, 쿼리 재구성에
  `claude-haiku-4-5-20251001`을 사용합니다. 모델 교체는 `.env` 값만 변경하면
  됩니다.

### 4.3 페르소나 변환 (6종)

대상 역할: 기획자, 개발자, QA, 디자이너, 운영자, 신규투입자

각 역할별 시스템 프롬프트는 다음 동적 변수를 채워 렌더링됩니다 (1차 범위):

- `retrieved_context` — RAG로 검색된 문서/코드/Git 컨텍스트
- `conversation_history` — Spring이 전달한 대화 히스토리

직무별 번역(예: 개발자 산출물을 디자이너가 이해하기 쉽게)은 `role` 필드에 따른
6종 템플릿 선택만으로 처리됩니다. `user_preference_summary`(같은 역할 내
개인별 스타일 차이)는 2차 변수로, 1차 템플릿에는 포함하지 않습니다.

### 4.4 user_preference_summary (2차 예정 — 보류)

같은 직무(role) 내에서도 개인별 코드 스타일/표현 선호가 다를 수 있다는 점을
반영하는 레이어이지만, 1차에는 구현하지 않습니다. 사용 데이터가 쌓인 뒤 2차에서
재검토합니다. 도입 시 격리 키:

```
key = (user_id, workspace_id)
```

한 사용자가 여러 워크스페이스에 속할 수 있으므로, 도입 시 preference summary는
반드시 `(user_id, workspace_id)` 복합키로 조회/저장되어야 하며, 워크스페이스 간
데이터 혼입이 발생하면 안 됩니다.

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

### 4.7 사용량 로깅과 비용 추적

채팅 비용과 인덱싱 비용은 추적 방식이 다릅니다.

- **채팅 비용**: `/chat` 응답의 `token_usage.main` / `token_usage.rewrite`에
  모델별 raw 토큰 수치를 담아 반환합니다. Credit 단가표는 Spring이 관리하며,
  `estimated_cost` 계산·저장은 Spring의 책임입니다.
- **인덱싱(임베딩) 비용**: 채팅 메시지와 무관한 백그라운드 비용이므로,
  `usage_logs` 테이블(`workspace_id`, `date`, `embedding_model`,
  `embedding_tokens`)에 일 단위로 누적합니다. `app/api/usage.py`의
  `/usage/summary?workspace_id=`를 통해 Spring에 집계값을 제공하며, 이 레포는
  단가를 알지 못하고 토큰 수치만 반환합니다.

## 5. DB 마이그레이션 (Alembic)

AI 도메인 스키마(`devbridge_ai`)는 Alembic으로 관리합니다.

### 초기 설정

```bash
# devbridge_ai 스키마 및 ai_engine_user 계정은 별도 MySQL 초기화 스크립트로 생성해야 합니다.
# CREATE DATABASE devbridge_ai;
# CREATE USER 'ai_engine_user'@'%' IDENTIFIED BY '...';
# GRANT ALL PRIVILEGES ON devbridge_ai.* TO 'ai_engine_user'@'%';
```

### 주요 명령어

```bash
# 마이그레이션 적용 (최신 리비전으로)
alembic upgrade head

# 현재 리비전 확인
alembic current

# 새 리비전 생성 (models.py 변경 후)
alembic revision --autogenerate -m "describe_change"

# 한 단계 롤백
alembic downgrade -1
```

### DB URL 주입 방식

`alembic/env.py`가 `app/config.py`의 `DATABASE_URL`을 읽어 동적으로 주입합니다.
`alembic.ini`의 `sqlalchemy.url`은 비워두며, `.env` 파일 값이 우선합니다.

### autogenerate 범위

`Base.metadata`(`app/db/models.py`)에 정의된 AI 도메인 테이블만 autogenerate 대상입니다.
Spring 소유 테이블(`USERS`, `WORKSPACES`, `CHAT_MESSAGES` 등)은 이 레포의 models.py에
정의되지 않으므로 리비전에 절대 포함되지 않습니다.

## 6. 1차 구현 범위

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
