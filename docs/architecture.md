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

- `retriever` 이전에 시맨틱 캐시 조회가 끼어들 수 있습니다(4.8 참고, 기본 off).
- `query_rewriter`는 turn 1에는 호출되지 않습니다.
- turn 2+에서는 대화 히스토리를 참고해 검색에 적합한 독립 쿼리로 재구성합니다.
- query_rewriter를 포함한 모든 LLM 호출은 `provider.py`의 추상 인터페이스를
  통해서만 이루어집니다. 현재 기본 설정(`.env`)은 메인 답변 생성/페르소나
  변환에 `claude-sonnet-4-6`, 그라운딩 판정에 `gemini-2.5-flash-lite`
  (`GROUNDING_MODEL`), 쿼리 재구성에 `claude-haiku-4-5-20251001`을 사용합니다.
  모델 교체는 `.env` 값만 변경하면 됩니다.

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

2차 판정(LLM) 관련 설정(`.env`, 기본값은 v2-strict / top-3이며 `v1` / `5`로 설정하면 이전 동작으로
복원됩니다. 근거: `docs/grounding-eval.md` 2회차):

- `GROUNDING_PROMPT_VERSION=v2-strict` — 판정 시스템 프롬프트 버전(`v1` | `v2-strict`,
  `app/core/rag/grounding_prompts.py`). 잘못된 값은 기동 시 검증 오류가 됩니다.
- `GROUNDING_JUDGE_TOP_K=3` — 판정 프롬프트에 넣는 청크 수(1 이상, retrieve
  top_k와 별개).
- `GROUNDING_JUDGE_MAX_CHUNK_CHARS=0` — 0이면 제한 없음, 양수면 판정 프롬프트의
  청크 content를 해당 길이로 자르고 `…(생략)`을 붙입니다.

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

- 입력은 Spring이 `POST /api/training-data/export` payload(`workspace_id`,
  `dataset_version`, `records[]`)로 전달합니다. 출력은
  `{TRAINING_DATA_DIR}/{workspace_id}/{dataset_version}.jsonl`이며 동일 버전은
  덮어씁니다. 응답: `path`, `record_count`, `scrubbed_field_count`.
- PII 스크러빙 포함 (`app/core/utils/pii.py`: 이메일/전화/주민번호/IPv4/URL, 이름은 제외).
  `user_id`는 받지도 기록하지도 않습니다.
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

### 4.8 시맨틱 캐시

의미가 같은 질문이 이미 답변된 경우 그라운딩과 메인 LLM 호출을 건너뛰고 저장된
답변을 재생합니다. `SEMANTIC_CACHE_ENABLED=false`(기본)이면 기존 흐름·호출 횟수·
임베딩 호출 지점이 그대로입니다.

```
turn 1/2+ : rewritten_query ─► embed ─► cache.lookup ─┬─ hit  ─► 접근 재검증 ─► token* ─► done
                                                      └─ miss ─► retriever(임베딩 재사용) ─► grounding ─► main LLM
                                                                   └─ 첫 턴 + 정상 판정이면 cache.store
```

- **네임스페이스**: `(workspace_id, role, access_fingerprint, prompt_version, main_model)`.
  `access_fingerprint`는 `(정렬된 accessible_task_ids, can_view_restricted)`이며,
  `None`(제한 없음)과 `[]`(공용만)은 서로 다른 값입니다. 하나라도 다르면 hit하지 않습니다.
- **조회**: `rewritten_query` 임베딩과 네임스페이스 내 엔트리의 코사인 유사도가
  `SEMANTIC_CACHE_THRESHOLD` 이상인 최선의 엔트리를 hit로 봅니다. 이 임베딩은 retriever에
  `query_embedding`으로 전달되어 miss 시 중복 호출되지 않습니다.
- **저장 조건**: 첫 턴(대화 히스토리 없음) + `is_groundable=True` + `fallback_reason=None`
  (정상 판정) + 비어있지 않은 답변. 스트림이 정상 종료된 경우에만 저장합니다.
- **hit 안전장치**: 캐시된 `citations[].chunk_id`가 현재 요청의 접근 범위에 여전히
  속하는지 `_allowed_chunk_ids`로 재검증하고, 실패하면 엔트리를 제거한 뒤 miss로 처리합니다.
  TTL(`SEMANTIC_CACHE_TTL_SECONDS`)이 지난 엔트리는 조회 중 제거됩니다.
- **무효화**: 문서·Git·담당자 답변 인덱싱이 성공하면 BM25 무효화와 함께
  `invalidate_workspace(workspace_id)`로 해당 워크스페이스의 캐시를 비웁니다.
  워크스페이스당 엔트리 수는 `SEMANTIC_CACHE_MAX_ENTRIES`로 제한되며 LRU로 제거합니다.
- **응답 형식**: `token`/`done` 이벤트 구조는 변하지 않습니다. 저장된 답변을 약 40자
  조각의 `token` 이벤트로 재생하고, `done`의 `token_usage.main`은 모델명만 채우고
  `prompt_tokens`/`completion_tokens`는 0, `grounding`은 `null`입니다. turn 2+의
  `rewrite` 사용량은 실제 값을 그대로 담습니다.
- **관측성**: `chat_metrics`에 `cache_enabled`, `cache_hit`, `cache_similarity`,
  `cache_lookup_ms`가 기록되며 hit 시 `grounding_stage="cache"`입니다.
  LLM 재검증이 켜져 있으면 `cache_verify_ms`(재검증 호출 지연)와 `cache_verify_result`
  (`yes`/`no`/`invalid`/`error`/`timeout`)도 hit·miss 양쪽 레코드에 기록되며, 재검증을
  호출하지 않은 요청은 둘 다 `null`입니다.
  `scripts/metrics_report.py`가 hit율, hit/miss별 `total_ms`·`ttft_ms` p50/p95, 재검증 결과 분포·지연·재검증 경유 hit 비율을 보여줍니다.
- **멀티 워커 제한**: 프로세스 메모리 캐시이므로 워커마다 독립이며 인덱싱 무효화가 다른
  워커에는 전파되지 않습니다. 멀티 워커 운영 시 TTL이 stale 허용 상한이 됩니다.
- **stale 저장 방지(generation)**: `SemanticCache`는 워크스페이스별 generation 카운터를
  가지며 `invalidate_workspace`마다 1 증가합니다. 파이프라인은 조회 직전에 값을 잡아 두고,
  메인 스트림이 끝난 뒤 저장 직전에 값이 달라졌으면(스트림 도중 인덱싱 무효화) 저장하지
  않습니다. 이전 컨텍스트로 만든 답변이 무효화 이후 TTL 동안 재생되는 것을 막습니다.
- **한계**:
  - 조회는 순수 Python 선형 스캔입니다. 네임스페이스당 약 500건·3072차원이면 조회 1회에
    약 50ms 동기 블로킹이 생길 수 있습니다(엔트리가 적을 때만 1ms 미만).
  - turn 2+는 `rewritten_query`만으로 첫 턴 답변을 재생하므로, 대화 히스토리에 의존하는
    답변이 필요한 경우에는 적합하지 않습니다. 또한 네임스페이스에 `user_id`가 없어 같은
    role·접근 권한을 가진 사용자 사이에서 답변이 재생됩니다. 개인정보가 답변에 반영되는
    질문이 있다면 주의해야 합니다.
  - 플래그가 켜지면 `embed_ms`가 `retrieve_ms` 밖에서 측정되고, BM25 검색과 임베딩의
    병렬 실행이 사라집니다(임베딩을 먼저 기다림). 따라서 off 상태와 `retrieve_ms`를
    직접 비교할 수 없습니다.
- **LLM 재검증(선택, 기본 off)**: 임베딩 유사도 분포가 압축되어 임계치만으로는 hit와 오적중을
  가르기 어려우므로, `SEMANTIC_CACHE_VERIFY_ENABLED=true`이면 후보 구간을 경량 LLM으로 확인합니다.
  - 유사도 ≥ `SEMANTIC_CACHE_THRESHOLD`: 기존처럼 검증 없이 즉시 hit.
  - `SEMANTIC_CACHE_CANDIDATE_THRESHOLD` ≤ 유사도 < `SEMANTIC_CACHE_THRESHOLD`: 최선 후보 **1건만**
    `app/core/cache/verifier.py`의 `verify_same_question`(REWRITE_MODEL, max_tokens 16, 타임아웃
    `SEMANTIC_CACHE_VERIFY_TIMEOUT_SECONDS`)으로 검증하고, 정확히 `YES`일 때만 hit입니다
    (비용 상한: 요청당 최대 1회). 접근 재검증을 먼저 하고 통과한 경우에만 검증을 호출합니다.
  - `NO`/예외/타임아웃/해석 불가 응답은 모두 miss(fail-closed)로 일반 경로를 탑니다. 검증에 실패한
    후보는 `hits`/LRU가 갱신되지 않습니다(`lookup(min_similarity=...)`은 후보 구간에서 부수효과가
    없고, 검증 통과 시 `mark_hit`으로 확정).
  - `CANDIDATE_THRESHOLD`가 `THRESHOLD`보다 크면 후보 구간이 없는 것으로 보고 재검증을 건너뜁니다
    (기동 오류 없이 경고 로그만 남김).
  - 관측: 검증 LLM 호출은 `llm_calls` 이벤트에 `purpose="cache_verify"`로 기록됩니다.
    `chat_metrics`에는 아직 검증 전용 필드가 없습니다(별도 작업).
  - **Spring 협의 항목**: 응답 계약은 변하지 않으므로 검증 호출의 토큰은 `done.token_usage`에
    포함되지 않습니다(`rewrite`/`main`/`grounding`만 보고). 과금 집계에 포함하려면 `token_usage`
    확장 여부를 Spring과 협의해야 합니다.
  - 꺼져 있으면 동작·호출 횟수가 기존과 완전히 같습니다.
- **설정**: `SEMANTIC_CACHE_ENABLED`(기본 false), `SEMANTIC_CACHE_THRESHOLD`(0.95),
  `SEMANTIC_CACHE_TTL_SECONDS`(3600), `SEMANTIC_CACHE_MAX_ENTRIES`(500, 워크스페이스당),
  `SEMANTIC_CACHE_VERIFY_ENABLED`(false), `SEMANTIC_CACHE_CANDIDATE_THRESHOLD`(0.86),
  `SEMANTIC_CACHE_VERIFY_TIMEOUT_SECONDS`(3.0).

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
