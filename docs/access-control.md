# RAG 데이터 접근 경계 설계 (초안 — 미구현)

> **상태**: 설계 문서만 존재, 코드에는 아직 반영되지 않았습니다. 실제 구현 전에
> `backend`(Spring) 쪽과 계약을 맞추는 논의가 필요합니다. 이 문서는 "왜 이 구조가
> 이 프로젝트에 맞는지" 근거를 남기는 것이 목적입니다.

## 1. 문제 정의

`workspace_id` 기준 격리는 이미 견고합니다. 그러나 **같은 워크스페이스 안에서는
문서/데이터소스 단위의 접근 제어가 전혀 없습니다.** 워크스페이스에 인사(HR)
문서, 급여 관련 DB 스키마 등 민감한 소스가 하나라도 인덱싱되면, 그 워크스페이스에
속한 모든 사용자가 챗봇 질의를 통해 해당 내용을 조회할 수 있습니다.

### 근거 (코드 확인 결과)

| 확인 지점 | 내용 |
| --- | --- |
| `app/db/vector_store.py:24-27` | Chroma 컬렉션을 `workspace_{workspace_id}`로 분리 → **워크스페이스 간 교차 조회는 구조적으로 불가능** (다른 컬렉션 자체를 조회하지 않음) |
| `app/core/rag/bm25_index.py:113-117` | BM25 인덱스도 `DocumentChunk.workspace_id == workspace_id`로만 필터링, 워크스페이스 단위로 캐시 |
| `app/core/rag/retriever.py:50-79` | `retrieve(query, workspace_id, db, top_k)` — **`user_id`/`role`을 인자로 받지 않음**. 검색 조건은 오직 `workspace_id` |
| `app/core/chat_pipeline.py:60` | `retriever.retrieve(rewritten_query, request.workspace_id, db, ...)` — `request.user_id`를 리트리벌에 전달하지 않음 |
| `app/schemas/chat.py` (`ChatRequest.role: PersonaRole`) | `PersonaRole` = 기획자/개발자/QA/디자이너/운영자/신규투입자. **어투(persona) 선택용이며 권한(RBAC)이 아님.** Admin/Member/Viewer 같은 개념과 무관 |
| `app/db/models.py` (`KnowledgeDocument`, `DataSource`, `DatabaseSchema`, `DocumentChunk`) | 문서/소스/청크 어디에도 가시성(ACL) 컬럼 없음 |
| `app/schemas/ingestion.py` (`DocumentIngestionRequest`) | `workspace_id`, `task_id` 등은 받지만 "누가 볼 수 있는지"는 전달받지 않음 |

**결론**: 워크스페이스 멤버십(Spring이 관리하는 RBAC)만으로는 부족합니다. 워크스페이스
*내부*의 문서/데이터소스 단위 접근 제어 계층이 하나 더 필요합니다.

## 2. 설계를 제약하는 조건

이 레포의 구조상 아래 조건을 반드시 지켜야 합니다 (`CLAUDE.md`, `.claude/rules/db-boundary.md`).

1. **AI 엔진은 Spring의 `USERS`/`TASKS`/RBAC 테이블에 직접 접근할 수 없습니다.** 권한
   판단(누가 어떤 task/문서를 볼 수 있는지)은 여전히 Spring의 책임이어야 합니다.
2. **Stateless 원칙** — AI 엔진은 매 요청 payload만 신뢰합니다. 즉 접근 범위는
   Spring이 매 `/chat` 호출마다 "계산해서" 넘겨줘야 하고, AI 엔진은 그 값을
   검색 단계에서 "강제"만 합니다.
3. **성능 제약** — BM25/벡터 인덱스는 워크스페이스 단위로 캐시됩니다
   (`bm25_index.py`의 TTL 캐시, Chroma 컬렉션). 사용자별/권한 조합별로 인덱스를
   쪼개는 방식은 조합 폭발이 일어나 비현실적입니다. → **필터링은 인덱스를
   쪼개는 방식이 아니라, 검색 결과를 사후 필터링하는 방식이어야 합니다.**
4. `PersonaRole`(`role`)은 이미 다른 의미(페르소나)로 쓰이고 있으므로, 접근 제어용
   필드는 **별도로 신설**해야 합니다 (`role` 재사용 금지).

## 3. 제안: 2계층 스코핑

문서와 데이터소스는 성격이 달라서 단일 ACL 모델보다 **용도별로 다른 스코핑 축**을
쓰는 것이 실제 구현 비용 대비 효과가 좋습니다.

### 3.1 Task 기반 스코핑 — 문서(`KNOWLEDGE_DOCUMENTS`)

`KnowledgeDocument.task_id`(`app/db/models.py:66`)가 **이미 존재**합니다. 지금은
아무 데도 안 쓰이고 있는데, 이걸 그대로 접근 제어 축으로 재사용하면 새 개념을
도입하지 않아도 됩니다.

- `task_id IS NULL` → 워크스페이스 전체 공개 문서 (지금과 동일하게 동작, 기본값)
- `task_id`가 있는 문서 → 해당 task에 배정된 사용자만 조회 가능

Spring은 이미 `TASKS`와 담당자 배정 정보를 갖고 있으므로 (`workspace/task` RBAC와
자연스럽게 맞물림), 매 `/chat` 요청마다 "이 `user_id`가 이 `workspace_id`에서
배정된 `task_id` 목록"을 계산해서 넘겨주기만 하면 됩니다.

### 3.2 민감도(sensitivity) 기반 스코핑 — 데이터소스(`DATA_SOURCES` / `DATABASE_SCHEMAS`)

DB 스키마 커넥터, Git 저장소처럼 **task와 무관하게 워크스페이스 레벨로 연결되는
소스**(예: 인사 시스템 DB 커넥터)는 task 기반 스코핑이 자연스럽지 않습니다. 대신
소스 자체에 민감도 플래그를 둡니다.

- `DATA_SOURCES.sensitivity_level`: `normal`(기본) / `restricted`
- Spring은 워크스페이스 RBAC(Admin/Member/Viewer 등, 기존 유스케이스 명세에 이미
  존재하는 개념)를 기준으로 `can_view_restricted: bool`을 계산해 `/chat` 요청마다
  전달

이 경로는 특히 "인사 테이블"처럼 **커넥터 자체를 아예 일반 멤버에게 숨겨야 하는
경우**에 맞습니다.

### 3.3 왜 조인이 아니라 chunk 스냅샷인가

`document_chunks`는 `source_type` + `source_id`만 갖고 원본(`KNOWLEDGE_DOCUMENTS`
등)을 가리킵니다(`docs/architecture.md` 4.1절). 검색 시점마다 원본 테이블과 조인해서
`task_id`/`sensitivity_level`을 매번 다시 조회하면, 특히 BM25 사후 필터링 시
N+1 조회가 될 수 있습니다.

이 레포는 이미 `document_chunks.chunk_metadata`(JSON)에 "인덱싱 시점의 스냅샷"을
저장하는 패턴을 쓰고 있습니다 (`owner_answer_ingestion.py`가 `owner_employee_id`,
`owner_name`을 청크 메타데이터에 그대로 복사해 넣는 방식과 동일). 같은 패턴으로:

- 문서 인덱싱 시점에 `KNOWLEDGE_DOCUMENTS.task_id`를 `document_chunks`에 그대로
  복사해 저장 (예: `task_id` 컬럼 또는 `chunk_metadata.task_id`)
- DB 스키마 인덱싱 시점에 `DATA_SOURCES.sensitivity_level`을 마찬가지로 복사

이렇게 하면 검색 단계에서는 `document_chunks` 한 테이블만 보고 필터링할 수
있습니다. 대신 task 배정/민감도가 바뀌면 **재인덱싱 전까지는 반영되지 않는다**는
트레이드오프가 생기는데, 이는 `embedding_model_version`처럼 "인덱싱 시점 스냅샷"을
이미 신뢰하는 이 레포의 기존 설계 원칙과 일관됩니다.

## 4. 스키마 변경안 (참고용 — 아직 적용 안 함)

```
document_chunks
├── (기존 컬럼 유지)
├── task_id                 (nullable, KNOWLEDGE_DOCUMENTS 인덱싱 시 스냅샷)
└── sensitivity_level       (default 'normal', DATA_SOURCES 인덱싱 시 스냅샷)

DATA_SOURCES
└── sensitivity_level       (select: normal | restricted, default 'normal')
```

`KNOWLEDGE_DOCUMENTS.task_id`는 이미 있으므로 변경 불필요.

## 5. API 계약 변경안 (참고용 — 아직 적용 안 함)

```jsonc
// ChatRequest 추가 필드
{
  "accessible_task_ids": ["task-1", "task-2"],  // null = 제한 없음(워크스페이스 관리자 등)
  "can_view_restricted": false                   // Spring이 RBAC로 계산해 전달
}
```

기본값은 항상 "제한 없음"으로 두어, **Spring이 아직 이 필드를 보내지 않는 동안에는
지금과 동일하게 동작**하도록 하위호환을 유지해야 합니다 (Breaking change 없음).

## 6. retriever 필터링 순서 (참고용)

```
overfetch (vector + BM25, 현재 top_k * 3)
  → ACL 필터 (task_id/sensitivity_level 사후 필터링)
  → RRF 병합
  → top_k 반환
```

주의: ACL 필터링으로 후보가 많이 걸러지면 최종 결과가 `top_k`보다 적게 나올 수
있습니다. 실제 구현 시 overfetch factor를 상향하거나, 부족하면 한 번 더 fetch하는
루프가 필요할 수 있습니다 (지금은 문서화만 해두고 구현 시점에 확정).

## 7. Spring(backend) 쪽에 필요한 작업 (이 레포 밖)

이 레포에서는 계약(스키마 컬럼 + payload 필드 + retriever 필터)까지만 준비할 수
있고, 아래는 `devbridge-backend` 레포에서 별도로 진행되어야 합니다.

- `task_id` 기준 사용자 배정 조회 → `accessible_task_ids` 계산
- 워크스페이스 RBAC(Admin/Member/Viewer) 기준 → `can_view_restricted` 계산
- 문서 업로드/데이터소스 등록 UI에 "민감 정보 포함" 토글 추가 → 인덱싱 요청 시
  `sensitivity_level` 전달
- 인사/급여 등 민감 데이터소스 연결 시 기본값을 `restricted`로 강제할지 여부 결정

## 8. 지금 당장 적용 가능한 운영 가이드 (코드 변경 없이)

위 구조가 구현되기 전까지 최소한의 완화책:

1. **`DATABASE_SCHEMAS.schema_ddl`에는 순수 DDL만 넣고 실제 데이터/샘플 값을
   포함하지 않는다.** (`docs/architecture.md` 5장 참고) — 이 규칙이 깨지면 스키마
   자체가 아니라 실제 PII가 벡터스토어에 들어갑니다.
2. **인사/급여 등 고민감 문서는 별도 워크스페이스로 분리 운영**하는 것을
   임시 권장 — 지금 구조에서 유일하게 확실한 격리 경계는 `workspace_id`뿐이기
   때문입니다.
3. 신규 데이터소스/문서 업로드 시 "이 문서가 워크스페이스 전원에게 공개된다"는
   점을 업로더에게 명시적으로 안내하는 것을 Spring/프론트 쪽에 권장.

## 9. 단계적 적용 로드맵

1. **Phase 1** (이 레포, 하위호환 유지): 스키마 컬럼 + `ChatRequest`/ingestion 필드
   추가. Spring이 아직 값을 안 보내면 전부 "제한 없음"으로 기존과 동일하게 동작.
2. **Phase 2** (backend 레포): task 배정/RBAC 계산 로직 구현, 실제 값 전달 시작.
3. **Phase 3** (이 레포): retriever 필터를 "느슨한 기본값"에서 "실제 강제"로 전환,
   테스트로 교차 접근 시나리오 검증.
