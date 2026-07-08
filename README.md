<div align="center">

# 🌁 AI Engine of DevBridge AX
### 사내 프로젝트 이해 지원을 위한 실시간 RAG & 멀티 페르소나 추론 엔진

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F27?style=for-the-badge&logo=sqlite&logoColor=white)](#)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](#)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FC9440?style=for-the-badge&logo=target&logoColor=white)](#)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=for-the-badge&logo=docker&logoColor=white)](#)

<p align="center">
  <b>DevBridge AI Engine</b>은 사내 프로젝트 이해를 지원하는 챗봇 시스템의 핵심 <b>Stateless AI 백엔드(FastAPI)</b>입니다.<br>
  Spring Boot 백엔드와 긴밀히 연동되어, 사내 문서/코드/Git 이력을 RAG(Retrieval-Augmented Generation) 기반으로 탐색하고,<br>
  다양한 직무별 페르소나에 맞춰 지식을 유연하게 전달합니다.
</p>

---
</div>

## 🌐 시스템 아키텍처 및 인터랙션 흐름

DevBridge 시스템은 **Spring Boot 백엔드(비즈니스 로직 및 세션 관리)**와 **FastAPI AI 엔진(Stateless 추론 및 벡터 저장소)**으로 철저히 분리되어 동작합니다. LLM 호출은 **GMS(사내 프록시 게이트웨이)**를 통해 Claude / GPT / Gemini 중 용도에 맞는 모델로 라우팅됩니다.

![RAG Architecture](RAG.png)

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자 (Client)
    participant Spring as Spring Boot Backend
    participant FastAPI as FastAPI AI Engine
    participant VS as Vector Store<br>(Chroma) + BM25
    participant DB as AI Domain DB (MySQL)
    participant GMS as GMS Gateway<br>(Claude / GPT / Gemini)

    User->>Spring: 1. 채팅 질의 요청 (대화 턴, 직무 페르소나 포함)
    Spring->>FastAPI: 2. POST /api/chat (대화 이력, Workspace, User, Role)

    rect rgb(28, 33, 41)
        Note over FastAPI: Stateless 추론 처리 파이프라인
        alt 멀티턴 대화 (Turn 2 이상)
            FastAPI->>GMS: 3. Query Rewriter를 통한 독립적 쿼리 재구성
            GMS-->>FastAPI: 4. 재구성된 쿼리 반환
        else 싱글턴 대화 (Turn 1)
            Note over FastAPI: 사용자 원본 쿼리 직접 사용
        end

        FastAPI->>VS: 5. 벡터 유사도 + BM25 하이브리드 검색
        VS-->>FastAPI: 6. 연관 청크 및 유사도 스코어 반환

        FastAPI->>GMS: 7. 유사도 1차 필터 통과 시 그라운딩 2차 판정 요청
        GMS-->>FastAPI: 8. is_groundable / confidence 반환

        FastAPI->>GMS: 9. 6종 직무별 시스템 프롬프트 + Context + 대화 이력 주입
        GMS-->>FastAPI: 10. SSE 토큰 스트림 반환
    end

    FastAPI-->>Spring: 11. SSE Stream 전달 (token* + done: Citations/Grounding/TokenUsage)
    Spring->>Spring: 12. 최종 신뢰도 임계치 검증 및 알림(UC-04) 트리거 판단
    Spring-->>User: 13. 최종 스트리밍 응답 (SSE)
```

> 통신 규격과 데이터 설계의 전체 상세본은 [`docs/architecture.md`](docs/architecture.md)를, 모델별 호출 스펙(엔드포인트/헤더/토큰 추적)은 [`docs/ai-api-usage.md`](docs/ai-api-usage.md)를 참고하세요. 이 README는 개요만 다룹니다.

---

## 🧩 책임 분담 (AI Engine ↔ Spring Backend)

기능 단위로 보면 두 서비스의 책임은 다음과 같이 나뉩니다. 테이블 단위 소유권은 아래 [DB 접근 경계](#-데이터베이스-접근-경계-가장-중요) 절을 참고하세요.

| 구분 | AI Engine 책임 | Spring Backend 책임 |
| :--- | :--- | :--- |
| Chat | RAG 검색(벡터+BM25), 프롬프트 구성, LLM 호출, SSE `token`/`done` 반환 | 세션 관리, `CHAT_MESSAGES`/`MESSAGE_CITATIONS` 저장, 사용자에게 응답 전달 |
| Document | 청킹·임베딩·인덱싱, `KNOWLEDGE_DOCUMENTS`(AI 소유 메타데이터)의 분석 상태/결과 저장 | 파일 업로드 수신, 문서-task 연동 및 접근 권한 판단, 인덱싱 API 트리거 |
| Git | 커밋 인덱싱, 요약/영향범위/리스크 분석(`GIT_COMMIT_ANALYSIS`) | Webhook 수신 후 내부망으로 전달, 사용자조회 API 제공(author_id 매핑용) |
| Grounding | 유사도 + LLM 자기평가로 `is_groundable`/`confidence` 산출 | 임계치 비교, 담당자 확인 요청/알림(UC-04) 생성 |
| DB Access | AI 전용 스키마(`KNOWLEDGE_DOCUMENTS` 등)만 소유·조회 | 비즈니스 도메인 DB(`USERS`/`WORKSPACES`/`CHAT_MESSAGES` 등) 소유 |
| RAG 접근 제어 *(설계 중)* | 워크스페이스 격리 강제(Chroma 컬렉션 분리) + task/민감도 기반 사후 필터링 | task 배정·RBAC 계산 → `accessible_task_ids`/`can_view_restricted` 전달 |

> ⚠️ `KNOWLEDGE_DOCUMENTS`는 AI Engine이 소유하는 테이블입니다. Spring은 별도의 자체 문서 엔티티(업로드/권한/task 연동용, `backend_document_id`로 AI 쪽 문서와 연결)를 따로 갖고 있으므로 두 메타데이터를 혼동하지 않아야 합니다.

---

## 🚫 데이터베이스 접근 경계 (가장 중요)

시스템 안정성과 아키텍처적 결합도를 낮추기 위해, **Spring Boot 백엔드와 FastAPI AI 엔진은 물리적으로 다른 MySQL 계정/스키마를 공유하거나 격리**하여 사용합니다.

> [!WARNING]
> **FastAPI AI 엔진은 비즈니스 도메인 테이블에 직접 접근(생성/조회/수정)할 수 없습니다.** 모든 비즈니스 데이터는 Spring이 API 호출 시 페이로드(Payload)로 제공하거나, 사전 인덱싱 시점에 매핑 처리됩니다.

### 🗂️ 테이블 소유권 매핑

| **FastAPI 소유 & 관리 (AI 전용 스키마)** ✅ | **Spring Boot 소유 (접근 절대 금지)** 🚫 |
| :--- | :--- |
| `KNOWLEDGE_DOCUMENTS` (인덱싱 문서 메타데이터) | `USERS` (사용자 정보) |
| `DATA_SOURCES` (인덱싱 데이터 소스 정보) | `WORKSPACES` (워크스페이스 설정) |
| `DATABASE_SCHEMAS` (데이터베이스 스키마 정보) | `TASKS` (워크스페이스별 태스크) |
| `GIT_COMMITS` (Git 커밋 메타 및 매핑 정보) | `CHAT_SESSIONS` / `CHAT_MESSAGES` (대화 데이터) |
| `GIT_COMMIT_ANALYSIS` (커밋 AI 분석 결과) | `MESSAGE_CITATIONS` / `NOTIFICATIONS` (출처/알림) |
| `document_chunks` (RAG용 원문 및 임베딩 버전 정보) | `OWNER_CONFIRMATIONS` (담당자 확인 이력) |
| `usage_logs` (워크스페이스별 임베딩 토큰 사용량) | 그 외 모든 Spring 소유 비즈니스 도메인 테이블 |

* FastAPI의 SQLAlchemy 모델(`app/db/models.py`)에는 우측의 **접근 금지 테이블을 절대 정의하거나 직접 조인할 수 없습니다.**

---

## ⚡ 핵심 기능 및 기술적 특징

### 1. Stateless 추론 아키텍처
대화의 맥락(Context)이나 세션 상태를 저장하지 않고, Spring이 매 호출마다 전달하는 payload(`session_id`, `workspace_id`, `user_id`, `role`, 대화 이력)를 기반으로 일관성 있는 RAG 추론만을 수행합니다.

### 2. GMS 기반 멀티 LLM 프로바이더 통합
단일 `GMS_API_KEY`로 Claude / GPT / Gemini를 함께 인증하며, `app/core/llm/provider.py`가 모델명 prefix(`claude-*`, `gpt-*`/`o*`, `gemini-*`)로 provider를 자동 감지해 호출합니다. 용도별로 별도 모델을 지정할 수 있습니다.

| 용도 | 환경변수 | 현재 기본값 | 선정 근거 |
| :--- | :--- | :--- | :--- |
| 메인 답변 생성 | `MAIN_MODEL` | `claude-sonnet-4-6` | 긴 컨텍스트 종합, 페르소나 문체 변환, 멀티턴 추론 품질이 필요해 가장 무거운 모델 배정 |
| 멀티턴 쿼리 재구성 (turn 2+) | `REWRITE_MODEL` | `claude-sonnet-4-6` (권장: `gemini-3.5-flash`) | 검색어 다듬기 수준의 낮은 난이도 태스크 — 경량 모델 전환 시 비용 5~10배 절감 가능 |
| 그라운딩 이진 판정 | `GROUNDING_MODEL` | `gemini-3.5-flash-lite` | `{is_groundable, confidence}` 2필드 구조화 출력(max_tokens=64)만 필요해 가장 저비용 티어로 배정 |
| 임베딩 생성 | `EMBEDDING_MODEL` | `gemini-embedding-2` | GMS 내 배치(최대 100건/요청) 지원, 별도 계약 없이 비용 효율적 |

모델 교체/비용 튜닝은 `.env` 값만 변경하면 되며 `provider.py` 코드 변경은 필요하지 않습니다. **태스크 난이도에 비례해 모델 등급을 배정하는 "모델 계층화"** 원칙을 따릅니다 — 메인 답변에만 고성능 모델을 쓰고, 분류/재구성처럼 짧고 결정적인 태스크는 경량 모델로 내려 비용과 지연시간을 함께 줄입니다. 상세 비교/호출 스펙(엔드포인트, 인증 헤더, 응답 형식)은 [`docs/ai-api-usage.md`](docs/ai-api-usage.md) 참고.

### 3. 벡터 유사도 + BM25 하이브리드 검색
`app/core/rag/retriever.py`가 Chroma 기반 벡터 유사도 검색과 `app/core/rag/bm25_index.py`의 BM25 키워드 검색을 결합해 RAG 컨텍스트를 구성합니다.

### 4. 원문과 임베딩의 물리적 분리 설계 (`document_chunks`)
- **원문(Content)**: 불변의 데이터 소스(Source of Truth)로 MySQL에 보존됩니다.
- **임베딩(Embedding)**: 재생성 및 업데이트가 가능한 데이터(Derived Data)로 관리하며, `embedding_model` 및 `embedding_model_version` 컬럼을 반드시 명시하여 향후 임베딩 모델 업그레이드 시 원본 유실 없이 임베딩만 재처리할 수 있도록 설계되었습니다.

### 5. 직무 맞춤형 6종 페르소나 전환 (`app/core/llm/persona_prompts.py`)
사용자의 직무(`Role`)에 따라 시스템 프롬프트를 동적으로 변환합니다.
- **지원 직무**: 기획자, 개발자, QA, 디자이너, 운영자, 신규 투입자
- **동적 바인딩 변수 (1차 범위)**: 검색된 지식 컨텍스트(`retrieved_context`), 이전 대화 이력(`conversation_history`)
- `user_preference_summary`(개인별 선호 반영)는 2차 예정 변수로 1차 템플릿에는 포함되지 않습니다.

### 6. 실시간 그라운딩 및 신뢰도 검증 (`app/core/rag/grounding.py`)
할루시네이션(Hallucination) 방지를 위해 2단계 검증을 수행합니다.
1. `retriever.py`에서 산출한 유사도 스코어의 1차 임계치 필터 (`GROUNDING_SIMILARITY_THRESHOLD`, 기본 `0.35`)
2. 1차 필터 통과 시에만 호출되는 LLM Structured Output 기반 2차 **그라운딩 유효성(`is_groundable`) 및 신뢰성 점수(`confidence`)**
3. 두 결과를 종합해 Spring에 값만 전달하며, 임계치 비교/알림(UC-04) 트리거는 **Spring의 책임**입니다.

### 7. Git Webhook 인덱싱 및 커밋 AI 분석 (`app/pipelines/git_ingestion.py`)
- Git Commit 로그 파이프라인 가동 시, Stateless 원칙을 지키기 위해 **인덱싱 시점**에 Spring의 사용자조회 API를 미리 호출하여 커밋 작성자 정보를 `USERS.id`와 사전 매핑하여 `GIT_COMMITS.author_id`로 저장합니다.
- 커밋 요약/영향범위/리스크 분석 결과는 `GIT_COMMIT_ANALYSIS` 테이블에 별도로 저장됩니다.
- 챗봇 질의 시점에는 복잡한 서비스 간 Join이나 API 재조회 없이 저장된 값을 즉시 응답에 내려줍니다.

### 8. 문서 AI 분석 (`app/pipelines/document_analysis.py`)
업로드 문서의 요약/키워드/리스크/다음조치를 생성합니다. `AI_ANALYSIS_MODE`가 `fallback`(규칙 기반, 기본값)인지 `llm`(`MAIN_MODEL` 호출)인지에 따라 동작이 달라집니다.

### 9. 담당자 답변(Owner Answer) 인덱싱 (`app/pipelines/owner_answer_ingestion.py`)
담당자가 제공한 답변을 임베딩하여 `document_chunks`(`source_type=owner_answer`)에 RAG 검색 대상으로 추가합니다.

### 10. 워크스페이스 요약 (`app/pipelines/workspace_summary.py`)
워크스페이스 단위 활동/문서 요약을 LLM 모드 또는 fallback 모드로 생성해 Spring에 제공합니다.

### 11. LoRA 학습데이터 export (`app/pipelines/export_training_data.py`)
워크스페이스 단위로 격리된 JSONL을 PII 스크러빙과 함께 생성합니다.

---

## 🚀 성능 및 토큰 최적화

RAG 아키텍처 자체 외에, 응답 지연시간과 LLM 비용을 줄이기 위해 적용된 최적화 기법입니다.

- **모델 계층화** — 위 "GMS 기반 멀티 LLM 프로바이더 통합" 표 참고. 태스크 난이도에 안 맞는 과스펙 모델 사용을 피합니다.
- **그라운딩 1차 필터** — 벡터 유사도가 임계치(`GROUNDING_SIMILARITY_THRESHOLD`, 기본 0.35) 미만이면 그라운딩 LLM 호출 자체를 생략해 불필요한 API 호출을 차단합니다.
- **turn 1 쿼리 재구성 생략** — 첫 질문은 재구성 LLM 호출 없이 바로 검색해 지연시간과 비용을 줄입니다.
- **벡터+BM25 비동기 병렬 검색 + 인덱스 캐싱** — 임베딩 호출과 BM25 검색을 `asyncio`로 동시에 실행해 왕복 지연을 겹쳐서 단축합니다. 워크스페이스별 BM25 인덱스는 TTL 기반으로 캐싱되어 인덱싱 시점에만 무효화됩니다 (`app/core/rag/bm25_index.py`).
- **대화 히스토리 토큰 예산 관리** (`app/core/utils/token_counter.py`) — `tiktoken`으로 추정해 히스토리를 22,000 토큰 예산 안에서 최신 턴 우선으로 자르고, Turn 1은 전체 예산의 30% 이하일 때만 보호합니다. 컨텍스트 초과와 불필요한 프롬프트 비용을 함께 방지합니다.
- **의미 단위 청킹 + 임베딩 배치 처리** — 헤더/함수/문장 경계로 분할 후 작은 조각은 인접 조각과 병합하고(문서 1500자·코드 800자 기준, `app/core/rag/chunker.py`), 임베딩은 최대 100건씩 배치 호출해 API 왕복 횟수를 최소화합니다.

---

## 📂 프로젝트 구조

```text
devbridge-ai-engine/
├── CLAUDE.md                       # 개발/컨트리뷰션 룰 명세서
├── docs/
│   └── architecture.md             # 백엔드 통신 및 데이터 설계 아키텍처 상세 문서
├── alembic/                        # DB 마이그레이션 (AI 도메인 스키마)
├── app/
│   ├── main.py                     # FastAPI 진입점 및 라우터 등록
│   ├── config.py                   # Pydantic 기반 환경 변수 로더
│   ├── api/
│   │   ├── chat.py                 # SSE 스트리밍 질의응답 엔드포인트 (/api/chat)
│   │   ├── ingestion.py            # 문서/Git push 인덱싱 엔드포인트 (/api/ingestion)
│   │   ├── usage.py                # 워크스페이스별 임베딩 토큰 사용량 집계 (/api/usage)
│   │   └── analysis.py             # 문서/커밋 AI 분석 엔드포인트 (/api/analysis)
│   ├── core/
│   │   ├── chat_pipeline.py        # /chat 요청 처리 오케스트레이션
│   │   ├── security.py             # Internal API Key 검증 (X-Internal-Api-Key)
│   │   ├── llm/
│   │   │   ├── provider.py             # GMS 기반 LLM 클라이언트 추상화 (provider 자동 감지)
│   │   │   ├── query_rewriter.py       # 멀티턴 쿼리 재구성
│   │   │   └── persona_prompts.py      # 직무별 시스템 프롬프트 6종
│   │   ├── rag/
│   │   │   ├── retriever.py            # 벡터 + BM25 하이브리드 검색
│   │   │   ├── bm25_index.py           # BM25 인메모리 인덱스
│   │   │   ├── tokenizer.py            # 검색용 토크나이저
│   │   │   ├── chunker.py              # 문서 청킹
│   │   │   └── grounding.py            # 신뢰도 판정
│   │   ├── embeddings/embedder.py      # 임베딩 생성, 모델 버전 관리
│   │   └── utils/token_counter.py      # 토큰 사용량 계산 유틸
│   ├── db/
│   │   ├── models.py                # SQLAlchemy 모델 (AI 도메인 테이블만)
│   │   ├── session.py               # MySQL Connection Session 풀 관리
│   │   └── vector_store.py          # Chroma 래퍼
│   ├── pipelines/
│   │   ├── git_ingestion.py             # Git 커밋 인덱싱 파이프라인
│   │   ├── document_ingestion.py        # 문서 청킹 및 RAG 적재 파이프라인
│   │   ├── document_analysis.py         # 문서 AI 분석 (fallback/llm)
│   │   ├── owner_answer_ingestion.py    # 담당자 답변 인덱싱
│   │   ├── workspace_summary.py         # 워크스페이스 요약 생성
│   │   └── export_training_data.py      # LoRA 학습 데이터셋 추출기 (PII 스크러빙 포함)
│   └── schemas/                     # 엔드포인트별 요청/응답 Pydantic 스키마
└── tests/
```

---

## 🔌 API 통신 규격

### `POST /api/chat` (SSE 스트리밍 질의)

Spring Boot Backend에서 전달받는 요청 구조와 FastAPI가 반환하는 SSE 이벤트 포맷입니다. 모든 요청에는 `X-Internal-Api-Key` 헤더가 필요합니다.

#### 📤 Request Body
```json
{
  "session_id": "sess-123",
  "content": "그럼 Spring DB 테이블에 직접 조인할 수도 없어?",
  "conversation_history": [
    {"role": "user", "content": "DevBridge의 데이터베이스 접근 경계에 대해 알려줘"},
    {"role": "assistant", "content": "DevBridge AI 엔진은..."}
  ],
  "workspace_id": "12",
  "user_id": "34",
  "role": "developer"
}
```

#### 📥 Response — SSE 이벤트 스트림

답변 텍스트와 메타데이터를 하나의 필드로 합치지 않고, **`token` 이벤트(반복)**와 **`done` 이벤트(스트림 종료 시 1회)**로 분리해서 보냅니다.

```
event: token
data: {"text": "아니요, "}

event: token
data: {"text": "조인할 수 없습니다..."}

event: done
data: {
  "citations": [
    {
      "source_type": "document",
      "source_id": "5",
      "chunk_id": 142,
      "title": "DB 접근 경계 정책",
      "similarity_score": 0.94
    }
  ],
  "is_groundable": true,
  "confidence": 0.98,
  "suggested_owner_id": null,
  "prompt_version": "persona-v1",
  "token_usage": {
    "main": {"model": "claude-sonnet-4-6", "prompt_tokens": 1024, "completion_tokens": 156},
    "rewrite": null,
    "grounding": {"model": "gemini-3.5-flash-lite", "prompt_tokens": 100, "completion_tokens": 10},
    "context_truncated": false
  }
}
```

- `citations[].source_type`은 `document` / `git_commit` / `db_schema` 중 하나이며, `source_id`는 해당 원본 테이블(`KNOWLEDGE_DOCUMENTS`/`GIT_COMMITS`/`DATABASE_SCHEMAS`)의 PK, `chunk_id`는 `document_chunks.id`입니다.
- `token_usage.rewrite`는 turn 1에서는 호출되지 않으므로 `null`, `token_usage.grounding`은 유사도 1차 필터에서 차단되면 `null`입니다.
- Credit 단가 적용, `estimated_cost` 계산, 임계치 비교/알림(UC-04) 트리거는 모두 **Spring의 책임**이며 이 레포는 값만 반환합니다.

그 외 엔드포인트(`/api/ingestion`, `/api/usage`, `/api/analysis`)의 요청/응답 스펙은 각 라우터 파일과 [`docs/architecture.md`](docs/architecture.md)를 참고하세요.

---

## 🚀 시작하기 (Getting Started)

### 1. 개발 환경 요구사항
- Python 3.11 이상
- MySQL 8.0 이상

### 2. 가상환경 및 패키지 설치
이 프로젝트는 최신 `pyproject.toml` 표준을 따르고 있습니다. 가상환경을 생성한 후 의존성을 설치하십시오.

```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 개발용 패키지(pytest 등)를 포함하여 설치
pip install -e ".[dev]"
```

### 3. 환경 변수 설정
`.env.example` 파일을 바탕으로 로컬용 `.env` 파일을 생성하고 값을 채웁니다.

```bash
cp .env.example .env
```

```ini
# --- Internal API 인증 (Spring -> FastAPI, X-Internal-Api-Key 헤더 검증) ---
INTERNAL_API_KEY=changeme

# --- Database (AI 도메인 전용 스키마, Spring과 별도 DB 계정) ---
DATABASE_URL=mysql+pymysql://ai_engine_user:changeme@localhost:3306/devbridge_ai

# --- Vector store (임베디드/파일 기반, 별도 서버 없음) ---
VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PATH=./data/vector_store

# --- GMS 통합 API Key (Claude / GPT / Gemini 공통) ---
GMS_API_KEY=your-gms-api-key

# --- LLM 모델 설정 (교체는 이 값들만 변경, 코드 수정 불필요) ---
MAIN_MODEL=claude-sonnet-4-6
REWRITE_MODEL=claude-sonnet-4-6
GROUNDING_MODEL=gemini-3.5-flash-lite
EMBEDDING_MODEL=gemini-embedding-2

# --- Spring backend 연동 ---
SPRING_BACKEND_BASE_URL=http://localhost:8080
```

전체 변수 목록과 각 변수의 의미는 [`.env.example`](.env.example)을 참고하세요.

### 4. 로컬 서버 실행
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Swagger 문서: `http://localhost:8000/docs`에서 확인 가능합니다.
- 헬스체크: `GET /health`

### 5. Docker로 실행
```bash
docker build -t devbridge-ai-engine .
docker run --env-file .env -p 8000:8000 devbridge-ai-engine
```

### 6. DB 마이그레이션 (Alembic)
```bash
alembic upgrade head
```
자세한 마이그레이션 명령어는 [`docs/architecture.md`](docs/architecture.md#5-db-마이그레이션-alembic) 참고.
