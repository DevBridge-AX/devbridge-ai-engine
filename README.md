<div align="center">

# 🌁 DevBridge AI Engine
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

DevBridge 시스템은 **Spring Boot 백엔드(비즈니스 로직 및 세션 관리)**와 **FastAPI AI 엔진(Stateless 추론 및 벡터 저장소)**으로 철저히 분리되어 동작합니다.

```mermaid
sequenceDiagram
    autonumber
    actor User as 사용자 (Client)
    participant Spring as Spring Boot Backend
    participant FastAPI as FastAPI AI Engine
    participant VS as Vector Store<br>(Chroma / FAISS)
    participant DB as AI Domain DB (MySQL)
    participant LLM as LLM Provider<br>(Abstracted)

    User->>Spring: 1. 채팅 질의 요청 (대화 턴, 직무 페르소나 포함)
    Spring->>FastAPI: 2. POST /chat (대화 이력, Workspace, User, Role)
    
    rect rgb(28, 33, 41)
        Note over FastAPI: Stateless 추론 처리 파이프라인
        alt 멀티턴 대화 (Turn 2 이상)
            FastAPI->>LLM: 3. Query Rewriter를 통한 독립적 쿼리 재구성
            LLM-->>FastAPI: 4. 재구성된 쿼리 반환
        else 싱글턴 대화 (Turn 1)
            Note over FastAPI: 사용자 원본 쿼리 직접 사용
        end
        
        FastAPI->>VS: 5. 유사도 기반 문서/코드 벡터 검색
        VS-->>FastAPI: 6. 연관 청크 및 유사도 스코어 반환
        
        FastAPI->>LLM: 7. 6종 직무별 시스템 프롬프트 + Context + 대화 이력 주입
        LLM-->>FastAPI: 8. SSE 응답 스트림 & 그라운딩 구조화 데이터 반환
        
        FastAPI->>FastAPI: 9. 유사도 + LLM 신뢰성 평가 (Grounding Assessment)
    end
    
    FastAPI-->>Spring: 10. SSE Stream 전달 (Answer, Citations, Grounding Info)
    Spring->>Spring: 11. 최종 신뢰도 임계치 검증 및 알림(UC-04) 트리거 판단
    Spring-->>User: 12. 최종 스트리밍 응답 (SSE)
```

---

## 🚫 데이터베이스 접근 경계 (가장 중요)

시스템 안정성과 아키텍처적 결합도를 낮추기 위해, **Spring Boot 백엔드와 FastAPI AI 엔진은 물리적으로 다른 MySQL 계정/스키마를 공유하거나 격리**하여 사용합니다.

> [!WARNING]
> **FastAPI AI 엔진은 비즈니스 도메인 테이블에 직접 접근(생성/조회/수정)할 수 없습니다.** 모든 비즈니스 데이터는 Spring이 API 호출 시 페이로드(Payload)로 제공하거나, 사전 인덱싱 시점에 매핑 처리됩니다.

### 🗂️ 테이블 소유권 매핑

| **FastAPI 소유 & 관리 (AI 전용 스키마)** ✅ | **Spring Boot 소유 (접근 절대 금지) 🚫** |
| :--- | :--- |
| `KNOWLEDGE_DOCUMENTS` (인덱싱 문서 메타데이터) | `USERS` (사용자 정보) |
| `DATA_SOURCES` (인덱싱 데이터 소스 정보) | `WORKSPACES` (워크스페이스 설정) |
| `DATABASE_SCHEMAS` (데이터베이스 스키마 정보) | `TASKS` (워크스페이스별 태스크) |
| `GIT_COMMITS` (Git 커밋 메타 및 매핑 정보) | `CHAT_SESSIONS` / `CHAT_MESSAGES` (대화 데이터) |
| `document_chunks` (RAG용 원문 및 임베딩 버전 정보) | `MESSAGE_CITATIONS` / `NOTIFICATIONS` (출처/알림) |

* Fast API의 SQLAlchemy 모델(`app/db/models.py`)에는 우측의 **접근 금지 테이블을 절대 정의하거나 직접 조인할 수 없습니다.**

---

## ⚡ 핵심 기능 및 기술적 특징

### 1. Stateless 추론 아키텍처
대화의 맥락(Context)이나 대화 세션 상태를 저장하지 않고, Spring이 전달해주는 payload 정보(`workspace_id`, `user_id`, `role`, 대화 이력)를 기반으로 일관성 있는 RAG 추론만을 수행합니다.

### 2. 원문과 임베딩의 물리적 분리 설계 (`document_chunks`)
- **원문(Content)**: 불변의 데이터 소스(Source of Truth)로 MySQL에 보존됩니다.
- **임베딩(Embedding)**: 재생성 및 업데이트가 가능한 데이터(Derived Data)로 관리하며, `embedding_model` 및 `embedding_model_version` 컬럼을 반드시 명시하여 향후 LLM 모델 업그레이드 시 원본 유실 없이 임베딩 재처리가 용이하도록 설계되었습니다.

### 3. 직무 맞춤형 6종 페르소나 전환 (`app/core/llm/persona_prompts.py`)
사용자의 직무(`Role`)에 따라 시스템 프롬프트를 동적으로 변환합니다.
- **지원 직무**: 기획자, 개발자, QA, 디자이너, 운영자, 신규 투입자
- **동적 바인딩 변수**: 검색된 지식 컨텍스트(`retrieved_context`), 이전 대화 이력(`conversation_history`), 워크스페이스별 개인 선호도 요약(`user_preference_summary`)

### 4. 실시간 그라운딩 및 신뢰도 검증 (`app/core/rag/grounding.py`)
할루시네이션(Hallucination) 방지를 위해 2단계 검증을 수행합니다.
1. `retriever.py`에서 산출한 임베딩 벡터 간의 1차 **유사도 스코어(Similarity Score)**
2. LLM Structured Output을 사용한 2차 **그라운딩 유효성(Is Groundable) 및 신뢰성 점수(Confidence Score)**
3. 두 점수를 종합적으로 계산하여 Spring에 전달하고, Spring은 이 임계치에 따라 신뢰도가 부족한 답변에 대한 알림(UC-04) 처리를 결정합니다.

### 5. Git Webhook 인덱싱 최적화 (`app/pipelines/git_ingestion.py`)
- Git Commit 로그 파이프라인 가동 시, Stateless 원칙을 지키기 위해 **인덱싱 시점**에 Spring의 `사용자조회 API`를 미리 호출하여 커밋 작성자 정보를 `USERS.id`와 사전 매핑하여 `GIT_COMMITS.author_id`로 저장합니다. 
- 이를 통해 챗봇 질의 시점에 복잡한 서비스 간 Join이나 API 재조회 없이 저장된 ID를 즉시 응답에 내려줄 수 있습니다.

---

## 📂 프로젝트 구조 및 파일 링크

아래는 `devbridge-ai-engine` 프로젝트의 코어 모듈 구조입니다. 클릭 시 각 구성 코드 파일로 직접 이동합니다.

```
ai-engine/
├── 📝 [CLAUDE.md](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/CLAUDE.md) - 개발/컨트리뷰션 룰 명세서
├── 📂 .claude/rules/ - 핵심 아키텍처 및 DB 제약 룰 파일들
├── 📂 docs/
│   └── 📑 [architecture.md](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/docs/architecture.md) - 백엔드 통신 및 데이터 설계 아키텍처 상세 문서
├── 📂 app/
│   ├── 🏁 [main.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/main.py) - FastAPI 진입점 및 미들웨어 설정
│   ├── ⚙️ [config.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/config.py) - Pydantic 기반 환경 변수 로더
│   ├── 📂 api/
│   │   ├── 💬 [chat.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/api/chat.py) - SSE(Server-Sent Events) 스트리밍 질의응답 엔드포인트
│   │   └── 📥 [ingestion.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/api/ingestion.py) - 사내 문서/Git push 웹훅 인덱싱 API
│   ├── 📂 core/
│   │   ├── 📂 llm/
│   │   │   ├── 🔌 [provider.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/llm/provider.py) - LLM API 통신 인터페이스 추상화 클래스
│   │   │   ├── 🔄 [query_rewriter.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/llm/query_rewriter.py) - 멀티턴 흐름 파악을 위한 쿼리 재구성 엔진
│   │   │   └── 🎭 [persona_prompts.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/llm/persona_prompts.py) - 직무별 6종 시스템 프롬프트 컴포지션
│   │   ├── 📂 rag/
│   │   │   ├── 🔍 [retriever.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/rag/retriever.py) - Chroma/FAISS 연동 의미 검색 모듈
│   │   │   ├── ✂️ [chunker.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/rag/chunker.py) - 지식 데이터 청킹 유틸리티
│   │   │   └── ⚖️ [grounding.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/rag/grounding.py) - 유사도 점수와 LLM 정합성 판정 결합 모듈
│   │   └── 📂 embeddings/
│   │       └── 🧬 [embedder.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/core/embeddings/embedder.py) - 버저닝 기능 포함 임베딩 벡터 생성 모듈
│   ├── 📂 db/
│   │   ├── 🛢️ [models.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/db/models.py) - SQLAlchemy AI 도메인 전용 테이블 매핑
│   │   ├── 🔌 [session.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/db/session.py) - MySQL Connection Session 풀 관리
│   │   └── 📦 [vector_store.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/db/vector_store.py) - Chroma/FAISS 온디바이스 래퍼 클래스
│   └── 📂 pipelines/
│       ├── 🐙 [git_ingestion.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/pipelines/git_ingestion.py) - Git Commit 로그/이력 파싱 및 인덱싱 파이프라인
│       ├── 📄 [document_ingestion.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/pipelines/document_ingestion.py) - 문서 구조 청킹 및 RAG 적재 파이프라인
│       └── 💾 [export_training_data.py](file:///Users/who._.hs/opt/DevBridge_workspace/devbridge-ai-engine/app/pipelines/export_training_data.py) - 워크스페이스별 개인정보(PII) 스크러빙 가미 LoRA 데이터셋 추출기
```

---

## 🔌 API 통신 규격

### `POST /api/chat` (SSE 스트리밍 질의)

Spring Boot Backend에서 전달받는 요청 구조와 FastAPI가 반환하는 SSE 데이터 포맷 규격입니다.

#### 📤 Request Body
```json
{
  "conversation_history": [
    {"role": "user", "content": "DevBridge의 데이터베이스 접근 경계에 대해 알려줘"},
    {"role": "assistant", "content": "DevBridge AI 엔진은..."}
  ],
  "workspace_id": 12,
  "user_id": 34,
  "role": "developer",
  "query": "그럼 Spring DB 테이블에 직접 조인할 수도 없어?"
}
```

#### 📥 Response Event Stream Payload (SSE)
```json
{
  "answer_stream": "아니요, 조인할 수 없습니다. 규칙에 따라 AI 엔진은 Spring DB 스키마에 직접...",
  "citations": [
    {
      "document_id": 5,
      "chunk_id": 142,
      "similarity_score": 0.94,
      "source_type": "markdown"
    }
  ],
  "similarity_scores": [0.94],
  "is_groundable": true,
  "confidence": 0.98,
  "suggested_owner_id": null,
  "token_usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 156
  }
}
```

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
복사된 `.env.example` 파일을 바탕으로 로컬용 `.env` 파일을 생성하고 적절히 구성합니다.

```bash
cp .env.example .env
```

```ini
# .env 설정 예시
DATABASE_URL=mysql+pymysql://ai_engine_user:changeme@localhost:3306/devbridge_ai
VECTOR_STORE_PROVIDER=chroma
VECTOR_STORE_PATH=./data/vector_store
EMBEDDING_MODEL=placeholder-embedding-model
EMBEDDING_MODEL_VERSION=v1
LLM_API_KEY=your-actual-api-key
SPRING_BACKEND_BASE_URL=http://localhost:8080
```

### 4. 로컬 서버 실행
FastAPI 개발 서버를 실행하여 동작을 확인합니다.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API Swagger 문서: `http://localhost:8000/docs`에서 확인 가능합니다.