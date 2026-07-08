# AI API 활용 내역

## AI 활용 서비스 개요

DevBridge AI Engine은 사내 프로젝트 이해 지원 챗봇의 AI 백엔드로, 다음 서비스에 AI를 활용합니다.

| 서비스 | 설명 | 사용 AI |
|---|---|---|
| **RAG 기반 질의응답** | 사용자 질문에 대해 인덱싱된 문서/코드/Git 이력을 검색하고, 직무별 페르소나(6종)로 답변을 생성 | 임베딩, 메인 LLM, 그라운딩 판정 |
| **멀티턴 대화** | turn 2+ 대화에서 생략된 맥락을 복원하여 검색 품질 유지 | 쿼리 재구성 LLM |
| **문서 인덱싱** | 업로드 문서를 청킹 → 임베딩하여 벡터 DB에 저장, RAG 검색 대상으로 등록 | 임베딩 |
| **Git 커밋 인덱싱 및 분석** | Git 커밋 diff를 인덱싱하고, 변경 요약/영향범위/리스크를 AI로 분석 | 임베딩, 메인 LLM |
| **담당자 답변(Q&A) 인덱싱** | 담당자가 제공한 답변을 벡터 DB에 인덱싱하여 RAG 검색 대상에 추가 | 임베딩 |
| **문서 AI 분석** | 업로드 문서의 요약/키워드/리스크/다음조치를 생성 (현재 fallback 모드) | 메인 LLM (LLM 모드 시) |
| **신뢰도/그라운딩 판정** | 검색 결과가 질문에 답변 가능한지 이진 판정, 불충분 시 담당자 추천 | 그라운딩 LLM |

---

## 공통 사항

- 모든 API는 **GMS(SSAFY 프록시 게이트웨이)** 경유로 호출합니다.
- 단일 API Key(`GMS_API_KEY`)로 Claude / GPT / Gemini를 모두 인증합니다.
- 모델 교체는 `.env` 환경변수만 변경하면 됩니다 (코드 수정 불필요).
- provider 자동 감지: 모델명 prefix(`claude-*`, `gpt-*`/`o*`, `gemini-*`)로 provider를 결정합니다.

---

## 1. 메인 답변 생성

| 항목 | 값 |
|---|---|
| 용도 | RAG 기반 챗봇 답변 스트리밍 생성 |
| 환경변수 | `MAIN_MODEL` |
| 현재 모델 | `claude-sonnet-4-6` |
| provider | Anthropic |
| 호출 함수 | `provider.call_main_stream()` |
| 엔드포인트 | `POST {anthropic_base_url}/v1/messages` |
| 인증 헤더 | `x-api-key: {GMS_API_KEY}` |
| 추가 헤더 | `anthropic-version: 2023-06-01` |
| 응답 형식 | SSE 스트리밍 (event: `content_block_delta` → 텍스트 청크) |
| 호출 시점 | 사용자 질문 → grounding 통과 후 답변 생성 시 |
| max_tokens | 4096 |

**비스트리밍 변형**: `provider.call_main()` — 동일 모델, 단일 응답 반환.

**JSON 구조화 변형**: `provider.call_structured()` — 동일 모델, JSON 응답 파싱. 문서 분석(LLM 모드), Git 커밋 분석에 사용.

---

## 2. 멀티턴 쿼리 재구성

| 항목 | 값 |
|---|---|
| 용도 | turn 2+ 대화에서 생략된 맥락을 복원하여 독립적 검색 쿼리로 변환 |
| 환경변수 | `REWRITE_MODEL` |
| 현재 모델 | `claude-sonnet-4-6` (권장: `gemini-3.5-flash`) |
| 호출 함수 | `provider.call_rewrite()` |
| 호출 시점 | `conversation_history`가 있는 turn 2+ 질문 시 |
| max_tokens | 512 |

**예시**:
```
이전 대화: "결제 API 문서 보여줘" → 답변
새 질문: "그거 소스 코드는 어디 있어?"
→ rewrite 결과: "결제 API 소스 코드 위치"
```

turn 1에서는 호출하지 않습니다 (원본 쿼리 그대로 사용).

> **최적화 참고**: 쿼리 재구성은 추론 난이도가 낮아 경량 모델(`gemini-3.5-flash`)로 충분합니다. 비용 5~10배 절감, 지연시간 감소 효과.

---

## 3. 그라운딩 판정

| 항목 | 값 |
|---|---|
| 용도 | 검색된 컨텍스트가 질문에 답변 가능한지 이진 판정 |
| 환경변수 | `GROUNDING_MODEL` |
| 현재 모델 | `gemini-3.5-flash-lite` (권장: `gemini-3.5-flash`) |
| provider | Gemini |
| 호출 함수 | `provider.call_grounding()` |
| 엔드포인트 | `POST {gemini_base_url}/models/{model}:generateContent` |
| 인증 헤더 | `x-goog-api-key: {GMS_API_KEY}` |
| 응답 형식 | JSON `{"is_groundable": bool, "confidence": 0.0~1.0}` |
| 호출 시점 | 벡터 유사도 1차 필터(≥0.35) 통과 후 2차 판정 시 |
| max_tokens | 64 |

**판정 규칙**:
1. 일상 잡담/인사 → 컨텍스트 무관하게 `is_groundable=true`
2. 범용 기술 질문 (JWT, SQL JOIN 등) → 컨텍스트 무관하게 `is_groundable=true`
3. 프로젝트 고유 질문 → 컨텍스트에 근거가 있을 때만 `is_groundable=true`

Gemini의 `responseSchema` (Structured Output)를 사용하여 JSON 형식을 강제합니다.

---

## 4. 임베딩 생성

| 항목 | 값 |
|---|---|
| 용도 | 문서/코드/Git 커밋 청크 및 검색 쿼리의 벡터 임베딩 생성 |
| 환경변수 | `EMBEDDING_MODEL` |
| 현재 모델 | `gemini-embedding-2` |
| provider | Gemini |
| 호출 함수 | `embedder.embed_texts()` |
| 엔드포인트 | `POST {gemini_base_url}/models/{model}:batchEmbedContents` |
| 인증 헤더 | `x-goog-api-key: {GMS_API_KEY}` |
| 배치 크기 | 최대 100건/요청 (초과 시 자동 분할) |
| 호출 시점 | 문서/Git/담당자답변 인덱싱 시, 채팅 검색 쿼리 임베딩 시 |

**참고**: Gemini Embedding API는 응답에 토큰 수를 포함하지 않습니다. `usage_logs`에는 텍스트 1건당 0.2 토큰으로 고정 추정하여 누적합니다.

---

## 5. 문서 AI 분석 (선택적)

| 항목 | 값 |
|---|---|
| 용도 | 업로드 문서의 요약/키워드/리스크/다음조치 생성 |
| 환경변수 | `AI_ANALYSIS_MODE`, `DOCUMENT_ANALYSIS_MODEL` |
| 현재 모드 | `fallback` (LLM 미사용, 규칙 기반) |
| LLM 모드 시 | `call_structured()` → MAIN_MODEL 사용 |
| 호출 시점 | Spring이 문서 등록 시 `POST /api/analysis/document` 호출 |

`AI_ANALYSIS_MODE=llm`으로 변경하면 실제 LLM 분석을 수행합니다. 현재는 fallback 모드로 파일명/미리보기 기반 규칙 분석만 수행.

---

## 6. Git 커밋 AI 분석 (선택적)

| 항목 | 값 |
|---|---|
| 용도 | Git 커밋의 요약/영향범위/리스크/다음조치 생성 |
| 호출 함수 | `call_structured()` → MAIN_MODEL |
| 현재 모드 | `AI_ANALYSIS_MODE` 설정에 따름 |
| 호출 시점 | Git 커밋 인덱싱 시 (`pipelines/git_ingestion.py`) |

---

## 호출 흐름 요약

```
사용자 질문 (turn 1)
  │
  ├─ [임베딩] embed_texts(query)         ← gemini-embedding-2
  ├─ [BM25] 인메모리 검색                 ← API 호출 없음
  │
  ├─ [grounding 1차] 유사도 < 0.35?      ← API 호출 없음
  │     └─ Yes → is_groundable=False (차단)
  │
  ├─ [grounding 2차] call_grounding()    ← gemini-3.5-flash
  │     └─ is_groundable=false → 차단
  │
  └─ [답변 생성] call_main_stream()      ← claude-sonnet-4-6

사용자 질문 (turn 2+)
  │
  ├─ [쿼리 재구성] call_rewrite()        ← gemini-3.5-flash (권장)
  │
  └─ 이후 turn 1과 동일
```

---

## GMS Base URL 정리

| Provider | Base URL | 용도 |
|---|---|---|
| Anthropic | `https://gms.ssafy.io/gmsapi/api.anthropic.com` | Claude 모델 (메인 답변) |
| OpenAI | `https://gms.ssafy.io/gmsapi/api.openai.com/v1` | GPT 모델 (현재 미사용) |
| Gemini | `https://gms.ssafy.io/gmsapi/generativelanguage.googleapis.com/v1beta` | Gemini 모델 (임베딩, 그라운딩, 쿼리 재구성) |

---

## 환경변수 일람

```env
GMS_API_KEY=                          # GMS 통합 API Key (필수)

MAIN_MODEL=claude-sonnet-4-6          # 메인 답변 생성
REWRITE_MODEL=gemini-3.5-flash        # 멀티턴 쿼리 재구성 (권장)
GROUNDING_MODEL=gemini-3.5-flash      # 그라운딩 판정 (권장)
EMBEDDING_MODEL=gemini-embedding-2    # 임베딩 생성

AI_ANALYSIS_MODE=fallback             # 문서 분석 모드 (fallback | llm)
DOCUMENT_ANALYSIS_MODEL=fallback-v1   # 문서 분석 표시용 모델명
```

---

## 토큰 사용량 추적

`/chat` 응답의 `done` 이벤트에서 모델별 토큰 사용량을 분리 전달합니다:

```json
{
  "token_usage": {
    "main":      {"model": "claude-sonnet-4-6",  "prompt_tokens": 512, "completion_tokens": 128},
    "rewrite":   {"model": "gemini-3.5-flash",   "prompt_tokens": 80,  "completion_tokens": 20},
    "grounding": {"model": "gemini-3.5-flash",   "prompt_tokens": 100, "completion_tokens": 10}
  }
}
```

- `main`: 답변 생성 토큰. `is_groundable=false`이면 0/0.
- `rewrite`: turn 2+에서만 발생. turn 1이면 `null`.
- `grounding`: 유사도 1차 필터에서 차단되면 LLM 호출 없으므로 `null`.
- Credit 단가 적용 및 비용 계산은 Spring 백엔드의 책임.

인덱싱(임베딩) 토큰은 `usage_logs` 테이블에 workspace별 일 단위로 별도 누적됩니다.
