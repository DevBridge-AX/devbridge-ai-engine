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
| 현재 모델 (권장) | `gemini-2.5-flash-lite` |
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

> **주의**: thinking 모델(`gemini-2.5-flash`, `gemini-3.5-flash`)은 grounding에
> 사용할 수 없습니다. `call_grounding()`의 `max_tokens=64`가 추론(thinking)
> 토큰에 모두 소비되어 `MAX_TOKENS`로 truncate되고 빈 응답이 반환됩니다
> (`provider.call_grounding()`은 이 경우 `finishReason`/`thoughtsTokenCount`를
> 감지해 경고 로그를 남기고 `{}`를 반환 — grounding.assess()는 이를 파싱 실패로
> 간주해 유사도 fallback을 적용합니다). 현재 `gemini-2.5-flash-lite`만 thinking
> 없이 정상 동작합니다.

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

**알려진 한계**: batchEmbedContents 응답의 usageMetadata.promptTokenCount(배치 요청 1건당 1개, 텍스트 건별 값 아님)를 통해 provider 실측 토큰을 확인할 수 있으며, `embedder.embed_texts()`가 이를 배치 단위로 합산해 `EmbedResult.provider_tokens`로 노출하고 인덱싱 메트릭(`embedding_provider_tokens`)에 기록합니다. 다만 `usage_logs.embedding_tokens`에 누적되는 값은 여전히 provider 실측치가 아니라, 텍스트 건수 × `embedding_tokens_per_text`(기본 0.2, GMS 과금 단위 추정치)로 계산한 추정치입니다(`EmbedResult.token_source="estimate_per_text"`). `usage_logs.embedding_tokens`이 provider 실측 토큰과 GMS 과금 단위 추정치 중 무엇을 의미해야 하는지는 Spring 쪽 결정이 아직 열려 있어, 그 값 자체는 변경하지 않았습니다. `usage_logs` 테이블 스키마 자체는 변경되지 않습니다.

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

**LLM 실패 시 fallback**: `call_structured()`가 HTTP 오류(`httpx.HTTPStatusError` 등)나
JSON 파싱 오류를 던지면 `analyze_document()`가 이를 잡아 fallback 규칙 분석 결과로
대체하고 `logger.warning()`으로 원인을 남깁니다(예외를 그대로 전파해 API 500으로
이어지던 기존 동작을 git 커밋 분석과 동일한 패턴으로 통일). 이때 응답의 `mode`는
`"llm_fallback"`으로 표시되어 정상 `"fallback"` 모드와 구분됩니다. `mode` 필드는
자유 문자열(`str`)이며, Spring `DocumentAnalysisResponse.mode`도 값으로 분기하지 않고
저장만 하는 것을 확인했습니다(2026-10-01 조사, `devbridge-backend` 읽기 전용 참조).

**건당 토큰 비용** (라이브 검증 1회차 기준, `llm_calls.jsonl` purpose=`document_analysis`
집계 — `RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live/test_analysis_live.py -s`
실행 후 기입):
- prompt_tokens 평균: `TODO(라이브 실행 후 기입)`
- completion_tokens 평균: `TODO(라이브 실행 후 기입)`

---

## 6. Git 커밋 AI 분석 (선택적)

| 항목 | 값 |
|---|---|
| 용도 | Git 커밋의 요약/영향범위/리스크/다음조치 생성 |
| 호출 함수 | `call_structured()` → MAIN_MODEL |
| 현재 모드 | `AI_ANALYSIS_MODE` 설정에 따름 |
| 호출 시점 | Git 커밋 인덱싱 시 (`pipelines/git_ingestion.py`) |

**한국어 출력 규칙 / PII 최소화**: 커밋 분석 system prompt에 문서 분석과 동일한 취지의
"summary/next_action은 반드시 한국어로 작성" 규칙을 추가했습니다(`risk_level`/
`impact_area`는 정해진 영문 값을 그대로 유지). `author_email`은 PII 최소화를 위해 LLM
프롬프트 입력(커밋 메타데이터, diff 미리보기 헤더 포함)에서 완전히 제외했습니다.
`GIT_COMMITS.author_email` 컬럼 저장과 임베딩/RAG 청크용 commit 텍스트에는 영향이
없습니다.

**비용 주의 — 커밋 수만큼 호출**: 커밋 분석은 **git push 1회에 포함된 커밋 수만큼**
`call_structured()`를 호출합니다(예: 30개 커밋을 한 번에 push하면 30회 호출). 대량
push 시 비용/지연이 선형으로 증가하므로, 필요 시 후속 과제로 배치당 분석 커밋 수를
제한하는 `COMMIT_ANALYSIS_MAX_PER_BATCH`(초과분은 fallback 처리) 도입을 검토할 수
있습니다(이번 범위에서는 구현하지 않음, 아이디어만 기록).

**건당 토큰 비용** (라이브 검증 1회차 기준, purpose=`commit_analysis` 집계):
- prompt_tokens 평균: `TODO(라이브 실행 후 기입)`
- completion_tokens 평균: `TODO(라이브 실행 후 기입)`

---

## 7. 워크스페이스 대시보드 AI 요약 (선택적)

| 항목 | 값 |
|---|---|
| 용도 | 워크스페이스 대시보드 상태 요약(3~5문장 한글) 생성 |
| 환경변수 | `AI_ANALYSIS_MODE` |
| 현재 모드 | `fallback` (LLM 미사용, 메트릭 기반 규칙 템플릿) |
| LLM 모드 시 | `call_structured()` → MAIN_MODEL 사용 |
| 호출 함수 | `workspace_summary.generate_workspace_summary()` |

`AI_ANALYSIS_MODE=llm`이고 `GMS_API_KEY`가 설정된 경우에만 LLM 요약을 시도합니다(그
외에는 항상 fallback). LLM 호출이 예외를 던지면 fallback 규칙 요약으로 대체하고
`logger.warning()`을 남기며, 이때 응답 `mode`는 `"llm_fallback"`으로 표시됩니다.
LLM 호출 자체는 성공했지만 `summary`가 빈 문자열인 경우는 기존과 동일하게 fallback
문구로 채우되 `mode`는 `"llm"`을 유지합니다(구분: 호출 실패=`llm_fallback`, 호출
성공+빈 값=`llm`).

**건당 토큰 비용** (라이브 검증 1회차 기준, purpose=`workspace_summary` 집계):
- prompt_tokens 평균: `TODO(라이브 실행 후 기입)`
- completion_tokens 평균: `TODO(라이브 실행 후 기입)`

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
  ├─ [grounding 2차] call_grounding()    ← gemini-2.5-flash-lite
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
GROUNDING_MODEL=gemini-2.5-flash-lite # 그라운딩 판정 (thinking 모델은 max_tokens=64에서 truncate되어 사용 불가)
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

---

## 라이브 검증 실행법

`tests/live/`는 실제 GMS API를 호출해 `/chat` SSE 계약·페르소나 6종·멀티턴 rewrite·
provider 각 함수를 검증하는 라이브 E2E 하네스입니다. 기본 `python3 -m pytest`는
`pyproject.toml`의 `addopts = "-m 'not live'"`로 이 테스트들을 항상 제외하므로,
평소 테스트/CI에서는 과금이 발생하지 않습니다.

### 실행 명령

```bash
RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live
```

- `RUN_LIVE_LLM=1`과 비어있지 않은 `GMS_API_KEY`(`.env`)가 **모두** 있어야 실행됩니다.
  `GMS_API_KEY`가 `.env`에 이미 있어도 `RUN_LIVE_LLM=1`을 명시하지 않으면
  `tests/live/conftest.py`의 게이트가 전체 skip 처리합니다(우발적 과금 방지).
- `python3 -m pytest -m live -q tests/live --collect-only`로 실행 없이 테스트
  목록만 확인할 수 있습니다.

### 무엇을 하는가

1. tmp 디렉터리에 `VECTOR_STORE_PATH`/`METRICS_DIR`를 격리하고, 비용 상한을 위해
   `MAIN_MAX_TOKENS=256`으로 낮춥니다(`main_max_tokens` 설정값, 운영 기본값은 4096로
   불변).
2. `scripts/eval/seed.py::seed_workspace()`가 `tests/live/fixtures/corpus/*.md`
   (배포 절차/결제 API/인증·JWT 정책/장애 대응 runbook, 4개 가상 문서)를 새
   워크스페이스(`live-{uuid8}`)에 실 임베딩으로 인덱싱합니다.
3. `/chat` SSE 계약(turn1, 코퍼스 밖 질문 not-groundable, 멀티턴 rewrite, 페르소나
   6종)과 `provider.py`의 grounding/embedding/call_structured를 실제 GMS 응답으로
   검증합니다. 페르소나 답변은 품질 판정 없이 앞 200자만 결과 파일에 남깁니다.
4. `tests/live/test_analysis_live.py`는 이 모듈 범위에서만 `AI_ANALYSIS_MODE=llm`으로
   전환해(운영 기본값 `fallback`은 불변) 문서/커밋/워크스페이스 요약 분석 3종의 llm
   계약을 검증합니다. 단독 실행: `RUN_LIVE_LLM=1 python3 -m pytest -m live -q
   tests/live/test_analysis_live.py -s`.

### 비용 상한 설계

- 코퍼스 문서 4개(각 1~2KB), `top_k=5` 유지.
- `main_max_tokens=256`으로 메인 답변 생성 비용을 제한합니다.
- 1회 실행 예상 호출량: `/chat` 약 10회 × (main 입력 ~2k / 출력 ≤256 + grounding
  ~1k/10) + rewrite 2~3회 + 임베딩 ~20건 수준(제안 상한: 총 5만 토큰 이하).

### 결과 확인

세션 종료 시(`pytest_sessionfinish`) 이번 실행의 `llm_calls.jsonl`/`chat_metrics.jsonl`을
purpose×model별 토큰 합계와 지연 p50/p95로 집계해 터미널에 출력하고,
`data/live_runs/{YYYYMMDD-HHMMSS}.json`(`data/`는 gitignore 대상)에 저장합니다. 페르소나
답변 미리보기도 같은 파일에 포함되므로 수동 검토에 활용합니다.
