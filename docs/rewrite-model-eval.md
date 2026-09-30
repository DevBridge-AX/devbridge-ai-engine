# REWRITE_MODEL 선택 실험

## 목적

멀티턴 대화(turn 2+)에서 사용자의 생략된 질문을 독립적인 검색 쿼리로 재구성하는
`REWRITE_MODEL`(`app/core/llm/query_rewriter.py`) 후보를 품질·지연·토큰 기준으로
비교하여 `.env`의 최종값을 확정한다. 이 문서는 실행법과 지표 정의, 결과를
기록하기 위한 것이며 **코드 기본값(`app/config.py`의 `rewrite_model`)은 이 문서의
1회차 결과만으로 변경하지 않는다** — 사용자(주 에이전트)가 결과를 검토한 뒤
결정한다.

## 실행 방법

```bash
# 레포 루트에서 실행 (scripts 패키지가 editable install에 포함되어 있지 않아
# -m 없이 직접 실행하면 ModuleNotFoundError가 발생한다)
python3 -m scripts.eval.rewrite_eval \
  --models claude-sonnet-4-6,claude-haiku-4-5-20251001,gemini-2.5-flash-lite
```

옵션:

| 옵션 | 설명 |
| --- | --- |
| `--models a,b,c` | 쉼표로 구분된 `REWRITE_MODEL` 후보 목록. 기본값: `claude-sonnet-4-6,claude-haiku-4-5-20251001,gemini-2.5-flash-lite` |
| `--limit N` | 데이터셋 앞에서부터 N건만 사용 (스모크용) |
| `--no-retrieval` | top-3 문서 적중률을 측정하지 않음 — 시드 워크스페이스 인덱싱(임베딩 API 호출)을 생략해 비용을 줄인다 |
| `--from-cache <file>` | 실 API를 호출하지 않고 저장된 `data/eval/rewrite-*.jsonl` 결과 파일로 오프라인 재집계만 수행 |
| `--dataset <path>` | 평가 데이터셋 경로 (기본값: `scripts/eval/datasets/rewrite_cases.jsonl`) |

**⚠️ 실 GMS API를 호출한다.** 모델 1개당 데이터셋 25건 × rewrite 1회(turn 2+만
포함되어 있어 전부 LLM 호출) + (`--no-retrieval` 미지정 시) 코퍼스 4개 문서 임베딩
인덱싱 1회(모델 수와 무관하게 실행당 1회만). 비용을 줄이려면 `--limit`으로 먼저
소규모 스모크를 돌린 뒤 전체를 실행한다.

결과는 `data/eval/rewrite-{YYYYMMDD-HHMMSS}.jsonl`에 저장된다(`data/`는
`.gitignore` 대상). 레코드 원문에는 프롬프트 전체가 아닌 질문/재작성 결과만
포함되며, `--from-cache`로 이 파일을 다시 읽어 표를 재출력할 수 있다.

## 데이터셋

`scripts/eval/datasets/rewrite_cases.jsonl` — 25건, `tests/live/fixtures/corpus`의
4개 가상 사내 문서(결제 API, 인증/JWT 정책, 배포 절차, 장애 대응 runbook) 사실을
기반으로 작성된 가상 데이터(PII 없음).

| 카테고리 | 건수 | 설명 |
| --- | --- | --- |
| `pronoun_restoration` | 10 | 이전 대화의 주어/대상을 대명사("그거", "거기")로 생략한 후속 질문 |
| `topic_switch` | 5 | 이전 대화와 무관한 새 주제로 완전히 전환하는 질문(이전 주제 키워드가 섞여 들어가면 안 됨) |
| `chitchat_passthrough` | 5 | 검색이 필요 없는 잡담/리액션 — 재작성 없이 원문 그대로 반환해야 함 |
| `already_standalone` | 5 | 이력이 있어도 질문 자체가 이미 맥락 없이 독립적인 경우 |

필드:

| 필드 | 설명 |
| --- | --- |
| `id`, `category` | 케이스 식별자, 카테고리 |
| `history` | `[{"role": "user"/"assistant", "content": "..."}]` — `query_rewriter.rewrite()`의 `conversation_history` 인자와 동일 형식 |
| `question` | 현재 사용자 입력 |
| `expected_keywords` | 재작성 결과에 포함되어야 하는 키워드 목록 (chitchat은 빈 배열) |
| `expect_passthrough` | `true`면 재작성 결과가 (strip 후) 원문과 동일해야 함 |
| `expected_doc` | 재작성된 쿼리로 검색 시 top-3에 포함되어야 하는 코퍼스 문서 파일명 stem (예: `payment-api`), 없으면 `null` |

## 지표 정의 (`scripts/eval/rewrite_metrics.py`)

- **답변형 출력률 (`answer_like_rate`)**: `is_answer_like(original, rewritten)`가
  True인 비율. 재작성 결과가 "쿼리"가 아니라 질문에 대한 "답변"처럼 나온 경우를
  탐지하는 휴리스틱(구현: `app/core/llm/rewrite_guard.py`)이다. 출력이 원문과
  (strip 후) 같으면(잡담 passthrough) 답변형에서 제외하며, 그 외에 아래 중 하나라도
  해당하면 답변형으로 판정한다.
  - 길이가 원래 질문의 4배 초과(단, 하한 40자) 또는 80자 초과
  - 줄바꿈 포함
  - "~니다." 또는 "~니다"로 끝남(입니다/습니다/합니다/됩니다 등 서술형 종결)
  - 출력 전체가 따옴표류 문자로 감싸져 있음(쿼리 중간의 부분 따옴표는 제외)
  - "출력:" 또는 "재작성:" 접두어로 시작
  - 주의: 아래 1회차 수치는 구 정의(입니다/습니다 종결, 4배 규칙 하한 없음, 따옴표 포함 여부, 원문 동일 제외 없음)로 측정되었다.
  - 평가 스크립트(`scripts/eval/rewrite_eval.py`)는 `rewrite(..., guard=False)`로 호출하여
    서비스 경로의 원문 fallback 가드를 우회한다. 가드를 거치면 답변형 출력이 원문으로
    바뀌어 이 지표가 구조적으로 0%가 되기 때문이다.
- **키워드 적중률 (`keyword_hit_rate`)**: 케이스별 `expected_keywords` 중 재작성
  결과에 (대소문자 무시) 포함된 비율의 평균. `expected_keywords`가 빈 케이스는
  1.0(만점)으로 집계된다(검증할 키워드가 없으므로).
- **passthrough 정확도 (`passthrough_accuracy`)**: `expect_passthrough=true`인
  케이스 중 재작성 결과가 (strip 후) 원문과 완전히 동일한 비율. 해당 케이스가
  없으면 `null`.
- **top-3 문서 적중률 (`doc_hit_rate`)**: `expected_doc`이 있고 검색을 수행한
  케이스 중, 재작성된 쿼리로 `retriever.retrieve(top_k=3)`를 호출했을 때
  `expected_doc`과 제목이 일치하는 청크가 top-3에 포함된 비율. `--no-retrieval`로
  실행하면 전부 `null`(미측정).
- **지연 (`latency_p50_ms`, `latency_p95_ms`)**: 케이스별 `query_rewriter.rewrite()`
  호출 왕복 시간(정상 응답 케이스만)의 50/95 백분위수.
- **평균 토큰 (`mean_prompt_tokens`, `mean_completion_tokens`)**: 정상 응답
  케이스의 `LLMUsage.prompt_tokens` / `completion_tokens` 평균.
- **에러 수 (`error_count`)**: `query_rewriter.rewrite()` 호출 중 예외가 발생한
  케이스 수. 에러 케이스는 위 품질/지연/토큰 지표 계산에서 제외되고 `error_count`
  에만 반영된다.

## 결정 규칙

- **품질 지표(답변형 출력률, 키워드 적중률, passthrough 정확도, top-3 문서
  적중률)가 후보 간 사실상 동률이면, 지연·토큰이 가장 낮은(저비용) 모델을
  우선한다.**
- **답변형 출력률이 5% 이상인 모델이 있으면**, 해당 모델을 그대로 채택하지 않거나,
  채택 시 `query_rewriter._SYSTEM_PROMPT`에 "절대 질문에 답하지 마세요" 규칙과
  실패 예시 few-shot 1개를 추가하고 `call_rewrite`의 `max_tokens`를 512→128로
  낮추는 프롬프트 보강을 후속 작업으로 진행한다(이번 PR 범위 밖 — 결과를 보고
  주 에이전트가 결정).
- `REWRITE_MODEL` 최종값, `grounding.assess()`에 원문 대신 `rewritten_query`를
  넘길지 여부, `CLAUDE.md` §2 / `.env.example` / `docs/ai-api-usage.md` /
  `docs/architecture.md`의 REWRITE_MODEL 표기 갱신은 이 문서의 결과를 근거로
  별도 PR에서 결정한다.

## 결과 (1회차)

**실행 정보**

| 항목 | 값 |
| --- | --- |
| 실행일 | 2026-09-30 |
| 데이터셋 버전 | `scripts/eval/datasets/rewrite_cases.jsonl` (25건, 2026-09-30 작성) |
| 결과 파일 | `data/eval/rewrite-20260930-215257.jsonl` |
| 실행 커맨드 | `python3 -m scripts.eval.rewrite_eval --models claude-sonnet-4-6,claude-haiku-4-5-20251001,gemini-2.5-flash-lite,gpt-5.4-nano` |

후보 3종 외에 로컬 환경에서 사용 중이던 `gpt-5.4-nano`를 비교 대상으로 추가했다.

**모델별 비교**

| model | 케이스 | 에러 | 답변형 출력률 | 키워드 적중률 | passthrough 정확도 | top-3 문서 적중률 | 지연 p50(ms) | 지연 p95(ms) | 평균 prompt 토큰 | 평균 completion 토큰 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-haiku-4-5-20251001 | 25 | 0 | 0.00 | 0.62 | 1.00 | 1.00 | 831 | 1020 | 651 | 20.2 |
| claude-sonnet-4-6 | 25 | 0 | 0.00 | 0.64 | 1.00 | 1.00 | 1709 | 2692 | 652 | 22.8 |
| gemini-2.5-flash-lite | 25 | 1 (429) | 0.00 | 0.65 | 0.60 | 1.00 | 955 | 1073 | 382 | 9.2 |
| gpt-5.4-nano | 25 | 1 (timeout) | 0.12 | 0.67 | 0.60 | 1.00 | 1271 | 1868 | 385 | 21.6 |

**관찰**

- top-3 문서 적중률은 4개 모델 모두 100%로, 소형 코퍼스(4문서)에서는 검색 결과 차이가 나지 않았다. 키워드 적중률 차이(0.62~0.67)도 오차 범위 수준이다.
- 차이는 **잡담 passthrough**에서 났다. `gemini-2.5-flash-lite`와 `gpt-5.4-nano`는 "오 알겠어요 감사합니다", "ㅋㅋㅋ 넵" 같은 잡담을 이전 대화 주제로 바꿔 검색 쿼리를 만들었다(각 2/5건). 이 경우 잡담에도 문서 검색과 그라운딩 판정 호출이 발생한다.
- `gpt-5.4-nano`는 답변형 출력률 12%(3건)로 결정 규칙의 5% 기준을 넘었다. 쿼리 대신 절차나 개념 설명을 나열했다(openai 분기는 `max_tokens`를 보내지 않아 길이 제한도 없다).
- `claude-haiku-4-5`는 `claude-sonnet-4-6`과 품질이 같고, 지연 p50은 약 2.1배 빠르다(831ms vs 1709ms). p95도 1.0s vs 2.7s다.

**권고**

- `REWRITE_MODEL=claude-haiku-4-5-20251001` 채택을 권고한다. 품질 동률이면 저비용·저지연 모델을 우선하는 규칙에 해당한다.
- `gpt-5.4-nano`는 답변형 출력과 잡담 변형 때문에 부적합하다. 로컬 `.env` 값도 교체를 권고한다.
- 채택 모델의 답변형 출력률이 0%이므로 프롬프트 보강은 지금 필수는 아니다. 다만 모델 교체에 대비한 방어 코드(답변형 출력이면 원문 fallback)는 저비용으로 넣을 가치가 있어 후속으로 제안한다.
- 코드 기본값(`app/config.py`의 `rewrite_model`), `.env.example`, 관련 문서의 모델명 변경은 승인 후 별도 커밋으로 진행한다.
- 2026-10-01: 답변형/빈 출력 시 원문 fallback 가드를 `app/core/llm/rewrite_guard.py`로 추가했다.
- 2026-10-01: 승인에 따라 코드 기본값을 `claude-haiku-4-5-20251001`로 변경했다.

## 범위에서 제외한 것

- `chat_pipeline.grounding.assess()`에 원문 대신 `rewritten_query`를 넘기는 비교
  (그라운딩 입력 쿼리 비교)는 이번 스크립트 범위에 포함하지 않는다. 결정이
  필요하면 별도 실험으로 진행한다.
- 실제 `REWRITE_MODEL` 기본값 변경, 프롬프트 보강(few-shot 추가, max_tokens
  조정), `.env.example` / `CLAUDE.md` / `docs/ai-api-usage.md` / `docs/architecture.md`
  갱신은 결과 확인 후 별도로 진행한다.
