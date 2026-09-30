# 그라운딩 임계치 오프라인 평가 (A4)

## 목적

`app/core/rag/grounding.py`의 그라운딩 판정은 두 단계로 동작합니다.

1. **1차 필터**: `retriever`의 최고 유사도 점수가 `grounding_similarity_threshold`
   (기본값 `0.35`) 미만이면 LLM 호출 없이 `is_groundable=False`를 즉시 반환합니다.
2. **2차 판단**: `GROUNDING_MODEL`(경량 Gemini 모델)로 `{is_groundable, confidence}`를
   이진 판정합니다.

이 평가는 라벨셋(`scripts/eval/datasets/grounding_cases.jsonl`)으로 위 두 단계의
정확도를 측정해, 임계치 조정과 fail-open 계약(2차 판정 실패 시 `is_groundable=True`로
유사도 fallback)의 근거 데이터를 만듭니다. **이 문서/스크립트는 `app/config.py`의
기본 임계치를 변경하지 않습니다** — 값 변경은 이 결과를 근거로 한 별도 결정입니다.

특히 다음 구조적 이슈를 정량적으로 확인하는 것이 목표입니다.

- `chitchat`(일상 잡담)과 `general_tech`(범용 기술 지식)는 시스템 프롬프트 규칙상
  컨텍스트와 무관하게 `is_groundable=true`여야 하지만, 코퍼스와 유사도가 낮아 **1차
  필터에서 LLM 호출 전에 차단**될 수 있습니다. 즉 규칙 1·2가 1차 필터 뒤에서만
  동작하는 셈입니다. `filter_false_block_rate`(카테고리별)로 이 규모를 측정합니다.
- 2차 판정 실패(LLM 호출 예외/JSON 파싱 실패) 시 fail-open(`is_groundable=True`)이
  적용되는데, 이 비율(`parse_failure_rate`)과 그 영향을 함께 확인합니다.

## 실행 방법

### 1) 라이브 수집 (실 GMS API 호출, 비용 발생)

`scripts/eval/seed.py::seed_workspace()`로 tmp 워크스페이스를 만들고
`tests/live/fixtures/corpus/*.md`(배포 절차/결제 API/인증-JWT/장애 runbook 4개 문서)를
실 임베딩으로 인덱싱한 뒤, 데이터셋의 각 케이스에 대해 `retriever.retrieve(question,
workspace_id, db, top_k=5)`로 검색하고, **임계치와 무관하게 항상**
`provider.call_grounding()`을 호출해 결과를 기록합니다. 이렇게 하면 이후 임계치
스윕을 API 재호출 없이 오프라인으로 할 수 있습니다.

```bash
RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live
```

`RUN_LIVE_LLM=1`과 비어있지 않은 `GMS_API_KEY`가 모두 필요합니다(`tests/live/conftest.py`와
동일한 게이트 — `.env`가 자동 로드되므로 키 존재만으로는 실행되지 않습니다). 결과는
`data/eval/grounding-{YYYYMMDD-HHMMSS}.jsonl`에 저장됩니다(`data/`는 `.gitignore` 대상).

스모크(소량) 실행:

```bash
RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live --limit 5
```

### 2) 오프라인 임계치 스윕 (API 호출 없음)

```bash
python3 scripts/eval/grounding_eval.py --from-cache data/eval/grounding-20260930-120000.jsonl
```

임계치 `0.20`~`0.60`(`0.05` 간격)을 스윕해 markdown 표를 출력합니다. `--dataset`으로
다른 케이스 파일을, `--out`으로 라이브 수집 결과 저장 경로를 지정할 수 있습니다.

## 데이터셋

`scripts/eval/datasets/grounding_cases.jsonl` — 40건(카테고리별 8건), 전부 가상
한국어 데이터(PII 없음)이며 `tests/live/fixtures/corpus`의 실제 사실에 근거합니다.

| 필드 | 설명 |
|---|---|
| `id` | 케이스 식별자 |
| `category` | `chitchat` \| `general_tech` \| `project_answerable` \| `project_unanswerable` \| `ambiguous` |
| `question` | 평가 질문(이번 turn) |
| `history` | (선택) 이전 대화 `[{role, content}, ...]`. 멀티턴 시나리오 문서화 목적이며, `--live` 수집은 `query_rewriter`를 호출하지 않고 `question` 원문을 그대로 검색/판정에 사용합니다(현재 `assess()`의 실제 입력인 `request.content`와 동일한 방식 — rewrite 자체의 품질 평가는 A5 범위) |
| `expected_groundable` | 라벨(bool) |
| `note` | 라벨 근거/함정 포인트 설명 |

카테고리 설계:

- **chitchat**: 인사/감사/정체성 확인 등 일상 잡담. 항상 `true`.
- **general_tech**: 워크스페이스에 종속되지 않는 범용 기술 지식. 항상 `true`. 일부는
  코퍼스 용어(JWT, blue-green, rollout 등)와 겹치도록 설계해 유사도 상승 트랩을 포함.
- **project_answerable**: 코퍼스 4개 문서에 명시된 사실로 답변 가능한 질문. `true`.
- **project_unanswerable**: 사내 질문처럼 들리지만 코퍼스에 근거가 없는 질문. `false`.
- **ambiguous**: 대명사 기반 멀티턴 질문, 범용/프로젝트 혼합 질문, 키워드만 겹치는
  트랩 질문 등 경계 사례.

**라벨(`expected_groundable`)과 `note`는 에이전트 초안이며 사용자 검수 대기 상태입니다.**
실행 전 라벨 타당성을 재검토해 주세요.

## 지표 정의 (`compute_metrics(records, threshold)`)

`scripts/eval/grounding_eval.py`의 순수 함수이며 `tests/test_grounding_eval_metrics.py`가
합성 레코드로 단위 테스트합니다. 최종 판정은 `app/core/rag/grounding.py::assess()`와
동일한 결합 로직입니다.

```
final = (top_similarity >= threshold) AND effective_llm_is_groundable
```

`effective_llm_is_groundable`은 fail-open 계약을 반영합니다 — `fallback_reason`
(`llm_error` | `parse_error`)이 있으면 실제 LLM 응답과 무관하게 항상 `True`입니다.

| 지표 | 정의 |
|---|---|
| `accuracy` | `final == expected_groundable` 비율(전체) |
| `not_groundable.precision/recall/f1` | `not-groundable`(final=False)을 positive class로 한 정밀도/재현율/F1. UC-04 알림 품질의 핵심 지표(재현율이 낮으면 담당자 알림이 누락됨) |
| `filter_false_block_rate.overall` / `.by_category` | 라벨이 `groundable=true`인데 `top_similarity < threshold`라서 LLM 호출 전에 차단된 비율(전체 및 카테고리별) — `chitchat`/`general_tech`에서 특히 높을 것으로 예상 |
| `llm_call_saving_rate` | `top_similarity < threshold`인 케이스 비율. 1차 필터가 LLM 호출을 절감하는 비율 |
| `llm_only_agreement` | 1차 필터를 무시하고 LLM 2차 판정(fail-open 반영)만으로 라벨과 일치하는 비율 |
| `parse_failure_rate` | `parse_ok=False`(2차 판정 예외 또는 JSON 파싱 실패) 비율 |
| `confidence_bucket_accuracy` | LLM이 유효하게 응답한 케이스(fallback 제외)를 confidence 구간(`0.0-0.5`/`0.5-0.8`/`0.8-1.0`)으로 나눈 정답률(보정 확인용) |
| `mean_prompt_tokens` / `mean_completion_tokens` / `mean_latency_ms` | 케이스당 그라운딩 LLM 호출 비용/지연 평균 |

## 1회차 결과

- 실행일: 2026-09-30
- 그라운딩 모델(`GROUNDING_MODEL`): `gemini-2.5-flash-lite`
- 데이터셋 버전: `scripts/eval/datasets/grounding_cases.jsonl` (2026-09-30, 40건, 라벨 사용자 검수 전)
- 코퍼스: `tests/live/fixtures/corpus` 4개 문서 (소형 코퍼스라 운영 분포와 다를 수 있음)

### 임계치 스윕

| threshold | accuracy | not-gr precision | not-gr recall | not-gr F1 | filter false-block | LLM 호출 절감률 | LLM 단독 일치율 | 파싱 실패율 |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 85.0% | 100.0% | 50.0% | 66.7% | 0.0% | 0.0% | 85.0% | 5.0% |
| 0.25 | 85.0% | 100.0% | 50.0% | 66.7% | 0.0% | 0.0% | 85.0% | 5.0% |
| 0.30 | 85.0% | 100.0% | 50.0% | 66.7% | 0.0% | 0.0% | 85.0% | 5.0% |
| **0.35 (현행)** | 85.0% | 100.0% | 50.0% | 66.7% | 0.0% | 0.0% | 85.0% | 5.0% |
| 0.40 | 85.0% | 100.0% | 50.0% | 66.7% | 0.0% | 0.0% | 85.0% | 5.0% |
| 0.45 | 82.5% | 85.7% | 50.0% | 63.2% | 3.6% | 2.5% | 85.0% | 5.0% |
| 0.50 | 62.5% | 40.0% | 50.0% | 44.4% | 32.1% | 22.5% | 85.0% | 5.0% |
| 0.55 | 60.0% | 37.5% | 50.0% | 42.9% | 35.7% | 25.0% | 85.0% | 5.0% |
| 0.60 | 52.5% | 31.6% | 50.0% | 38.7% | 46.4% | 32.5% | 85.0% | 5.0% |

### 카테고리별 top 유사도 분포 / filter false-block rate

| category | 유사도 min | median | max | false-block @0.45 | @0.50 |
|---|---|---|---|---|---|
| chitchat | 0.449 | 0.481 | 0.539 | 12.5% | 87.5% |
| general_tech | 0.498 | 0.632 | 0.693 | 0.0% | 12.5% |
| ambiguous | 0.473 | 0.698 | 0.757 | 0.0% | 25.0% |
| project_answerable | 0.623 | 0.718 | 0.770 | 0.0% | 0.0% |
| project_unanswerable | 0.624 | 0.698 | 0.775 | - | - |

### confidence 구간별 정답률 / 평균 토큰·지연

- confidence 0.8~1.0: 32건 중 27건 정답(84.4%), 0~0.5: 6건 중 6건 정답, 0.5~0.8: 0건
- LLM 오류(fail-open) 2건(a001, a008) → 파싱 실패율 5.0%
- 판정 1건당 입력 평균 약 1,991 토큰, 지연 p50 1.03s / p95 1.14s
- 모든 호출에서 thinking 토큰 1~2가 발생했으나 `finishReason=STOP`으로 잘림은 없음

### 해석

1. **1차 유사도 필터가 사실상 동작하지 않음**: 잡담조차 top 유사도가 0.449 이상이라 현행 0.35에서는 LLM 호출 절감이 0%다. 모든 질문이 LLM 판정까지 간다.
2. **임계치를 올려도 얻는 것이 없음**: 0.45에서 절감 2.5%에 정확도 -2.5%p, 0.50부터는 잡담·일반 기술 질문을 대량 오차단한다. 도메인 질문(answerable/unanswerable)은 유사도 분포가 겹쳐 유사도만으로는 분리할 수 없다.
3. **LLM 판정의 약점은 "주제는 코퍼스와 같지만 답은 없는" 질문**: not-groundable recall 50%. 오판 5건 모두 confidence 0.8~0.9로 높게 응답해 confidence로는 걸러낼 수 없다(보정 안 됨).

### 권고

- 임계치 기본값 0.35는 **유지**(변경 이득 없음). 유사도 필터는 비용 절감 장치가 아니라 "검색 결과 없음" 방어선으로 보는 것이 맞다.
- 개선 여지는 LLM 판정 쪽: 판정 프롬프트에 "질문의 구체적 사실(수치·주체·절차)이 컨텍스트에 명시되어 있을 때만 true" 규칙과 반례 few-shot을 추가하는 후속 WP를 제안한다.
- 판정 입력(약 2k 토큰)이 호출 비용의 대부분이므로, 컨텍스트를 top-3 또는 청크 요약으로 줄이는 실험도 같은 WP에서 비교한다.
- 매 호출 thinking 1~2 토큰으로 `call_grounding` 경고가 과다하게 출력된다. 경고 조건을 `MAX_TOKENS`일 때로 좁히는 것을 후속으로 검토한다.

## 2회차: 판정 변형 비교 (G1)

1회차 권고(판정 프롬프트 강화, 컨텍스트 축소)를 같은 40건 데이터셋에서 비교하기 위한
도구입니다. **기본 설정은 변경되지 않았으며(`grounding_prompt_version=v1`,
`grounding_judge_top_k=5`, `grounding_judge_max_chunk_chars=0`), 운영 기본값 전환은 결과
검토 후 별도 승인이 필요합니다.**

### 변형 정의 (`scripts/eval/grounding_eval.py::VARIANTS`)

| variant | 프롬프트 | 판정 청크 수(top_k) | 청크 길이 상한 |
| --- | --- | --- | --- |
| `baseline` | `v1` (현행) | 5 | 없음 |
| `strict` | `v2-strict` | 5 | 없음 |
| `top3` | `v1` | 3 | 없음 |
| `strict_top3` | `v2-strict` | 3 | 없음 |
| `strict_top3_cap` | `v2-strict` | 3 | 600자 |

- 프롬프트는 `app/core/rag/grounding_prompts.py`(`v2-strict`는 규칙 3을 "질문이 요구하는
  구체적 사실이 컨텍스트에 명시된 경우에만 true"로 강화하고 반례 few-shot 2개 추가).
- 검색은 케이스당 1회(top_k=5)만 수행하고, 변형마다 판정 프롬프트를 달리 만들어
  `call_grounding`을 호출합니다(비용은 변형 수에 비례).

### 실행 방법

```bash
# 라이브 수집 (비용 발생, RUN_LIVE_LLM=1 + GMS_API_KEY 필요). --variants 기본값은 전체
RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live
RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live --variants baseline,strict

# 오프라인 비교 (API 호출 없음). 임계치 기본값은 settings.grounding_similarity_threshold(0.35)
python3 scripts/eval/grounding_eval.py --from-cache data/eval/grounding-{timestamp}.jsonl
python3 scripts/eval/grounding_eval.py --from-cache data/eval/grounding-{timestamp}.jsonl --threshold 0.35
```

캐시 레코드에는 `variant`, `prompt_chars` 필드가 추가되며, `variant`가 없는 기존(A4)
캐시는 `baseline`으로 읽습니다.

### 출력 표

- baseline 임계치 스윕(기존과 동일)
- 변형 비교: variant / accuracy / not-gr precision / not-gr recall / not-gr F1 /
  groundable 오차단(expected groundable인데 최종 false인 건수·비율) / 평균 prompt_tokens /
  평균 latency_ms / 파싱 실패율
- 변형 간 판정이 갈린 케이스 표(id, category, expected, variant별 정답 여부)

### 결과

미실행
