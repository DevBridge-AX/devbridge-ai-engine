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

> 아래 표는 `RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live` 실행 후
> `--from-cache`로 스윕한 결과를 채워 넣습니다(주 에이전트가 실행 예정).

- 실행일: TBD
- 그라운딩 모델(`GROUNDING_MODEL`): TBD
- 데이터셋 버전: `scripts/eval/datasets/grounding_cases.jsonl` (2026-09-30, 40건, 라벨 사용자 검수 전)
- 결과 캐시 파일: TBD (`data/eval/grounding-{ts}.jsonl`)

### 임계치 스윕

| threshold | accuracy | not-gr precision | not-gr recall | not-gr F1 | filter false-block | LLM 호출 절감률 | LLM 단독 일치율 | 파싱 실패율 |
|---|---|---|---|---|---|---|---|---|
| 0.20 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.30 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.35 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.40 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.45 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.55 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.60 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 카테고리별 filter false-block rate

TBD (스크립트 출력의 두 번째 표를 그대로 붙여 넣기)

### confidence 구간별 정답률 / 평균 토큰·지연

TBD

### 권고

TBD — 위 결과를 근거로 임계치 변경 여부, chitchat/general_tech 1차 필터 우회(별도
WP) 필요 여부를 결정합니다. 이 PR에서는 기본값을 변경하지 않습니다.
