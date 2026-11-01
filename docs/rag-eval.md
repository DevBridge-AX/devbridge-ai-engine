# RAG 품질 검증 문서 셋 (X2)

## 목적

가상 문서 10건과 질의 30건으로 (1) 검색 품질(recall@k, MRR, hit@1)과 (2) 접근 제어
(task / sensitivity 필터) 정확성을 반복 측정한다. 데이터셋과 지표 계산은 API 없이 동작하며,
실제 검색 실행(`--live`)만 임베딩 API를 호출한다. 운영 문서나 실제 개인정보는 포함하지 않는다.

## 데이터셋 구성

위치: `scripts/eval/datasets/rag_docset/` (`manifest.json` + 마크다운 40건). 아래 표는 1회차 문서 10건이고, 2회차(2026-10-28)에 30건을 추가했다(§2회차 확장 참고).

| doc_key | 유형 | task_id | 민감도 | 주제 |
| --- | --- | --- | --- | --- |
| code-review-rules | 공개 | - | normal | 코드 리뷰 규칙 |
| notification-api | 공개 | - | normal | 알림 서비스 API 명세 |
| order-schema | 공개 | - | normal | 주문 DB 스키마 |
| batch-staging | 공개 (distractor) | - | normal | 스테이징 배치 스케줄 |
| batch-prod | 공개 (distractor) | - | normal | 운영 배치 스케줄 |
| log-retention | 공개 | - | normal | 로그 보존 정책 |
| settlement-design | task 한정 | task-billing | normal | 결제 정산 과제 설계 |
| retro-billing | task 한정 | task-billing | normal | 정산 과제 스프린트 3 회고 |
| onboarding | task 한정 | task-onboarding | normal | 신규 입사자 온보딩 체크리스트 |
| hr-payroll-schema | 제한 | - | restricted | 인사 급여 테이블 스키마 |

`batch-staging` / `batch-prod`는 구조는 같고 수치(시각, 타임아웃, 재시도, 서버 대수)만 다른
distractor 쌍이다. 기존 `tests/live/fixtures/corpus`와 주제가 겹치지 않는다.

### 2회차 확장 (30건 추가, 총 40건)

같은 주제에서 환경·대상만 다른 distractor 계열을 늘려 순위 마진을 압박하고, 청크가 여러 개인 긴 문서를 넣었다.

| 계열 | doc_key | 접근 |
| --- | --- | --- |
| 배포 절차 | deploy-dev, deploy-staging, deploy-prod | 공개 |
| 로그 보관 | log-retention(기존, 앱 로그), log-retention-audit, log-retention-access | 공개 |
| 결제 재시도 | payment-retry-card, payment-retry-bank | 공개 |
| 온보딩 | onboarding-backend, onboarding-frontend, onboarding-qa | 공개 |
| API 가이드 | search-api, payment-api, file-upload-api | 공개 |
| 장애 runbook | runbook-db, runbook-queue, runbook-cache | 공개 |
| 보안 정책 | password-policy, token-policy, vpn-policy | 공개 |
| 회의록 | meeting-2026-09-w1, meeting-2026-09-w2, meeting-2026-10-w1 | task-billing |
| 정산 | settlement-daily, settlement-monthly | task-billing |
| 인사 | hr-benefits, hr-leave-policy | restricted |
| 긴 문서 | architecture-overview(8,013자), incident-postmortem-2026-08(6,175자), release-notes-2026(6,027자), api-style-guide(6,038자) | 공개 |

모든 문서는 가상이며 PII가 없다. 문서마다 수치·고유명사를 다르게 두어 distractor를 구별할 근거를 만들었다.

## 케이스 카테고리

위치: `scripts/eval/datasets/rag_cases.jsonl` (80건. 1회차 r001~r030 + 2회차 r031~r080)

| category | 건수 (1회차 → 2회차) | 설명 |
| --- | --- | --- |
| single_doc | 12 → 30 | 문서 1건이 정답, 전체 권한. 절반은 패러프레이즈·약어 표현 |
| cross_doc | 3 → 10 | 2개 문서가 모두 필요 |
| distractor | 4 → 15 | 한쪽 문서가 정답, 같은 계열 나머지는 forbidden(1건 이상) |
| acl_task | 5 → 10 | task 문서 질문. 권한 있음 / 없음(빈 목록, 다른 task) 쌍 |
| acl_restricted | 3 → 6 | restricted 문서. 열람 가능 / 불가 |
| out_of_corpus | 3 → 9 | 코퍼스 밖 질문. top similarity만 기록 |

긴 문서를 대상으로 한 8건(single_doc 6, cross_doc 2)은 `expected_substrings`(정답 청크에만 있는 문구)를 가져 청크 단위 적중을 채점한다.

각 케이스의 `access`(`accessible_task_ids`, `can_view_restricted`)는 `retriever.AccessFilter`로
변환되어 검색에 적용된다. 접근 불가 케이스는 `expected_doc_keys: []`이고 해당 문서는
`forbidden_doc_keys`에 들어간다.

## 지표 정의

- recall@k: 상위 k개 문서(청크가 아닌 문서 단위, 중복 제거) 중 expected 문서 비율
- hit@1: 1위 문서가 expected에 속하는 비율
- MRR: expected 중 가장 먼저 나온 문서의 1/rank 평균 (못 찾으면 0)
- ACL leak: acl_task / acl_restricted 케이스에서 forbidden(접근 불가) 문서가 검색 결과에 하나라도 나타난 케이스 수 (0이어야 함)
- forbidden 노출: 카테고리와 무관하게 forbidden 문서가 검색 결과에 나타난 케이스 수. distractor의 forbidden은
  접근 가능한 유사 문서라 leak이 아니며 순위 혼동 참고용이다(판정은 distractor hit@1).
- out_of_corpus top similarity: 최상위 유사도 평균/최대 (코퍼스 밖 질문이 얼마나 높게 매칭되는지)
- expected가 비어 있는 케이스(접근 거절, out_of_corpus)는 recall/MRR/hit@1 집계에서 제외한다.
- 청크 hit@k (2회차 추가): `expected_substrings`가 있는 케이스만 채점. 상위 k 청크 중 하나라도 정답 문구를 포함하면 1. 결과 파일에는 청크 원문 대신 포함 여부(`substring_hit`)만 저장한다.
- 마진 (2회차 추가): RRF 1위 청크와 2위 청크의 **벡터 유사도** 차이. 순위는 BM25를 합친 RRF 순서라 1위의 벡터 유사도가 2위보다 낮으면 음수가 된다(BM25가 끌어올린 경우). distractor는 정답이 1위인 케이스의 마진과 forbidden이 2위인 케이스의 마진을 따로 본다.

## 실행법

```bash
# live: 임베딩 API 호출 (RUN_LIVE_LLM=1 필수)
RUN_LIVE_LLM=1 PYTHONPATH=. python3 scripts/eval/rag_eval.py --live [--limit N] [--top-k 5]

# 저장된 결과 집계 (API 호출 없음)
PYTHONPATH=. python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-YYYYmmdd-HHMMSS.jsonl
```

- 비용: 임베딩만 발생한다. 시드 40문서(청크 단위 임베딩, 약 130청크) + 질의 80건(질의당 1회). LLM 호출은 없다.
- `--manifest PATH --cases PATH`로 다른 문서 셋·케이스 파일을 지정할 수 있다.
- `--rerank`(`--live` 전용, `RERANK_ENABLED=true` 필요): 후보 `RERANK_CANDIDATES`건을 가져와 `app/core/rag/reranker.py`로 재정렬한 결과(`reranked`)를 원래 순서(`retrieved`)와 함께 저장한다. 케이스당 리랭크 모델 호출 1회가 추가된다. `--rerank-interval SEC`는 케이스 간 대기, `--rerank-retries N`은 429(요청 제한) 시 15·30·60초 백오프 재시도 횟수다(평가 전용, 앱 경로는 재시도 없이 fail-open). `--from-cache`는 `reranked`가 있으면 기준선·리랭크·차이 표를 함께 출력한다.
- `--live`는 tmp 디렉터리에 sqlite DB와 벡터스토어를 만들어 운영 데이터와 격리하며,
  결과는 `data/eval/rag-{YYYYmmdd-HHMMSS}.jsonl`에 저장된다.
- 오프라인 검증: `python -m pytest tests/test_rag_docset.py tests/test_rag_eval_metrics.py`
- 배관 스모크: `tests/test_rag_eval_smoke.py`가 임베딩을 모킹해 `--live` 경로(시딩, retrieve, 캐시, 리포트)를 오프라인으로 검증한다(품질 아님, plumbing only).

## 채택/판정 기준 제안

- single_doc recall@5 >= 90%
- ACL leak = 0 (acl_task, acl_restricted 전체)
- distractor hit@1 >= 75%
- out_of_corpus top similarity는 판정 기준 없이 기록하고, 그라운딩 임계치 검토 시 참고한다.

## 결과

### 1회차 - 실험 환경 기준

모든 수치는 **실험 환경(가상 문서 10건, 질의 30건) 기준**이다.

- 실행일: 2026-10-27
- 원본 파일: `data/eval/rag-20261027-153819.jsonl` (`data/`는 `.gitignore` 대상)
- 임베딩 모델: `gemini-embedding-2` (`v1`)
- 임베딩 task type: 미지정 (현행 embedder 기본값, `EMBEDDING_QUERY_TASK_TYPE` 미설정)
- 명령: `RUN_LIVE_LLM=1 PYTHONPATH=. python3 scripts/eval/rag_eval.py --live --top-k 5`
- 리포트: `PYTHONPATH=. python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-<timestamp>.jsonl`

#### 전체 요약

| 지표 | 목표 (채택/판정 기준) | 실측 | 달성 여부 |
| --- | --- | --- | --- |
| single_doc recall@5 | >= 90% | 100.0% (12/12) | 달성 |
| ACL leak (acl_task + acl_restricted만) | = 0 | 0 | 달성 |
| forbidden 노출 (전체 카테고리, 참고용) | 판정 기준 없음 (distractor는 순위 혼동 참고) | 4 (모두 distractor, 2위) | - |
| distractor hit@1 | >= 75% | 100.0% (4/4) | 달성 |
| out_of_corpus top similarity 최대 | 판정 기준 없음 (그라운딩 임계치 검토 시 참고) | 0.589 | - |

#### 카테고리별

| 카테고리 | n | 채점 n | recall@5 | MRR | hit@1 | ACL leak | forbidden 노출 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| single_doc | 12 | 12 | 100.0% | 1.000 | 100.0% | - | 0 |
| cross_doc | 3 | 3 | 100.0% | 1.000 | 100.0% | - | 0 |
| distractor | 4 | 4 | 100.0% | 1.000 | 100.0% | - | 4 |
| acl_task | 5 | 2 | 100.0% | 1.000 | 100.0% | 0 | 0 |
| acl_restricted | 3 | 1 | 100.0% | 1.000 | 100.0% | 0 | 0 |
| out_of_corpus | 3 | 0 | - | - | - | - | 0 |
| overall | 30 | 22 | 100.0% | 1.000 | 100.0% | 0 | 4 |

out_of_corpus top similarity: 평균 0.568 / 최대 0.589

#### 결론

- 판정 기준 3개(single_doc recall@5, ACL leak, distractor hit@1)를 모두 달성했다. 채점 대상 22건 전부 정답 문서가 1위(MRR 1.000).
- distractor 4건은 모두 정답 1위, forbidden 문서 2위였다. 1·2위 유사도 차이는 0.041~0.066으로 크지 않아, 문서가 늘면 순위가 뒤집힐 여지가 있다.
- ACL 필터는 검색 단계에서 접근 불가 문서를 완전히 배제했다(leak 0).
- out_of_corpus 최대 유사도 0.589는 그라운딩 1차 필터 임계치(`GROUNDING_SIMILARITY_THRESHOLD` 0.35)보다 높다. 유사도만으로는 범위 밖 질의를 거를 수 없고 LLM 판정(v2-strict)에 의존한다.

#### 후속

- recall 미달이 없어 리랭커(R-2)·재검색(R-3) 도입 근거는 현재 문서 셋 규모에서는 없다. 문서 셋이 10건으로 작아 천장 효과 가능성이 있으므로, 문서 수를 늘린 2회차에서 distractor 순위 마진을 다시 본다.
- 그라운딩 유사도 임계치 상향 검토 시 out_of_corpus 최대 0.589와 정답 1위 유사도 분포를 함께 비교한다(임계치만 올리면 정답 질의도 탈락할 수 있음).

### 2회차 (문서 40·질의 80) - 실험 환경 기준

리랭커(E3b)·청크 문맥 보강(E3c) 비교용 기준선이다. 판정 기준은 적용하되 목표는 "개선 여지 측정"이다.

- 실행일: 2026-10-28
- 원본 파일: `data/eval/rag-v2-round2.jsonl` (`data/`는 `.gitignore` 대상)
- 임베딩 모델: `gemini-embedding-2` (`v1`), taskType 미지정, 하이브리드(BM25 + 벡터, RRF), top-k 5
- 명령: `RUN_LIVE_LLM=1 PYTHONPATH=. python3 scripts/eval/rag_eval.py --live --top-k 5`

#### 전체 요약

| 지표 | 목표 | 1회차 (문서 10) | 2회차 (문서 40) | 달성 여부 |
| --- | --- | --- | --- | --- |
| single_doc recall@5 | >= 90% | 100.0% | 93.3% (28/30) | 달성 |
| ACL leak | = 0 | 0 | 0 | 달성 |
| distractor hit@1 | >= 75% | 100.0% | 93.3% (14/15) | 달성 |
| overall recall@5 / MRR / hit@1 | 참고 | 100% / 1.000 / 100% | 95.9% / 0.915 / 86.9% (채점 61건) | - |
| 청크 hit@5 (8건) | 참고 | - | 100.0% | - |
| out_of_corpus top similarity 최대 | 참고 | 0.589 | 0.662 (평균 0.578) | - |

#### 카테고리별

| 카테고리 | n | 채점 n | recall@5 | MRR | hit@1 | ACL leak | forbidden 노출 | 청크 hit@5 | 마진 평균/최소 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_doc | 30 | 30 | 93.3% | 0.878 | 83.3% | - | 0 | 100.0% (6) | 0.074/-0.017 |
| cross_doc | 10 | 10 | 95.0% | 1.000 | 100.0% | - | 0 | 100.0% (2) | 0.029/-0.037 |
| distractor | 15 | 15 | 100.0% | 0.967 | 93.3% | - | 13 | - | 0.084/-0.013 |
| acl_task | 10 | 4 | 100.0% | 0.750 | 50.0% | 0 | 0 | - | 0.009/-0.065 |
| acl_restricted | 6 | 2 | 100.0% | 1.000 | 100.0% | 0 | 0 | - | 0.066/0.023 |
| out_of_corpus | 9 | 0 | - | - | - | 0 | 0 | - | 0.071/-0.027 |
| overall | 80 | 61 | 95.9% | 0.915 | 86.9% | 0 | 13 | 100.0% (8) | 0.061/-0.065 |

distractor 마진(정답 1위 14건): 평균 0.091 / 최소 0.042. forbidden이 2위인 8건의 평균 마진 0.083.

#### 틀린 케이스 (9건)

| id | 카테고리 | 기대 | 실제 상위 | 원인 추정 |
| --- | --- | --- | --- | --- |
| r004 | single_doc | notification-api | payment-retry-bank, payment-retry-card, **notification-api**(3위) | 결제 재시도 문서가 알림 관련 문구를 포함 |
| r005 | single_doc | order-schema | architecture-overview, api-style-guide, … (5위 밖) | 긴 문서들이 스키마 용어를 흡수. **유일한 recall 실패** |
| r011 | single_doc | onboarding | onboarding-backend, onboarding-frontend, … (5위 밖) | 온보딩 계열 신규 문서가 원 문서를 밀어냄 |
| r013 | cross_doc | notification-api + order-schema | notification-api만 (order-schema 없음) | r005와 같은 원인 |
| r020 | acl_task | settlement-design | settlement-monthly, **settlement-design**(2위) | 정산 계열 확장 |
| r035 | single_doc | payment-api | payment-retry-card, **payment-api**(2위) | 결제 계열 |
| r044 | single_doc | architecture-overview | search-api, **architecture-overview**(2위) | 검색 API 문서와 혼동 |
| r058 | distractor | deploy-dev | deploy-staging, **deploy-dev**(2위), 마진 -0.013 | 환경 단서가 약함. 유일한 distractor 실패 |
| r070 | acl_task | settlement-daily | meeting-2026-09-w1, **settlement-daily**(2위), 마진 -0.065 | 회의록이 정산 배치를 언급 |

#### 결론

- 1회차의 천장(전부 100%)이 사라졌다. 틀린 9건 중 7건은 정답이 2~3위에 있어 **순위 문제**이고(리랭커 대상), 2건(r005·r013의 order-schema)은 5위 안에 없어 **검색 자체 문제**다(청크 문맥 보강 대상).
- 같은 계열 문서가 늘자 acl_task hit@1이 50%(채점 4건 중 2건)까지 떨어졌다. 권한 필터는 정상(leak 0)이지만 계열 내 순위는 흔들린다.
- 마진이 음수인 케이스가 각 카테고리에 있다. RRF가 BM25 순위를 합치므로 벡터 유사도가 낮은 청크가 1위가 될 수 있다는 뜻이며, 리랭커는 이 구간을 보정할 수 있다.
- out_of_corpus 최대 유사도가 0.589 → 0.662로 올랐다(r030 휴가 질의가 restricted `hr-leave-policy`와 가까움). 그라운딩 1차 임계치(0.35)와 더 멀어져, 범위 밖 판정은 여전히 LLM 판정에 의존한다.
- ACL leak 0 유지.

#### 후속

- E3b 리랭커: 순위 문제 7건(r004, r011, r020, r035, r044, r058, r070)을 추적 대상으로 한다. 목표는 single_doc hit@1 83.3% → 90% 이상, acl_task hit@1 50% → 75% 이상.
- E3c 청크 문맥 보강: r005·r013(order-schema) recall 회복 여부와 긴 문서 청크 hit 유지.

### 3회차 (LLM 리랭커) - 실험 환경 기준

E3b 리랭커(`app/core/rag/reranker.py`, 기본 off)의 효과 실측이다. 후보 10건(RRF)을 `gemini-2.5-flash-lite`가 listwise로 재정렬해 상위 5건을 고른다. 순서만 바뀌고 `similarity_score`는 유지된다.

- 실행일: 2026-10-28
- 원본 파일: `data/eval/rag-v3-rerank.jsonl`
- 설정: `RERANK_ENABLED=true`, `RERANK_MODEL=gemini-2.5-flash-lite`, `RERANK_CANDIDATES=10`, `RERANK_TIMEOUT_SECONDS=2.0`, `RERANK_MAX_CHUNK_CHARS=600`, 임베딩·하이브리드 설정은 2회차와 동일
- 명령: `RUN_LIVE_LLM=1 RERANK_ENABLED=true PYTHONPATH=. python3 scripts/eval/rag_eval.py --live --rerank --rerank-interval 4 --rerank-retries 3 --top-k 5`
- 기준선: 같은 실행에서 가져온 후보 10건의 원래 순서 상위 5건. 2회차(top-k 5 검색)와 후보 수가 달라 수치가 조금 다르다(overall recall@5 95.9% → 97.5%, hit@1 동일 86.9%). 비교는 이 기준선과 한다.

#### 전체 요약

| 지표 | 목표(E3a 후속) | 기준선 (후보 10 상위 5) | 리랭크 | 달성 여부 |
| --- | --- | --- | --- | --- |
| single_doc hit@1 | >= 90% | 83.3% | 100.0% (30/30) | 달성 |
| acl_task hit@1 | >= 75% | 50.0% | 100.0% (4/4) | 달성 |
| distractor hit@1 | >= 기준선 | 93.3% | 100.0% (15/15) | 달성 |
| overall recall@5 / MRR / hit@1 | >= 기준선 | 97.5% / 0.919 / 86.9% | 98.4% / 0.989 / 98.4% (채점 61건) | 달성 |
| ACL leak | = 0 | 0 | 0 | 달성 |
| 청크 hit@5 (8건) | 유지 | 100.0% | 100.0% | 달성 |
| 리랭크 적용률 / fallback | fallback <= 5% | - | 78/80 적용, fallback 2건(2.5%, 모두 `parse`) | 달성 |
| 리랭크 지연 (applied 78건) | p50 <= 1,000ms | - | p50 889ms / p95 1,161ms / 최대 1,276ms | 달성 |

#### 카테고리별 (리랭크)

| 카테고리 | n | 채점 n | recall@5 | MRR | hit@1 | ACL leak | forbidden 노출 | 청크 hit@5 | 마진 평균/최소 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_doc | 30 | 30 | 100.0% | 1.000 | 100.0% | - | 0 | 100.0% (6) | 0.089/-0.004 |
| cross_doc | 10 | 10 | 90.0% | 0.933 | 90.0% | - | 0 | 100.0% (2) | 0.019/-0.061 |
| distractor | 15 | 15 | 100.0% | 1.000 | 100.0% | - | 13 | - | 0.107/0.013 |
| acl_task | 10 | 4 | 100.0% | 1.000 | 100.0% | 0 | 0 | - | 0.027/-0.016 |
| acl_restricted | 6 | 2 | 100.0% | 1.000 | 100.0% | 0 | 0 | - | 0.042/-0.028 |
| out_of_corpus | 9 | 0 | - | - | - | 0 | 0 | - | 0.003/-0.048 |
| overall | 80 | 61 | 98.4% | 0.989 | 98.4% | 0 | 13 | 100.0% (8) | 0.063/-0.061 |

차이(리랭크 - 기준선): overall recall@5 +0.8%p / MRR +0.070 / hit@1 +11.5%p. single_doc hit@1 +16.7%p, acl_task +50.0%p, distractor +6.7%p, cross_doc recall@5 -5.0%p·hit@1 -10.0%p, acl_restricted 변화 없음.

마진(1·2위 벡터 유사도 차이)은 순서만 바뀌므로 분포가 달라질 뿐 품질 지표가 아니다. 리랭크 후 1위가 BM25 전용 청크(벡터 유사도 0)가 될 수 있어 음수가 커질 수 있다.

#### 순위 추적 케이스 (2회차 후속 7건 + 검색 실패 2건)

| id | 카테고리 | 기준선 정답 순위 | 리랭크 후 | 비고 |
| --- | --- | --- | --- | --- |
| r004 | single_doc | 3위 | 1위 | |
| r011 | single_doc | 5위 밖(후보 7위) | 1위 | 후보 10건에 포함돼 리랭커가 끌어올림 |
| r020 | acl_task | 2위 | 1위 | |
| r035 | single_doc | 2위 | 1위 | |
| r044 | single_doc | 2위 | 1위 | |
| r058 | distractor | 2위 | 1위 | 마진 -0.013이던 환경 단서 약한 케이스 |
| r070 | acl_task | 2위 | 1위 | |
| r005 | single_doc | 4위 | 1위 | 2회차에서 검색 실패로 분류했으나 후보 10건 기준으로는 4위였음 |
| r013 | cross_doc | notification-api 1위, order-schema 없음 | notification-api 3위, order-schema 없음 | order-schema는 후보 10건에도 없음(E3c 대상) |

#### 나빠진 케이스

- r053(cross_doc, `architecture-overview` + `incident-postmortem-2026-08`): 기준선은 4위에 `architecture-overview`가 있었으나 리랭커가 `incident-postmortem` 청크 5개만 골라 recall@5가 0.5로 떨어졌다. listwise 리랭크는 문서 다양성을 고려하지 않으므로 같은 문서의 청크가 상위를 독점할 수 있다.
- r013(cross_doc): `notification-api`가 1위 → 3위. 2문서 질의에서 리랭커가 `architecture-overview`를 더 관련 있다고 봤다.
- fallback 2건(r078, r079, out_of_corpus): 응답 JSON 파싱 실패(`fallback:parse`). 코퍼스에 없는 질의에서 모델이 schema 밖 응답을 냈다. 원래 순서를 그대로 사용했고 채점 대상이 아니다.

#### 요청 제한 (중요)

- GMS 경유 `gemini-2.5-flash-lite`는 연속 호출 약 20회 이후 `429 RESOURCE_EXHAUSTED`를 반환했고, 간격 4초(15 RPM)로도 다른 사용자의 트래픽과 겹치면 429가 났다(공유 키 쿼터로 추정). 간격 없이 돌린 1차 실행은 80건 중 52건이 `fallback:http_429`였다.
- 최종 실측은 간격 4초 + 429 재시도 3회(15·30·60초 백오프)로 수행했고 재시도 3건이 모두 성공했다.
- 운영 의미: 리랭커는 fail-open이므로 429 시 품질이 기준선으로 돌아갈 뿐 응답은 실패하지 않는다. 다만 그라운딩 판정도 같은 모델을 쓰므로 동시 요청이 많으면 둘 다 fallback될 수 있다. 켜기 전에 GMS의 flash-lite 분당 한도를 확인하고, `chat_metrics.rerank_stage`의 `fallback:http_429` 비율을 모니터링해야 한다.

#### 결론

- 순위 문제 7건이 전부 1위로 회복됐고 single_doc·acl_task·distractor hit@1이 100%가 됐다. 지연 p50 889ms, fallback 2.5%로 판정 기준을 모두 충족한다.
- cross_doc은 r053 한 건 때문에 recall@5 95% → 90%로 떨어졌다. 2문서 질의에서 문서 다양성이 필요하다.
- 리랭크를 켜도 `similarity_score`·citations 구조·`token_usage`는 변하지 않는다. 리랭크 토큰(케이스당 입력 약 1,500·출력 약 15토큰)은 `llm_calls(purpose="rerank")`와 `chat_metrics.rerank_*`에만 기록된다.
- 기본값은 off 유지. 켜는 결정은 요청 제한 확인 후 한다.

#### 후속

- 문서 다양성: 리랭크 결과 상위 N에 같은 문서 청크가 몰릴 때 다른 문서의 최상위 청크를 1건 보장하는 규칙(또는 프롬프트에 "서로 다른 문서를 우선" 지시) 검토. r053 재측정.
- E3c 청크 문맥 보강: r013의 `order-schema`는 후보 10건에도 없어 리랭커로 해결되지 않는다.
- 그라운딩 판정과 리랭크를 병렬화하거나 그라운딩 입력을 리랭크 결과 상위 3건으로 고정하는 지연 최적화.

