# RAG 품질 검증 문서 셋 (X2)

## 목적

가상 문서 10건과 질의 30건으로 (1) 검색 품질(recall@k, MRR, hit@1)과 (2) 접근 제어
(task / sensitivity 필터) 정확성을 반복 측정한다. 데이터셋과 지표 계산은 API 없이 동작하며,
실제 검색 실행(`--live`)만 임베딩 API를 호출한다. 운영 문서나 실제 개인정보는 포함하지 않는다.

## 데이터셋 구성

위치: `scripts/eval/datasets/rag_docset/` (`manifest.json` + 마크다운 10건)

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

## 케이스 카테고리

위치: `scripts/eval/datasets/rag_cases.jsonl` (30건)

| category | 건수 | 설명 |
| --- | --- | --- |
| single_doc | 12 | 문서 1건이 정답, 전체 권한 |
| cross_doc | 3 | 2개 문서가 모두 필요 |
| distractor | 4 | 한쪽 문서가 정답, 반대쪽은 forbidden |
| acl_task | 5 | task 문서 질문. 권한 있음 / 없음(빈 목록, 다른 task) 쌍 |
| acl_restricted | 3 | restricted 문서. 열람 가능 / 불가 |
| out_of_corpus | 3 | 코퍼스 밖 질문. top similarity만 기록 |

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

## 실행법

```bash
# live: 임베딩 API 호출 (RUN_LIVE_LLM=1 필수)
RUN_LIVE_LLM=1 python3 scripts/eval/rag_eval.py --live [--limit N] [--top-k 5]

# 저장된 결과 집계 (API 호출 없음)
python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-YYYYmmdd-HHMMSS.jsonl
```

- 비용: 임베딩만 발생한다. 시드 10문서(청크 단위 임베딩) + 질의 30건(질의당 1회). LLM 호출은 없다.
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
- 명령: `RUN_LIVE_LLM=1 python3 scripts/eval/rag_eval.py --live --top-k 5`
- 리포트: `python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-<timestamp>.jsonl`

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
