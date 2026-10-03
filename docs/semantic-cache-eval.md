# 시맨틱 캐시 효과 측정 (A9)

## 목적

시맨틱 캐시는 "같은 의도"의 질문이 다시 들어왔을 때 이전 답변을 재사용해 LLM 호출
비용과 지연을 줄이는 기능입니다. 핵심 위험은 **표면은 비슷하지만 답이 다른 질문을 같은
질문으로 오인(false hit)해 틀린 답을 돌려주는 것**입니다. 이 평가는 다음을 측정해
캐시 유사도 임계치를 정할 근거 데이터를 만듭니다.

- `hit_rate`: 같은 의도 질문 쌍 중 캐시가 hit하는 비율 (절감 효과)
- `false_hit_rate`: 다른 의도 질문 쌍 중 캐시가 hit하는 비율 (오답 위험)

이 문서/스크립트는 캐시의 기본 임계치를 변경하지 않습니다 — 값 변경은 이 결과를 근거로 한
별도 결정입니다.

## 데이터셋 구성

`scripts/eval/datasets/cache_pairs.jsonl` — 한국어 40쌍, 한 줄에 한 쌍.

```
{"id": "p001", "kind": "same_intent" | "different_intent", "q1": "...", "q2": "...", "note": "..."}
```

| kind | 쌍 수 | 정답 | 의미 |
| --- | --- | --- | --- |
| `same_intent` | 20 | hit | 같은 질문을 다른 표현(동의어, 존댓말/반말, 어순, 축약)으로 물음 |
| `different_intent` | 20 | miss | 표현은 비슷하지만 코퍼스상 답이 다름 (hit하면 오답) |

모든 질문은 `tests/live/fixtures/corpus/*.md` 4개 문서(인증/JWT 정책, 배포 절차, 장애 대응
runbook, 결제 API)의 사실로 답할 수 있으며, 문서별로 고르게 출제했습니다. `note`에 각 쌍이
같은/다른 의도인 이유를 적었습니다.

### 하드 네거티브 종류 (`different_intent`)

- 같은 대상, 다른 속성: 예) JWT 서명 알고리즘 vs 서명 키 로테이션 주기
- 같은 속성, 다른 대상: 예) Access Token vs Refresh Token 만료 시간, 배포 명령어 vs 롤백 명령어
- 부정 대비: 예) JWT 클레임에 포함되는 항목 vs 포함하지 않는 정보
- 수치 대 절차: 예) 배포 직후 5xx 급증 시 대응 vs 5xx 허용 비율
- 같은 문형, 다른 단계/사례: 예) 15분 vs 30분 에스컬레이션 대상, 장애 유형별 대응

## 측정 방법

### 1) 오프라인 임계치 스윕

쌍마다 q1/q2를 임베딩(`app/core/embeddings/embedder.py::embed_texts`, 40쌍 = 80건을 한 번에
호출)해 코사인 유사도를 구하고 캐시 jsonl로 저장합니다. 이후 스윕은 API 재호출 없이
오프라인으로 반복할 수 있습니다. 임베딩 비용만 발생하며 LLM 호출은 없습니다.

```bash
# 수집 (실 임베딩 API 호출) — RUN_LIVE_LLM=1과 비어있지 않은 GMS_API_KEY 필요
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --limit 5   # 스모크용

# 스윕 (API 호출 없음) — 임계치 0.85~0.99, 0.01 간격
python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-{YYYYMMDD-HHMMSS}.jsonl
```

`--task-type`(`--collect` 전용)으로 임베딩 요청에 Gemini `taskType`을 붙여 수집할 수 있습니다
(L3 실험, 기본은 미전송). 레코드에 `embedding_task_type`이 기록되고 리포트 실험 조건에
`임베딩 모델: ... (taskType: ...)`로 표시됩니다. 구버전 캐시(필드 없음)도 그대로 읽힙니다.

```bash
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --task-type SEMANTIC_SIMILARITY \
  --dataset scripts/eval/datasets/cache_pairs_holdout.jsonl
```

운영 설정 `EMBEDDING_QUERY_TASK_TYPE`은 이 평가로 유사도 분포와 임계치를 재확인한 뒤에만 바꿉니다
(문서는 taskType 없이 인덱싱되어 있어 질의 측만 바꾸면 검색 유사도 분포가 달라질 수 있음).

결과는 `data/eval/cache-{YYYYMMDD-HHMMSS}.jsonl`(`data/`는 `.gitignore` 대상)에 저장되며,
스윕 출력에는 유사도 분포, 임계치별 지표, 권고 임계치, 경계 사례가 포함됩니다.

권고 임계치는 두 가지로 계산합니다.

- 엄격: `false_hit_rate == 0` 을 만족하는 임계치 중 `hit_rate`가 가장 높은 값(동률이면 최저 임계치)
- 완화: `false_hit_rate <= 2%` 기준으로 같은 방식

### 2) 라이브 측정 (캐시 hit 경로)

바꿔 말한 질문의 적중률은 오프라인 스윕이 더 신뢰할 수 있으므로, 라이브는 두 가지만 봅니다.
(a) 동일 질문 반복 시 hit 경로의 지연/토큰 절감, (b) 기본 임계치(0.95)에서 오프라인상 유일하게
적중하는 paraphrase(p001) 1건의 적중 여부. 질문 8개(p001, p006, p008, p010, p011, p016, p017, p019의
q1)를 miss(기준선) → hit(동일 질문 재요청) 순으로 요청하고 p001 q2를 1회 추가 요청합니다.
LLM 호출은 약 8회(질문당 grounding + main)입니다.

```bash
RUN_LIVE_LLM=1 uv run --extra dev python -m pytest -m live -q tests/live/test_semantic_cache_live.py
```

결과는 터미널 출력과 `data/live_runs/*.json`의 `persona_answers_preview.semantic_cache`에 남습니다.
paraphrase 적중 여부는 단언하지 않고 기록만 합니다(miss 시 `cache_similarity`는 metrics에 기록되지 않아 null).

#### 라이브 측정 2회차(재검증 on) 방법

재검증 경로(후보 구간 0.80~0.95 paraphrase -> LLM 재검증 YES -> hit)의 지연/토큰은
`test_semantic_cache_verify_path`가 측정합니다. 전용 fixture가 `SEMANTIC_CACHE_ENABLED=true`,
`SEMANTIC_CACHE_VERIFY_ENABLED=true`, `SEMANTIC_CACHE_CANDIDATE_THRESHOLD=0.80`(프롬프트 v2, 임계치 0.95)을
설정하고 시작/종료 시 캐시를 비웁니다. 오프라인 2회차에서 후보 구간 YES였던 p004, p007, p008, p015~p018과
NO 사례 p011의 q1을 miss로 시드한 뒤 q2를 요청합니다. 호출 예산은 main/grounding 각 8회 + q2 miss 건수,
재검증은 q2 8회(+ q1끼리 후보 구간에 든 경우)입니다. q1 단계에서 hit 된 쌍은 `phase1_cache_hits`로 기록하고 비교에서 제외합니다.

```bash
RUN_LIVE_LLM=1 uv run --extra dev python -m pytest -m live -q tests/live/test_semantic_cache_live.py -k verify
```

결과는 `persona_answers_preview.semantic_cache_verify`에 남으며 쌍별 유사도/재검증 결과, hit 경로
total_ms p50/p95, verify_ms, 요청당 평균 LLM 토큰, 재검증 호출당 평균 토큰(llm_calls의
`purpose="cache_verify"`, token_usage에는 포함되지 않음), `meets_l6`(hit 경로 total_ms p50 <= 2000)을 포함합니다.
단언은 불변식만 다루며 hit/miss 결과는 실패 조건이 아닙니다.

## 결과

아래 모든 수치는 실험 환경(코퍼스 4문서, 40쌍) 기준입니다.

### 오프라인 스윕 1회차

- 데이터: `data/eval/cache-20261001-105951.jsonl`, 임베딩 모델 `gemini-embedding-2`

유사도 분포:

| kind | min | p25 | median | p75 | max | mean |
| --- | --- | --- | --- | --- | --- | --- |
| same_intent | 0.6464 | 0.7437 | 0.8352 | 0.8764 | 0.9556 | 0.8161 |
| different_intent | 0.6547 | 0.7378 | 0.8221 | 0.8876 | 0.9405 | 0.8107 |

임계치 스윕 (hit_rate = same_intent 적중률, false_hit_rate = different_intent 오적중률):

| threshold | hit_rate | false_hit_rate |
| --- | --- | --- |
| 0.85 | 50% | 35% |
| 0.86 | 40% | 35% |
| 0.87 | 30% | 30% |
| 0.88 | 25% | 25% |
| 0.89 | 20% | 25% |
| 0.90 | 10% | 25% |
| 0.91 | 5% | 25% |
| 0.92 | 5% | 20% |
| 0.93 | 5% | 5% |
| 0.94 | 5% | 5% |
| 0.95 | 5% | 0% |
| 0.96 | 0% | 0% |
| 0.97 | 0% | 0% |
| 0.98 | 0% | 0% |
| 0.99 | 0% | 0% |

권고 임계치: **0.95** (false hit 0% 및 <=2% 기준 모두 동일, hit_rate 5% = 1/20, false hit 0/20).

경계 사례 — 유사도가 가장 낮은 same_intent 5건 (놓치기 쉬운 쌍):

| id | similarity | q1 | q2 |
| --- | --- | --- | --- |
| p003 | 0.6464 | 개인키 로테이션 주기 알려줘 | 서명 키는 얼마마다 교체하나요? |
| p014 | 0.6912 | 포스트모템은 언제까지 써야 해? | 장애 사후 보고서 작성 기한이 어떻게 되나요 |
| p012 | 0.7070 | 롤백 명령어 알려줘 | 이전 버전으로 되돌리는 kubectl 명령이 뭐야? |
| p013 | 0.7260 | 장애 에스컬레이션 기준이 뭐야? | 장애가 안 풀리면 언제 누구한테 올려? |
| p006 | 0.7319 | 배포 절차 알려줘 | 운영 배포는 어떤 순서로 해? |

경계 사례 — 유사도가 가장 높은 different_intent 5건 (false hit 위험 쌍):

| id | similarity | q1 | q2 |
| --- | --- | --- | --- |
| p040 | 0.9405 | 서명 키는 며칠 주기로 로테이션해? | 로테이션 후 이전 서명 키는 며칠간 유지돼? |
| p024 | 0.9258 | JWT 클레임에 포함되는 항목은? | JWT 클레임에 포함하지 않는 정보는? |
| p029 | 0.9210 | deploy.sh의 --tag에는 무엇을 지정해야 해? | deploy.sh의 --tag에 지정하면 안 되는 값은? |
| p023 | 0.9206 | 로그아웃 시 Refresh Token은 어떻게 처리돼? | 로그아웃 시 Access Token은 어떻게 처리돼? |
| p021 | 0.9103 | Access Token 만료 시간은? | Refresh Token 만료 시간은? |

0.95 이상인 same_intent 쌍은 p001(0.9556) "Access Token 만료 시간이 얼마야?" / "액세스 토큰은 몇 분 지나면 만료돼?" 하나뿐입니다.

해석:

- 두 분포가 거의 완전히 겹칩니다(median 0.835 vs 0.822, mean 0.816 vs 0.811). 같은 의도의 쌍과
  다른 의도의 쌍을 임베딩 유사도만으로는 분리할 수 없습니다. 이는 부정적 결과이며 그대로 기록합니다.
- 오적중 0%를 지키면 바꿔 말한 질문의 적중률은 5%에 그칩니다. 반대로 hit_rate를 높이려 임계치를
  낮추면 false hit가 함께 늘어납니다(예: 0.88에서 25%/25%, 0.85에서 50%/35%).
- 따라서 임베딩 유사도 단독 캐시는 "사실상 동일 문장 재질문" 수준의 적중만 안전하게 보장합니다.
  기본값 `semantic_cache_threshold=0.95`를 유지하는 것을 권고하며, 이 평가로 기본값을 변경하지 않습니다.

### 라이브 측정 1회차 (2026-10-01)

실험 환경: 코퍼스 4문서(tests/live/fixtures/corpus)를 tmp 워크스페이스에 시드, `MAIN_MAX_TOKENS=256`,
main `claude-sonnet-4-6`, grounding `gemini-2.5-flash-lite`, 임베딩 `gemini-embedding-2`, 임계치 0.95,
role=developer, 단일 턴. 질문 8개(p001·p006·p008·p010·p011·p016·p017·p019의 q1)를 1회씩(Phase 1,
전부 miss) → 동일 질문 8개 재요청(Phase 2, 전부 hit) → p001 q2 1회(Phase 3). 결과 파일
`data/live_runs/20261001-110709.json`.

| 항목 | 값 |
| --- | --- |
| 비교 대상 질문 수 | 8 / 8 (그라운딩 실패 제외 0건) |
| miss total_ms p50 / p95 | 6,782 / 7,127 |
| hit total_ms p50 / p95 | 604 / 653 |
| miss ttft_ms p50 / p95 | 2,742 / 3,267 |
| hit ttft_ms p50 / p95 | 604 / 653 |
| 요청당 LLM 토큰 (miss / hit) | 4,881 / 0 (main+grounding, prompt+completion 평균) |
| 토큰 절감률 (hit) | 100% |
| hit 지연 절감 배수 (total p50) | 11.2배 |
| p001 paraphrase 적중 | hit (유사도 0.9556) |
| 이번 실행 LLM 토큰 합계 | grounding 15,888 / 140, main 21,056 / 1,966 (prompt / completion) |

- hit 경로의 지연(약 0.6s)은 질문 임베딩 1회(p50 약 0.6s)가 대부분이며, 캐시 조회 자체는 1ms 미만이다.
- hit 응답은 Phase 1 답변·citations와 동일했고 `token_usage.main`은 0/0, `grounding`은 null로 보고됐다(계약 불변).
- 이 수치는 "hit가 발생했을 때"의 효과다. 실제 절감 총량은 적중률에 비례하며, 바꿔 말한 질문의 적중률은 위
  오프라인 스윕대로 임계치 0.95에서 5%(1/20)에 그친다. 동일 질문 반복(FAQ·재접속 재질문)에서만 효과를
  기대할 수 있다.

## 2회차 (재검증)

1회차에서 임베딩 유사도 분포가 압축되어(같은 의도 중앙값 0.835 / 다른 의도 0.822) 임계치만으로는
hit와 오적중을 가르기 어렵다는 결론이 났다. 2회차는 후보 구간(candidate 하한 ≤ 유사도 < 0.95)의
최선 후보 1건을 경량 LLM(REWRITE_MODEL)으로 재검증하는 방식(`SEMANTIC_CACHE_VERIFY_ENABLED`)의
효과를 측정한다. 검증기 프롬프트에는 이 데이터셋의 문장을 예시로 넣지 않았다(평가 누수 방지).

### 방법

- `--collect --verify`: 임베딩 수집 후 **모든 쌍**에 `verify_same_question(q1, q2)`를 순차 호출해
  레코드에 `verify_outcome/verify_same/verify_latency_ms/verify_prompt_tokens/
  verify_completion_tokens/verify_model/verify_prompt_version`을 저장한다(쌍당 LLM 호출 1회).
  모든 쌍을 검증해 두므로 후보 임계치는 오프라인에서 스윕한다.
- 오프라인 판정(파이프라인과 동일): 유사도 ≥ direct(0.95) → 즉시 hit, candidate ≤ 유사도 < direct →
  검증 YES일 때만 hit, 그 미만 → miss. candidate를 [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]로 스윕한다.
- 채택 기준: false_hit_rate ≤ 2% **그리고** hit_rate ≥ 50%.
- 지표: hit_rate, false_hit_rate, 후보별 검증 호출 수, 검증 지연 p50/p95, 호출당 평균 토큰,
  검증기 단독 정확도와 오판 쌍 id. hit 경로 지연은 1회차 hit 경로 p50(약 604ms) + 검증 p50으로 추정한다.

### 명령

```bash
# 수집 (실 임베딩 + 실 LLM 40회 호출, 비용 발생)
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --verify

# 오프라인 리포트 (API 호출 없음)
python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-<timestamp>.jsonl --direct-threshold 0.95
```

direct 임계치 스윕과 "항상 검증" 모드(검증 없는 즉시 hit 지름길 제거)는 같은 캐시로 오프라인 계산한다
(`--collect --verify`가 전 쌍을 검증하므로 추가 API 호출 없음, verify 필드가 없는 캐시는 오류 종료).

```bash
# direct 0.95~0.98 + 항상 검증을 candidate 0.80/0.85로 비교 (L2: false_hit<=2% AND hit>=40%)
python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-<timestamp>.jsonl --direct-sweep
python3 scripts/eval/cache_eval.py --from-cache <file> --direct-sweep --direct-thresholds 0.95,0.97
```

### 결과 (2026-10-02, `data/eval/cache-20261002-004305.jsonl`)

- 검증 모델 / 프롬프트 버전: `claude-haiku-4-5-20251001` / `v1`, 임베딩 `gemini-embedding-2`
- 즉시 hit 임계치(direct) 0.95

| candidate | hit_rate | false_hit_rate | hits(same/diff) | 검증 호출(same/diff) |
| --- | --- | --- | --- | --- |
| 0.80 | **40.0%** | 0.0% | 8/0 | 24 (11/13) |
| 0.82 | 35.0% | 0.0% | 7/0 | 19 (9/10) |
| 0.84 | 35.0% | 0.0% | 7/0 | 17 (9/8) |
| 0.85 | 35.0% | 0.0% | 7/0 | 16 (9/7) |
| 0.86 | 30.0% | 0.0% | 6/0 | 14 (7/7) |
| 0.87 | 25.0% | 0.0% | 5/0 | 11 (5/6) |
| 0.88 | 20.0% | 0.0% | 4/0 | 9 (4/5) |
| 0.90 | 10.0% | 0.0% | 2/0 | 6 (1/5) |

- 비교: 1회차 임계치 단독 0.95 → hit 5% / 오적중 0%, 0.85 → 50% / 35%
- 검증기 단독(전 40쌍): 정확도 70.0%, **false YES 0건 / false NO 12건**
  (p002, p003, p005, p006, p009, p010, p011, p012, p013, p014, p019, p020), outcome yes 8 / no 32 (invalid·오류 0)
- 검증 지연 p50/p95: 694 / 877 ms → hit 경로 p50 추정 약 1.3s (기준 2s 이하 충족)
- 호출당 평균 토큰: prompt 469 / completion 4 (hit 1건이 절감하는 토큰 약 4,881 대비 소량)

### 결론

- **채택 기준 미달**: 오적중 0%는 유지했으나 hit_rate 최대 40%(candidate 0.80)로 50%에 못 미친다.
  기본값(`semantic_cache_enabled=false`, `semantic_cache_verify_enabled=false`, candidate 0.86)은 변경하지 않는다.
- 원인 1 — **검증기 보수성**: 오판 12건이 모두 false NO. "애매하면 NO" 규칙 때문에 워크스페이스 문맥이
  있어야 같다고 볼 수 있는 쌍(예: 특정 PG사명 ↔ "결제 게이트웨이", "몇 분 안에 대응" ↔ "SLA")을 거절한다.
- 원인 2 — **임베딩 상한**: same_intent 20쌍 중 8쌍은 유사도 0.80 미만(최저 0.646)이라 후보에도 들지 못한다.
  검증기가 완벽해도 hit_rate 상한은 candidate 0.85에서 50%, 0.80에서 60%다.
- 켠다면 candidate 0.80 권고(이 데이터셋에서 오적중 0%로 hit 8배). 단, 다른 의도 20쌍 기준 오적중 2% 이하는
  사실상 0건 요구이며 표본이 작아 운영 안전을 보장하지 않는다.

### 후속 과제

- 검증 프롬프트 v2: "애매하면 NO" 완화, 표기·용어 차이 허용. 같은 40쌍으로 튜닝하면 과적합되므로
  별도 검증용 쌍을 추가해 분리 측정할 것.
- 후보 하한을 낮추는 대신 질문 임베딩에 `taskType=SEMANTIC_SIMILARITY`를 적용해 임베딩 상한 자체를 개선.

### 운영 참고

- 검증 호출 토큰은 `done.token_usage`에 포함되지 않는다(응답 계약 불변). `llm_calls`의
  `purpose="cache_verify"` 이벤트로만 확인 가능하며, 과금 집계 반영은 Spring과 협의할 항목이다.
- 재검증 타임아웃으로 취소된 호출은 `llm_calls`에 `error_type="CancelledError"`로 기록된다
  (이전에는 `error_type=null`·토큰 0의 성공 호출처럼 보였다). `chat_metrics`에는
  `cache_verify_ms`/`cache_verify_result`가 남는다.

## 3회차 (검증 프롬프트 v2, 홀드아웃)

2회차 결론: 검증기 v1은 false YES 0 / false NO 12로 지나치게 보수적이었다("판단이 애매하면 NO").
3회차는 이를 완화한 프롬프트 v2(`app/core/cache/verifier.py`의 `CACHE_VERIFY_SYSTEM_PROMPT_V2`)를
같은 조건에서 v1과 비교한다. 설정 `SEMANTIC_CACHE_VERIFY_PROMPT_VERSION`은 측정 전 기본 `v1`이었고, 아래 결과를 근거로 `v2`로 변경했다.

### 방법

- v1 vs v2: v2는 표기(영문/한글/약어), 어순, 높임말, 동의어, 일반 용어 ↔ 사내 고유명사, 간접 표현은 같은
  질문(YES)으로 보고, 값·대상/주체·조건·요구 정보 종류가 다르면 NO로 둔다. "애매하면 NO" 대신
  "두 질문의 정답이 동일한 한 문장일 가능성이 높으면 YES"를 기준으로 한다. v1 텍스트는 그대로 보존된다.
- 튜닝용 40쌍(`scripts/eval/datasets/cache_pairs.jsonl`) vs 홀드아웃 30쌍
  (`scripts/eval/datasets/cache_pairs_holdout.jsonl`, same 15 / different 15). 프롬프트 설계에 홀드아웃은
  사용하지 않았고, 두 데이터셋의 문장은 프롬프트에 예시로 넣지 않았다(누수 테스트로 고정).
  홀드아웃 구성: 코퍼스 4문서 기반 15쌍(40쌍에서 쓰지 않은 사실) + 신규 워크스페이스 주제 15쌍
  (코드리뷰, 온보딩, 알림, 로그 보존, 배치, 캐시 설정). same은 영문/한글 표기, 고유명사 ↔ 일반 용어,
  간접 ↔ 직접 표현, 의문문 ↔ 명령문, 반말/존댓말을 포함하고, different는 수치·기간 / 대상 / 환경 /
  정보 종류 / 조건 차이의 하드 네거티브다.
- `--verify-prompt-version v1,v2`: 쌍마다 버전별로 검증해 (쌍, 버전)당 레코드 1건을 저장한다. 임베딩은 쌍당
  1회만 계산한다. LLM 호출 수는 쌍 수 x 버전 수(40쌍 80회 + 홀드아웃 30쌍 60회 = 총 140회).
- `--from-cache`는 버전이 2개 이상이면 버전별 섹션과 비교표(version | verifier accuracy | false YES |
  false NO | hit_rate@0.80/0.85/0.86 | false_hit_rate)를 출력한다.

### 명령

```bash
# 수집 (실 임베딩 + 실 LLM 호출, 비용 발생) — 튜닝용 40쌍 / 홀드아웃 30쌍 각각 v1,v2 비교
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --verify --verify-prompt-version v1,v2
RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --verify --verify-prompt-version v1,v2 \
    --dataset scripts/eval/datasets/cache_pairs_holdout.jsonl

# 오프라인 리포트 (API 호출 없음)
python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-<tuning-timestamp>.jsonl --direct-threshold 0.95
python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-<holdout-timestamp>.jsonl --direct-threshold 0.95
```

### 채택 기준

- **홀드아웃 기준** hit_rate ≥ 50% 그리고 오적중(false_hit_rate) ≤ 2%. 튜닝용 40쌍 결과는 참고용이다.
- 홀드아웃 different 15쌍 기준 2% 이하는 사실상 오적중 0건을 뜻한다(표본이 작아 운영 안전 보장은 아님).
- 임베딩 상한(유사도 낮은 same 쌍은 후보에도 들지 못함)은 프롬프트로 개선되지 않으므로 hit_rate 상한을 함께 해석한다.

### 결과 (2026-10-02, `data/eval/cache-tuning-v1v2-20261002-111641.jsonl`, `data/eval/cache-holdout-v1v2-20261002-111744.jsonl`)

- 검증 모델 `claude-haiku-4-5-20251001`, 임베딩 `gemini-embedding-2`, 즉시 hit 임계치 0.95. 튜닝용 40쌍·홀드아웃 30쌍에 v1/v2 각 1회 호출(LLM 140회).

**검증기 단독 (유사도 무관, 전 쌍)**

| 데이터셋 | version | 정확도 | false YES | false NO | 지연 p50/p95 (ms) | prompt 토큰/호출 |
| --- | --- | --- | --- | --- | --- | --- |
| 튜닝 40쌍 | v1 | 67.5% | 0 | 13 | 691 / 924 | 469 |
| 튜닝 40쌍 | v2 | 72.5% | 0 | 11 | 695 / 1,027 | 724 |
| 홀드아웃 30쌍 | v1 | 70.0% | 0 | 9 | 714 / 773 | 471 |
| **홀드아웃 30쌍** | **v2** | **90.0%** | 0 | 3 | 710 / 858 | 726 |

**임계치 + 재검증 스윕 (direct 0.95)**

| 데이터셋 | version | hit@0.80 | hit@0.85 | hit@0.86 | false_hit |
| --- | --- | --- | --- | --- | --- |
| 튜닝 | v1 | 35.0% | 30.0% | 25.0% | 0.0% |
| 튜닝 | v2 | 35.0% | 30.0% | 25.0% | 0.0% |
| 홀드아웃 | v1 | 26.7% | 20.0% | 13.3% | 13.3% (2/15) |
| 홀드아웃 | v2 | 40.0% | 26.7% | 13.3% | 13.3% (2/15) |

- 홀드아웃 오적중 2건(h016 0.961, h018 0.957)은 **즉시 hit 임계치 0.95 구간**에서 검증 없이 hit된 것으로, 검증기의
  false YES는 두 버전·두 데이터셋 모두 0건이다. "30일 ↔ 90일", "02:00 ↔ 14:00"처럼 값만 다른 쌍은 임베딩 유사도가
  0.95를 넘는다.
- 홀드아웃 same_intent 유사도 중앙값 0.798 → 15쌍 중 8쌍이 후보 하한 0.80 미만이라 검증기가 완벽해도 hit 상한은 약 47%.
- v1 튜닝 정확도가 2회차(70.0%)와 다른 것은 모델 비결정성(2쌍 차이).

### 결론

- **채택 기준(홀드아웃 hit ≥ 50%, 오적중 ≤ 2%) 미달**. 재검증 기본값(`semantic_cache_verify_enabled=false`)은 유지한다.
- **v2 프롬프트는 유효**: 홀드아웃 false NO 9 → 3(정확도 70 → 90%), false YES 0 유지, hit@0.80 26.7 → 40.0%.
  튜닝셋 개선 폭(67.5 → 72.5%)이 작은 것은 남은 오판이 "맥락 지식 필요"(p003 로테이션↔교체 주기 등) 유형이기 때문.
  비용은 prompt 토큰 +54%(469 → 724), 지연은 동일.
- **즉시 hit 임계치 0.95는 값만 다른 쌍에 안전하지 않다**: 홀드아웃 오적중 2건 모두 이 구간. 재검증을 켠다면
  direct 임계치를 0.97 이상으로 올리거나(홀드아웃 different_intent 최대 0.961) 모든 후보를 검증하는 편이 안전하다.
- **기본값 변경(승인됨, 2026-10-02)**: `semantic_cache_verify_prompt_version="v2"`. 재검증 자체가 기본 off이므로 운영 동작은 바뀌지 않으며, v1은 비교·회귀 확인용으로 유지한다.

### 후속 과제

- 임베딩 상한 개선: 질문 임베딩 `taskType=SEMANTIC_SIMILARITY` 실험(10-03 계획 D1-b). 홀드아웃 same_intent 8/15가 0.80 미만.
- direct 임계치 상향(0.97) 또는 "항상 검증" 모드 추가 시 오적중 0%·hit@0.80 40%(v2) 조합이 가능한지 재스윕.

### direct 스윕 오프라인 재스윕 (2026-10-03, `data/eval/cache-20261002-004305.jsonl`)

- `--direct-sweep`으로 2회차 캐시(튜닝 40쌍, v1, haiku)를 API 호출 없이 재스윕. 3회차 원본 캐시(v1v2 튜닝·홀드아웃)는 로컬에서 분실되어 사용 불가.

| mode/direct | hit@0.80 | hit@0.85 | false_hit | verify 호출@0.80 |
| --- | --- | --- | --- | --- |
| direct 0.95 | 40.0% | 35.0% | 0.0% | 24 |
| direct 0.96~0.98 | 40.0% | 35.0% | 0.0% | 25 |
| 항상 검증 | 40.0% | 35.0% | 0.0% | 25 |

- 튜닝셋에는 유사도 0.95 이상 different_intent 쌍이 없어 direct 구간 오적중이 0이고, direct 상향·항상 검증 모두 결과가 같다(검증 호출만 +1).
- 오적중 원천(값만 다른 쌍, 0.957·0.961)은 홀드아웃에만 있으므로, L2 판단은 홀드아웃을 v2로 다시 수집(`--collect --verify`, LLM 30회)한 뒤 `--direct-sweep`으로 확인해야 한다.

## 후속 선택지

- **질문 임베딩에 Gemini `taskType=SEMANTIC_SIMILARITY` 적용**: 문장 유사도에 맞춘 임베딩이라 same/different 분리도가 개선될 수 있음. 현재 embedder는 taskType을 지정하지 않고 검색용 임베딩과 분리해야 하므로 질문당 임베딩 1회 추가 비용.
- **hit 후보에 경량 LLM "같은 질문인가" 검증 단계 추가**: 유사도로 후보를 좁힌 뒤 판정해 오적중을 크게 억제하고 낮은 임계치로 hit_rate를 확보할 수 있음. 후보 hit마다 경량 모델 호출 1회의 소폭 비용/지연.
- **수치·고유명사 토큰 불일치 시 hit 차단하는 어휘 가드**: 예) 30분 vs 14일, Access vs Refresh 같은 대상/수치 차이로 인한 오적중을 비용 없이 차단. 동의어 표기 차이(액세스 토큰 vs Access Token)는 정규화가 필요하고 hit_rate 상승 효과는 없음.

## 해석 시 주의

- **소형 코퍼스/표본**: 문서 4개, 쌍 40건(kind별 20건)입니다. 1건이 5%p에 해당하므로 `false_hit_rate`
  0%는 "이 데이터셋에서 관측되지 않음"일 뿐 운영 환경에서의 안전을 보장하지 않습니다. 임계치는
  운영 코퍼스와 실제 질문 로그로 재검증해야 합니다.
- **유사도 분포 압축**: 임베딩 모델에 따라 유사도가 좁은 대역(예: 0.8 이상)에 몰릴 수 있어 임계치
  0.01~0.02 차이로 hit/false hit가 크게 바뀔 수 있습니다. 스윕 간격(0.01) 내 민감도를 함께 보세요.
- **환경 의존성**: 결과는 측정 시점의 임베딩 모델/버전(`embedding_model`)과 데이터셋에 종속됩니다.
  모델을 바꾸면 임계치를 다시 측정해야 합니다.
- **질문 단독 유사도**: 이 스윕은 질문 문장 간 유사도만 봅니다. 워크스페이스, 역할(role),
  멀티턴 맥락이 다르면 같은 질문이라도 답이 달라질 수 있으며, 이는 이 데이터셋의 범위 밖입니다.
- 이 결과는 임계치 결정의 근거 자료이며, 값의 적용은 별도 작업입니다.
