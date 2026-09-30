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
