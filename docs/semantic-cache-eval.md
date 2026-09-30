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

### 2) 라이브 before/after 측정

후속 커밋에서 추가 예정입니다. (시맨틱 캐시 구현 위에서 캐시 off/on 비교 — hit 시 LLM 호출
절감과 지연 변화, 오답 여부 확인.)

## 결과

### 1회차 (임베딩 유사도 스윕)

미실행. 수집 후 아래 표를 채웁니다.

| 항목 | 값 |
| --- | --- |
| 실행 일시 | 미실행 |
| 임베딩 모델 | 미실행 |
| same_intent 유사도 (min / median / max) | 미실행 |
| different_intent 유사도 (min / median / max) | 미실행 |
| 권고 임계치 (false hit 0%) | 미실행 |
| 권고 임계치 (false hit <= 2%) | 미실행 |

| threshold | hit_rate | false_hit_rate | precision | f1 |
| --- | --- | --- | --- | --- |
| 미실행 | - | - | - | - |

### 라이브 before/after

미실행 (후속 커밋).

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
