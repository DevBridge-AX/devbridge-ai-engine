"""
시맨틱 캐시: 의미가 같은 질문의 답변을 프로세스 메모리에 저장해 재생합니다.

- 조회 키는 네임스페이스(workspace_id, role, access_fingerprint, prompt_version,
  main_model)이며, 네임스페이스가 하나라도 다르면 절대 hit하지 않습니다.
  access_fingerprint는 (정렬된 accessible_task_ids 또는 None, can_view_restricted)이고,
  None(제한 없음)과 [](공용만)은 서로 다른 값으로 취급합니다.
- 네임스페이스 안에서 쿼리 임베딩의 코사인 유사도(순수 Python 선형 스캔)가
  threshold 이상인 최선의 엔트리를 반환합니다. 벡터는 저장·조회 시 L2 정규화합니다.
- TTL은 time.monotonic() 기준이며 clock을 주입해 테스트할 수 있습니다.
- 워크스페이스당 총 엔트리 수가 max_entries를 넘으면 가장 오래 쓰이지 않은(LRU)
  엔트리부터 제거합니다.

제약:
- 락을 쓰지 않습니다(단일 이벤트 루프 가정).
- 프로세스 메모리 캐시이므로 멀티 워커(uvicorn --workers N)에서는 워커마다 캐시가
  독립이며, 한 워커에서 일어난 인덱싱 무효화가 다른 워커의 캐시에는 전파되지 않습니다.
  멀티 워커 운영 시 TTL이 stale 허용 상한이 됩니다.
"""

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings

CacheNamespace = tuple  # (workspace_id, role, access_fingerprint, prompt_version, main_model)


@dataclass
class CacheEntry:
    embedding: list[float]  # L2 정규화된 쿼리 임베딩
    query: str  # rewritten_query (디버그 전용, 로그 기록 금지)
    answer: str
    citations: list[dict]  # Citation.model_dump()
    confidence: float
    suggested_owner_id: str | None
    created_at: float  # time.monotonic()
    hits: int = 0


def make_namespace(
    workspace_id: str,
    role: str,
    accessible_task_ids: list[str] | None,
    can_view_restricted: bool,
    prompt_version: str,
    main_model: str,
) -> CacheNamespace:
    """캐시 격리 단위를 만듭니다. accessible_task_ids의 None과 []는 구분됩니다."""
    fingerprint = (
        tuple(sorted(accessible_task_ids)) if accessible_task_ids is not None else None,
        can_view_restricted,
    )
    return (workspace_id, role, fingerprint, prompt_version, main_model)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SemanticCache:
    def __init__(
        self,
        *,
        threshold: float,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._store: dict[CacheNamespace, OrderedDict[int, CacheEntry]] = {}
        # 워크스페이스 단위 LRU 순서: (ns, key) -> None (앞쪽이 가장 오래된 것)
        self._lru: dict[str, OrderedDict[tuple[CacheNamespace, int], None]] = {}
        self._next_key = 0
        # 워크스페이스별 무효화 세대. invalidate_workspace마다 +1 되어, 조회 시점 이후
        # 무효화가 일어났는지(저장 시 stale 여부)를 호출자가 확인할 수 있게 한다.
        self._generations: dict[str, int] = {}

    def generation(self, workspace_id: str) -> int:
        """워크스페이스의 현재 무효화 세대(기본 0)를 반환합니다."""
        return self._generations.get(workspace_id, 0)

    def now(self) -> float:
        """캐시가 사용하는 현재 시각(CacheEntry.created_at과 같은 시계)."""
        return self._clock()

    def lookup(self, ns: CacheNamespace, embedding: list[float]) -> tuple[CacheEntry, float] | None:
        """네임스페이스 내 최선의 유사 엔트리와 유사도를 반환합니다. 없으면 None."""
        entries = self._store.get(ns)
        if not entries:
            return None

        query = _normalize(embedding)
        now = self._clock()
        best_key: int | None = None
        best_sim = -2.0
        for key in list(entries):
            entry = entries[key]
            if now - entry.created_at > self._ttl:
                self._drop(ns, key)
                continue
            sim = _dot(query, entry.embedding)
            if sim > best_sim:
                best_sim, best_key = sim, key

        if best_key is None or best_sim < self._threshold:
            return None

        entry = self._store[ns][best_key]
        entry.hits += 1
        self._store[ns].move_to_end(best_key)
        self._lru[ns[0]].move_to_end((ns, best_key))
        return entry, best_sim

    def store(self, ns: CacheNamespace, embedding: list[float], entry: CacheEntry) -> None:
        entry.embedding = _normalize(embedding)
        key = self._next_key
        self._next_key += 1
        self._store.setdefault(ns, OrderedDict())[key] = entry
        lru = self._lru.setdefault(ns[0], OrderedDict())
        lru[(ns, key)] = None
        while len(lru) > self._max_entries:
            old_ns, old_key = next(iter(lru))
            self._drop(old_ns, old_key)

    def remove(self, ns: CacheNamespace, entry: CacheEntry) -> None:
        """접근 재검증 실패 등으로 특정 엔트리를 제거합니다."""
        entries = self._store.get(ns)
        if not entries:
            return
        for key, value in list(entries.items()):
            if value is entry:
                self._drop(ns, key)

    def invalidate_workspace(self, workspace_id: str) -> None:
        for ns in [ns for ns in self._store if ns[0] == workspace_id]:
            del self._store[ns]
        self._lru.pop(workspace_id, None)
        self._generations[workspace_id] = self._generations.get(workspace_id, 0) + 1

    def clear(self) -> None:
        self._store.clear()
        self._lru.clear()

    def size(self, workspace_id: str | None = None) -> int:
        return sum(
            len(entries)
            for ns, entries in self._store.items()
            if workspace_id is None or ns[0] == workspace_id
        )

    def _drop(self, ns: CacheNamespace, key: int) -> None:
        entries = self._store.get(ns)
        if entries is not None:
            entries.pop(key, None)
            if not entries:
                del self._store[ns]
        lru = self._lru.get(ns[0])
        if lru is not None:
            lru.pop((ns, key), None)


@lru_cache
def get_semantic_cache() -> SemanticCache:
    """SemanticCache 싱글톤을 반환합니다."""
    settings = get_settings()
    return SemanticCache(
        threshold=settings.semantic_cache_threshold,
        ttl_seconds=settings.semantic_cache_ttl_seconds,
        max_entries=settings.semantic_cache_max_entries,
    )
