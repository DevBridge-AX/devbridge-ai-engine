"""SemanticCache 유닛 테스트 (네임스페이스 격리, 임계치, TTL, LRU, 무효화)."""

import time

import pytest

from app.core.cache.semantic_cache import CacheEntry, SemanticCache, make_namespace


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _ns(**overrides):
    args = dict(
        workspace_id="ws", role="dev", accessible_task_ids=None,
        can_view_restricted=False, prompt_version="p1", main_model="m1",
    )
    args.update(overrides)
    return make_namespace(**args)


def _entry(answer="a", clock=None):
    return CacheEntry(
        embedding=[], query="q", answer=answer, citations=[], confidence=0.9,
        suggested_owner_id=None, created_at=clock() if clock else time.monotonic(),
    )


def _cache(clock=None, **kw):
    params = dict(threshold=0.95, ttl_seconds=60, max_entries=10)
    params.update(kw)
    if clock:
        params["clock"] = clock
    return SemanticCache(**params)


class TestLookup:

    def test_identical_vector_hits_with_similarity_one(self):
        cache = _cache()
        cache.store(_ns(), [1.0, 2.0, 3.0], _entry())
        found = cache.lookup(_ns(), [1.0, 2.0, 3.0])
        assert found is not None
        assert found[1] == pytest.approx(1.0)

    def test_hit_at_threshold_and_miss_below(self):
        cache = _cache(threshold=0.6 - 1e-9)
        cache.store(_ns(), [1.0, 0.0], _entry())
        assert cache.lookup(_ns(), [0.6, 0.8]) is not None  # cos = 0.6 (경계)
        assert cache.lookup(_ns(), [0.59, 0.8071]) is None

    def test_scaled_vector_is_equivalent(self):
        cache = _cache()
        cache.store(_ns(), [3.0, 4.0], _entry())
        assert cache.lookup(_ns(), [6.0, 8.0]) is not None

    def test_hits_counter_increments(self):
        cache = _cache()
        cache.store(_ns(), [1.0, 0.0], _entry())
        cache.lookup(_ns(), [1.0, 0.0])
        entry, _ = cache.lookup(_ns(), [1.0, 0.0])
        assert entry.hits == 2

    def test_best_match_is_returned(self):
        cache = _cache(threshold=0.5)
        cache.store(_ns(), [1.0, 0.0], _entry("far"))
        cache.store(_ns(), [0.8, 0.6], _entry("near"))
        entry, _ = cache.lookup(_ns(), [0.8, 0.6])
        assert entry.answer == "near"


class TestNamespaceIsolation:

    @pytest.mark.parametrize(
        "override",
        [
            {"workspace_id": "ws2"},
            {"role": "qa"},
            {"accessible_task_ids": []},
            {"accessible_task_ids": ["t1"]},
            {"can_view_restricted": True},
            {"prompt_version": "p2"},
            {"main_model": "m2"},
        ],
    )
    def test_different_namespace_misses(self, override):
        cache = _cache()
        cache.store(_ns(), [1.0, 0.0], _entry())
        assert cache.lookup(_ns(**override), [1.0, 0.0]) is None

    def test_none_vs_empty_vs_set_are_distinct(self):
        assert _ns(accessible_task_ids=None) != _ns(accessible_task_ids=[])
        assert _ns(accessible_task_ids=[]) != _ns(accessible_task_ids=["t1"])

    def test_task_id_order_is_irrelevant(self):
        assert _ns(accessible_task_ids=["b", "a"]) == _ns(accessible_task_ids=["a", "b"])


class TestLifecycle:

    def test_ttl_expiry(self):
        clock = FakeClock()
        cache = _cache(clock=clock, ttl_seconds=60)
        cache.store(_ns(), [1.0, 0.0], _entry(clock=clock))
        clock.t += 30
        assert cache.lookup(_ns(), [1.0, 0.0]) is not None
        clock.t += 31
        assert cache.lookup(_ns(), [1.0, 0.0]) is None
        assert cache.size() == 0

    def test_lru_cap_per_workspace_across_namespaces(self):
        cache = _cache(max_entries=2)
        ns_a, ns_b = _ns(role="dev"), _ns(role="qa")
        cache.store(ns_a, [1.0, 0.0], _entry("first"))
        cache.store(ns_b, [0.0, 1.0], _entry("second"))
        cache.lookup(ns_a, [1.0, 0.0])  # first를 최근 사용으로
        cache.store(ns_b, [0.7, 0.7], _entry("third"))
        assert cache.size("ws") == 2
        assert cache.lookup(ns_a, [1.0, 0.0]) is not None
        assert cache.lookup(ns_b, [0.0, 1.0]) is None  # second가 제거됨

    def test_cap_does_not_affect_other_workspace(self):
        cache = _cache(max_entries=1)
        cache.store(_ns(workspace_id="a"), [1.0, 0.0], _entry())
        cache.store(_ns(workspace_id="b"), [1.0, 0.0], _entry())
        assert cache.size("a") == 1 and cache.size("b") == 1

    def test_invalidate_workspace_only_clears_that_workspace(self):
        cache = _cache()
        cache.store(_ns(workspace_id="a"), [1.0, 0.0], _entry())
        cache.store(_ns(workspace_id="a", role="qa"), [1.0, 0.0], _entry())
        cache.store(_ns(workspace_id="b"), [1.0, 0.0], _entry())
        cache.invalidate_workspace("a")
        assert cache.size("a") == 0
        assert cache.size("b") == 1
        assert cache.size() == 1

    def test_remove(self):
        cache = _cache()
        entry = _entry()
        cache.store(_ns(), [1.0, 0.0], entry)
        cache.remove(_ns(), entry)
        assert cache.lookup(_ns(), [1.0, 0.0]) is None
        assert cache.size() == 0

    def test_generation_increments_on_invalidate_per_workspace(self):
        cache = _cache()
        assert cache.generation("a") == 0
        cache.invalidate_workspace("a")
        cache.invalidate_workspace("a")
        assert cache.generation("a") == 2
        assert cache.generation("b") == 0

    def test_clear(self):
        cache = _cache()
        cache.store(_ns(), [1.0, 0.0], _entry())
        cache.clear()
        assert cache.size() == 0
