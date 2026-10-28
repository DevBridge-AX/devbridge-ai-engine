"""
scripts/eval/rag_docset.py + 데이터셋(scripts/eval/datasets/rag_docset, rag_cases.jsonl) 정합성 테스트.

임베딩/LLM API를 호출하지 않으며 seed_docset_async도 호출하지 않습니다.
"""

import json
import re
from collections import Counter

import pytest

from app.core.rag.chunker import chunk_document
from scripts.eval.rag_docset import (
    CaseAccess,
    DocEntry,
    access_filter_for,
    doc_allowed,
    load_cases,
    load_manifest,
)

EXPECTED_COUNTS = {
    "single_doc": 30,
    "cross_doc": 10,
    "distractor": 15,
    "acl_task": 10,
    "acl_restricted": 6,
    "out_of_corpus": 9,
}

# 6,000자 이상·청크 5개 이상이어야 하는 긴 문서(cross-chunk 질의용)
LONG_DOC_MIN_CHARS = {
    "architecture-overview": 8000,
    "incident-postmortem-2026-08": 6000,
    "release-notes-2026": 6000,
    "api-style-guide": 6000,
}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture(scope="module")
def by_key(manifest):
    return {e.doc_key: e for e in manifest}


def test_manifest_files_exist_and_keys_unique(manifest):
    assert len(manifest) == 40
    assert len({e.doc_key for e in manifest}) == len(manifest)
    assert len({e.file for e in manifest}) == len(manifest)
    for entry in manifest:
        assert entry.path.is_file(), entry.file
        length = len(entry.path.read_text(encoding="utf-8"))
        if entry.doc_key in LONG_DOC_MIN_CHARS:
            assert length >= LONG_DOC_MIN_CHARS[entry.doc_key], entry.file
        else:
            assert 150 <= length <= 700, entry.file


def test_long_docs_have_enough_chunks(manifest):
    for entry in manifest:
        if entry.doc_key in LONG_DOC_MIN_CHARS:
            text = entry.path.read_text(encoding="utf-8")
            assert len(chunk_document(text, "md")) >= 5, entry.doc_key


def test_docs_have_no_pii(manifest):
    pii = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}|01[0-9]-\d{3,4}-\d{4}", re.IGNORECASE)
    for entry in manifest:
        assert not pii.search(entry.path.read_text(encoding="utf-8")), entry.doc_key


def test_manifest_access_design(manifest):
    public = [e for e in manifest if e.task_id is None and e.sensitivity_level == "normal"]
    billing = [e for e in manifest if e.task_id == "task-billing"]
    onboarding = [e for e in manifest if e.task_id == "task-onboarding"]
    restricted = [e for e in manifest if e.sensitivity_level == "restricted"]
    assert (len(public), len(billing), len(onboarding), len(restricted)) == (29, 7, 1, 3)
    assert all(e.task_id is None for e in restricted)


def test_cases_load_and_counts(cases):
    assert len({c.id for c in cases}) == len(cases)
    assert Counter(c.category for c in cases) == EXPECTED_COUNTS
    assert [c.id for c in cases] == [f"r{i:03d}" for i in range(1, 81)]


def test_case_keys_exist_in_manifest(cases, by_key):
    for case in cases:
        for key in case.expected_doc_keys + case.forbidden_doc_keys:
            assert key in by_key, f"{case.id}: {key}"


def test_case_acl_consistency(cases, by_key):
    for case in cases:
        for key in case.expected_doc_keys:
            assert doc_allowed(by_key[key], case.access), f"{case.id}: expected {key}는 접근 불가"
        if case.category.startswith("acl_") and not case.expected_doc_keys:
            assert case.forbidden_doc_keys, case.id
        if case.category.startswith("acl_"):
            for key in case.forbidden_doc_keys:
                assert not doc_allowed(by_key[key], case.access), f"{case.id}: forbidden {key}가 허용됨"
        if case.category == "distractor":
            assert len(case.expected_doc_keys) == 1 and len(case.forbidden_doc_keys) >= 1
            assert not set(case.expected_doc_keys) & set(case.forbidden_doc_keys), case.id
        if case.category == "out_of_corpus":
            assert not case.expected_doc_keys and not case.forbidden_doc_keys


def test_cross_doc_has_two_expected(cases):
    for case in cases:
        if case.category == "cross_doc":
            assert len(case.expected_doc_keys) == 2


def test_expected_substrings_live_only_in_expected_docs(cases, by_key):
    texts = {k: e.path.read_text(encoding="utf-8") for k, e in by_key.items()}
    chunk_texts = {k: [c.content for c in chunk_document(t, "md")] for k, t in texts.items()}
    with_subs = [c for c in cases if c.expected_substrings]
    assert len(with_subs) == 8
    for case in with_subs:
        assert all(key in LONG_DOC_MIN_CHARS for key in case.expected_doc_keys), case.id
        for sub in case.expected_substrings:
            owners = [k for k, chunks in chunk_texts.items() if any(sub in c for c in chunks)]
            assert owners and set(owners) <= set(case.expected_doc_keys), f"{case.id}: {sub!r} -> {owners}"


def test_load_cases_defaults_expected_substrings(cases):
    assert next(c for c in cases if c.id == "r001").expected_substrings == []


def _entry(task_id=None, sensitivity="normal"):
    return DocEntry("k", "f.md", "t", "markdown", task_id, sensitivity, "topic", "")


def test_doc_allowed_predicate():
    full = CaseAccess(None, True)
    assert doc_allowed(_entry("task-a"), full)
    assert doc_allowed(_entry(None, "restricted"), full)
    assert not doc_allowed(_entry(None, "restricted"), CaseAccess(None, False))
    assert doc_allowed(_entry(None), CaseAccess([], False))
    assert not doc_allowed(_entry("task-a"), CaseAccess([], False))
    assert not doc_allowed(_entry("task-a"), CaseAccess(["task-b"], True))
    assert doc_allowed(_entry("task-a"), CaseAccess(["task-a"], False))
    assert doc_allowed(_entry("task-a"), CaseAccess(None, False))


def test_access_filter_for(cases):
    case = next(c for c in cases if c.access.accessible_task_ids == ["task-billing"])
    flt = access_filter_for(case)
    assert flt.accessible_task_ids == ["task-billing"]
    assert flt.can_view_restricted is False


def test_each_doc_produces_chunks(manifest):
    for entry in manifest:
        text = entry.path.read_text(encoding="utf-8", errors="replace")
        assert len(chunk_document(text, "md")) >= 1, entry.doc_key


def test_load_cases_rejects_bad_input(tmp_path):
    bad_cat = tmp_path / "a.jsonl"
    bad_cat.write_text(
        json.dumps(
            {
                "id": "x", "category": "nope", "question": "q", "expected_doc_keys": [],
                "forbidden_doc_keys": [], "access": {"accessible_task_ids": None, "can_view_restricted": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_cases(bad_cat)

    missing = tmp_path / "b.jsonl"
    missing.write_text(json.dumps({"id": "x", "category": "single_doc", "question": "q"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(missing)
