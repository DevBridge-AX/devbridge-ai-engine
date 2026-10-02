"""
RAG 품질 검증 문서 셋(X2) 로더/시딩 모듈.

- 문서 셋: scripts/eval/datasets/rag_docset/ (manifest.json + 10개 가상 마크다운 문서)
- 질의 케이스: scripts/eval/datasets/rag_cases.jsonl
- load_manifest / load_cases / doc_allowed / access_filter_for 는 순수 함수(API 호출 없음)이며
  tests/test_rag_docset.py가 오프라인으로 검증합니다.
- seed_docset_async 는 실제 임베딩 API를 호출하므로 RUN_LIVE_LLM=1 게이트 뒤에서만 호출하세요.
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.rag.retriever import AccessFilter

_DATASET_DIR = Path(__file__).resolve().parent / "datasets"
DOCSET_DIR = _DATASET_DIR / "rag_docset"
DEFAULT_MANIFEST = DOCSET_DIR / "manifest.json"
DEFAULT_CASES = _DATASET_DIR / "rag_cases.jsonl"

VALID_CATEGORIES = (
    "single_doc",
    "cross_doc",
    "distractor",
    "acl_task",
    "acl_restricted",
    "out_of_corpus",
)
VALID_SENSITIVITY = ("normal", "restricted")


@dataclass(frozen=True)
class DocEntry:
    doc_key: str
    file: str
    title: str
    document_type: str
    task_id: str | None
    sensitivity_level: str
    topic: str
    note: str

    @property
    def path(self) -> Path:
        return DOCSET_DIR / self.file


@dataclass(frozen=True)
class CaseAccess:
    accessible_task_ids: list[str] | None
    can_view_restricted: bool


@dataclass(frozen=True)
class RagCase:
    id: str
    category: str
    question: str
    expected_doc_keys: list[str]
    forbidden_doc_keys: list[str]
    access: CaseAccess
    note: str


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[DocEntry]:
    """manifest.json을 읽어 DocEntry 목록으로 반환합니다. 필드 누락/값 오류는 ValueError."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for i, item in enumerate(raw):
        try:
            entry = DocEntry(
                doc_key=item["doc_key"],
                file=item["file"],
                title=item["title"],
                document_type=item["document_type"],
                task_id=item["task_id"],
                sensitivity_level=item["sensitivity_level"],
                topic=item["topic"],
                note=item.get("note", ""),
            )
        except KeyError as exc:
            raise ValueError(f"{path}[{i}]: 필수 필드 {exc}가 없습니다.") from exc
        if entry.sensitivity_level not in VALID_SENSITIVITY:
            raise ValueError(f"{path}[{i}]: 알 수 없는 sensitivity_level {entry.sensitivity_level!r}")
        entries.append(entry)
    return entries


def load_cases(path: Path = DEFAULT_CASES) -> list[RagCase]:
    """rag_cases.jsonl을 읽어 RagCase 목록으로 반환합니다. 알 수 없는 category/필드 누락은 ValueError."""
    cases = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        try:
            access = item["access"]
            case = RagCase(
                id=item["id"],
                category=item["category"],
                question=item["question"],
                expected_doc_keys=list(item["expected_doc_keys"]),
                forbidden_doc_keys=list(item["forbidden_doc_keys"]),
                access=CaseAccess(
                    accessible_task_ids=access["accessible_task_ids"],
                    can_view_restricted=bool(access["can_view_restricted"]),
                ),
                note=item.get("note", ""),
            )
        except KeyError as exc:
            raise ValueError(f"{path}:{lineno}: 필수 필드 {exc}가 없습니다.") from exc
        if case.category not in VALID_CATEGORIES:
            raise ValueError(
                f"{path}:{lineno}: category는 {VALID_CATEGORIES} 중 하나여야 합니다 (got {case.category!r})."
            )
        if not case.question:
            raise ValueError(f"{path}:{lineno}: question이 비어 있습니다.")
        cases.append(case)
    return cases


def doc_allowed(entry: DocEntry, access: CaseAccess) -> bool:
    """retriever의 AccessFilter 의미와 동일한 허용 판정.

    - restricted 문서는 can_view_restricted일 때만 허용
    - task_id가 없으면 워크스페이스 공개로 허용
    - accessible_task_ids가 None이면 task 제한 없음, 리스트면 포함된 task만 허용
    """
    if entry.sensitivity_level == "restricted" and not access.can_view_restricted:
        return False
    if entry.task_id is None:
        return True
    if access.accessible_task_ids is None:
        return True
    return entry.task_id in access.accessible_task_ids


def access_filter_for(case: RagCase) -> AccessFilter:
    return AccessFilter(
        accessible_task_ids=case.access.accessible_task_ids,
        can_view_restricted=case.access.can_view_restricted,
    )


async def seed_docset_async(tmp_dir: Path) -> tuple[Session, str, dict[str, str]]:
    """tmp_dir 아래 sqlite DB를 만들고 manifest의 문서를 task/민감도 메타와 함께 인덱싱합니다.

    scripts/eval/seed.py::seed_workspace_async 와 같은 방식이며, 문서마다 task_id /
    sensitivity_level을 document_ingestion._run에 전달합니다.

    주의: 이 함수는 임베딩 API를 실제로 호출합니다(비용 발생). 반드시 RUN_LIVE_LLM=1 게이트
    뒤에서만 호출하고, 테스트에서는 호출하지 마세요.

    Returns:
        (db, workspace_id, doc_key -> knowledge_document_id). db는 호출자가 close()합니다.
    """
    from app.db.models import Base, KnowledgeDocument
    from app.pipelines.document_ingestion import _run

    workspace_id = f"live-{uuid.uuid4().hex[:8]}"
    engine = create_engine(f"sqlite:///{tmp_dir}/live.db")
    Base.metadata.create_all(engine)
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    doc_ids: dict[str, str] = {}
    for entry in load_manifest():
        doc_id = str(uuid.uuid4())
        db.add(
            KnowledgeDocument(
                id=doc_id,
                workspace_id=workspace_id,
                source_id="seed-docset",
                task_id=entry.task_id,
                title=entry.title,
                document_type=entry.document_type,
                analysis_status="PENDING",
            )
        )
        db.commit()
        await _run(
            db,
            workspace_id,
            doc_id,
            str(entry.path),
            "md",
            task_id=entry.task_id,
            sensitivity_level=entry.sensitivity_level,
        )
        doc_ids[entry.doc_key] = doc_id

    return db, workspace_id, doc_ids
