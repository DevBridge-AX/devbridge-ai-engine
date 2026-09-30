"""
평가/라이브 검증용 tmp 워크스페이스 공통 시딩 로직.

tests/live/conftest.py(A3)와 향후 scripts/eval/grounding_eval.py(A4)가 이 모듈을
공유합니다. tmp sqlite 파일 DB에 KnowledgeDocument 행을 만들고,
app.pipelines.document_ingestion._run()으로 tests/live/fixtures/corpus/*.md를
실제로 청킹/임베딩/인덱싱합니다 — 이 과정에서 임베딩 API가 실제 호출되므로
반드시 RUN_LIVE_LLM=1 게이트(tests/live/conftest.py) 뒤에서만 호출해야 합니다.

workspace_id는 "live-{uuid8}" 형태로 매 호출마다 새로 발급하여, BM25 인메모리
캐시(app.core.rag.bm25_index)나 벡터스토어 컬렉션이 이전 실행과 충돌하지 않게 합니다.
"""

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, KnowledgeDocument
from app.pipelines.document_ingestion import _run

_CORPUS_DIR = Path(__file__).resolve().parents[2] / "tests" / "live" / "fixtures" / "corpus"


def seed_workspace(tmp_dir: Path) -> tuple[Session, str]:
    """동기 호출부(pytest fixture 등)용 래퍼. 실행 중인 이벤트 루프 안에서는
    seed_workspace_async()를 await하세요."""
    return asyncio.run(seed_workspace_async(tmp_dir))


async def seed_workspace_async(tmp_dir: Path) -> tuple[Session, str]:
    """tmp_dir 아래에 sqlite DB를 만들고 corpus 문서를 실 임베딩으로 인덱싱합니다.

    Args:
        tmp_dir: DB 파일(live.db)을 생성할 디렉터리. 호출 전에 존재해야 합니다.

    Returns:
        (db, workspace_id). db는 호출자가 계속 사용하며, 필요 시 호출자가 close()합니다.
    """
    workspace_id = f"live-{uuid.uuid4().hex[:8]}"

    engine = create_engine(f"sqlite:///{tmp_dir}/live.db")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_factory()

    for path in sorted(_CORPUS_DIR.glob("*.md")):
        doc_id = str(uuid.uuid4())
        db.add(
            KnowledgeDocument(
                id=doc_id,
                workspace_id=workspace_id,
                source_id="seed-corpus",
                title=path.stem,
                document_type="markdown",
                analysis_status="PENDING",
            )
        )
        db.commit()
        await _run(db, workspace_id, doc_id, str(path), "md")

    return db, workspace_id
