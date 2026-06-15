"""
업로드 문서 분석/청킹/인덱싱 파이프라인.

업로드된 문서를 KNOWLEDGE_DOCUMENTS에 등록하고, core.rag.chunker로 청킹한 뒤
core.embeddings.embedder로 임베딩을 생성하여 document_chunks에 저장
(원문=content, 임베딩=embedding, embedding_model/embedding_model_version 포함)하고
db.vector_store에 색인합니다.

TODO:
- ingest_document(payload) 구현
- KNOWLEDGE_DOCUMENTS row 생성 -> chunker.chunk_document -> embedder.embed_texts
  -> document_chunks 저장 + vector_store.add
"""


async def ingest_document(payload: dict) -> None:
    """업로드 문서를 분석/청킹/인덱싱. TODO: 구현."""
    raise NotImplementedError
