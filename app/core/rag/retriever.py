"""
document_chunks 벡터 검색 (Chroma/FAISS).

db.vector_store의 벡터스토어 래퍼를 사용하여 쿼리 임베딩과 유사한 document_chunks를
검색하고, 각 결과의 유사도 점수를 포함해 반환합니다. 이 유사도 점수는
core.rag.grounding의 1차 필터로 사용됩니다.

TODO:
- retrieve(query, workspace_id, top_k) -> list[RetrievedChunk] 구현
- workspace_id 기준 필터링 (워크스페이스 간 문서 혼입 방지)
"""


async def retrieve(query: str, workspace_id: int, top_k: int = 5):
    """쿼리에 대한 관련 document_chunks를 유사도 점수와 함께 검색. TODO: 구현."""
    raise NotImplementedError
