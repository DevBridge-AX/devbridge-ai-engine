"""
Chroma/FAISS 벡터스토어 래퍼.

config.vector_store_path 경로의 임베디드/파일 기반 벡터스토어를 다룹니다.
별도 서버 없이 동작하며, 온디바이스 전환 시 이 경로(파일/디렉토리)를 그대로 이동하는
것만으로 동작해야 합니다.

TODO:
- VectorStore 인터페이스 메서드 확정 (add, search, delete)
- Chroma/FAISS 구현체 추가 (config.vector_store_provider로 선택)
- document_chunks.embedding과의 동기화 전략 (재임베딩 시 upsert 등)
"""

from app.config import get_settings

settings = get_settings()


class VectorStore:
    """벡터스토어 래퍼 인터페이스. TODO: Chroma/FAISS 구현."""

    def add(self, chunk_id: int, embedding: list[float], metadata: dict) -> None:
        """청크 임베딩을 벡터스토어에 추가/갱신. TODO: 구현."""
        raise NotImplementedError

    def search(self, query_embedding: list[float], top_k: int, workspace_id: int):
        """워크스페이스 범위 내에서 유사 청크 검색. TODO: 구현."""
        raise NotImplementedError

    def delete(self, chunk_id: int) -> None:
        """청크 임베딩을 벡터스토어에서 삭제. TODO: 구현."""
        raise NotImplementedError
