"""
[Chroma 벡터스토어 래퍼]

config.vector_store_path 경로의 파일 기반 Chroma PersistentClient 사용
workspace_id별로 컬렉션을 분리하여 워크스페이스 간 문서 혼입 방지
컬렉션명: "workspace_{workspace_id}", 거리 함수: cosine.

온디바이스 전환 시 vector_store_path 디렉토리를 그대로 이동하면 동작
"""

from functools import lru_cache

import chromadb

from app.config import get_settings


class VectorStore:
    """Chroma 기반 벡터스토어. workspace_id별 컬렉션으로 격리됩니다."""

    def __init__(self, path: str) -> None:
        self._client = chromadb.PersistentClient(path=path)

    def _collection(self, workspace_id: str):
        return self._client.get_or_create_collection(
            name=f"workspace_{workspace_id.lower()}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        vector_id: str,
        chunk_id: int,
        embedding: list[float],
        metadata: dict,
        workspace_id: str,
    ) -> None:
        """청크 임베딩을 벡터스토어에 추가합니다. vector_id가 이미 존재하면 덮어씁니다."""
        self._collection(workspace_id).upsert(
            ids=[vector_id],
            embeddings=[embedding],
            metadatas=[{**metadata, "chunk_id": chunk_id}],
        )

    def search(
        self,
        query_embedding: list[float],
        workspace_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """워크스페이스 범위 내에서 유사 청크를 검색합니다.

        반환값: [{"chunk_id": int, "score": float}] (score는 cosine 유사도, 높을수록 유사)
        """
        col = self._collection(workspace_id)
        n = min(top_k, col.count())
        if n == 0:
            return []

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["distances", "metadatas"],
        )
        return [
            {"chunk_id": int(meta["chunk_id"]), "score": 1.0 - dist}
            for dist, meta in zip(results["distances"][0], results["metadatas"][0])
        ]

    def delete(self, vector_id: str, workspace_id: str) -> None:
        """청크 임베딩을 벡터스토어에서 삭제합니다."""
        self._collection(workspace_id).delete(ids=[vector_id])


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """VectorStore 싱글톤을 반환합니다. PersistentClient는 하나만 생성해야 합니다."""
    return VectorStore(path=get_settings().vector_store_path)
