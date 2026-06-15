"""
임베딩 생성 및 모델 버전 관리.

document_chunks.embedding(derived)을 생성합니다. content(원문)는 source of truth로
불변이며, embedding_model/embedding_model_version이 바뀌면 원문은 유지한 채 임베딩만
재생성하는 구조를 지원해야 합니다.

TODO:
- embed_texts(texts: list[str]) -> list[list[float]] 구현 (config.embedding_model 사용)
- re_embed_chunks(chunk_ids, new_model, new_version) 류의 재생성 유틸 설계
"""


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록에 대한 임베딩 벡터 생성. TODO: 구현."""
    raise NotImplementedError
