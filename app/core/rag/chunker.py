"""
문서 청킹.

업로드된 문서(원문)를 document_chunks.content(원문, source of truth)로 저장할
단위로 분할합니다. 청킹 결과는 core.embeddings.embedder를 통한 임베딩 생성의
입력입니다.

TODO:
- chunk_document(text, ...) -> list[str] 구현
- 문서 타입(코드/일반 문서/Git diff 등)별 청킹 전략 분리 검토
"""


def chunk_document(text: str) -> list[str]:
    """문서를 청크 단위로 분할. TODO: 구현."""
    raise NotImplementedError
