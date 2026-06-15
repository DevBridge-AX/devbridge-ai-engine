"""
Git push 변경분 수집 파이프라인.

Git push 이벤트로 전달된 변경 커밋/파일을 수집하여 GIT_COMMITS 테이블에 저장하고,
변경된 코드 파일을 chunker -> embedder를 거쳐 document_chunks로 인덱싱합니다.

GIT_COMMITS.author_id 매핑:
- 인덱싱 시점에 Spring의 사용자조회 API를 호출하여 커밋 작성자(Git 계정)를 USERS.id로
  미리 매핑해 author_id에 저장합니다.
- 채팅 응답 시점에는 이 값을 그대로 반환하며 추가 조회를 하지 않습니다.

TODO:
- ingest_git_push(payload) 구현
- Spring 사용자조회 API 호출 클라이언트 (httpx, config.spring_backend_base_url +
  config.spring_user_lookup_path 사용)
- 변경 파일 -> core.rag.chunker.chunk_document -> core.embeddings.embedder.embed_texts
  -> db.vector_store.VectorStore.add
"""


async def ingest_git_push(payload: dict) -> None:
    """Git push 변경분을 인덱싱. TODO: 구현."""
    raise NotImplementedError
