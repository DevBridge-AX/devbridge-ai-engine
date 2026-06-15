"""
문서/Git push 인덱싱 엔드포인트.

업로드된 문서 또는 Git push 이벤트를 받아 pipelines.document_ingestion /
pipelines.git_ingestion으로 위임하여 청킹 -> 임베딩 -> document_chunks 저장을 수행합니다.

TODO:
- POST /ingestion/documents (문서 업로드/인덱싱)
- POST /ingestion/git (Git push 변경분 인덱싱)
- GIT_COMMITS.author_id는 인덱싱 시점에 Spring 사용자조회 API로 USERS.id를 매핑해 저장
  (채팅 시점에는 추가 조회 없이 그대로 반환)
"""

from fastapi import APIRouter

router = APIRouter()

# TODO: 문서/Git 인덱싱 엔드포인트 구현
# @router.post("/documents")
# async def ingest_document(request: DocumentIngestionRequest):
#     ...
#
# @router.post("/git")
# async def ingest_git_push(request: GitIngestionRequest):
#     ...
