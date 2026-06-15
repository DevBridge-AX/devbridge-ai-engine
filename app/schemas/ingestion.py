"""
인덱싱(문서/Git) 요청/응답 Pydantic 스키마.

TODO:
- DocumentIngestionRequest, GitIngestionRequest 필드 정의
- GitIngestionRequest에는 Spring 사용자조회 API 매핑에 필요한 커밋 작성자 식별 정보 포함
"""

from pydantic import BaseModel


class DocumentIngestionRequest(BaseModel):
    """문서 업로드/인덱싱 요청 스키마. TODO: 필드 정의."""


class GitIngestionRequest(BaseModel):
    """Git push 인덱싱 요청 스키마. TODO: 필드 정의."""
