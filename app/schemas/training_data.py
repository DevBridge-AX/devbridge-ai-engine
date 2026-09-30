"""
LoRA 학습데이터 export API 스키마.

입력 레코드는 Spring이 (자신이 소유한 CHAT_MESSAGES / OWNER_CONFIRMATIONS 등에서
추출해) payload로 전달합니다. 이 레포는 해당 테이블을 조회하지 않으며, user_id 같은
개인 식별자는 스키마에 포함하지 않습니다.
"""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import PersonaRole

# workspace_id / dataset_version은 파일 경로 구성요소로 쓰이므로 경로 탈출 문자를 막는다.
SAFE_NAME_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"


class TrainingRecordIn(BaseModel):
    original_question: str
    target_role: PersonaRole
    final_answer: str
    source: Literal["chat", "owner_answer", "faq"]
    prompt_version: str
    is_faq: bool = False
    owner_verified: bool = False


class TrainingDataExportRequest(BaseModel):
    workspace_id: str = Field(pattern=SAFE_NAME_PATTERN)
    dataset_version: str = Field(pattern=SAFE_NAME_PATTERN)
    records: list[TrainingRecordIn] = Field(min_length=1, max_length=10000)


class TrainingDataExportResponse(BaseModel):
    workspace_id: str
    dataset_version: str
    path: str
    record_count: int
    scrubbed_field_count: int
