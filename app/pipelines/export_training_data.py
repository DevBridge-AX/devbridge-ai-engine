"""
workspace별 LoRA 학습데이터 export ETL.

워크스페이스 단위로 격리된 JSONL을 생성합니다. PII 스크러빙을 포함하며,
각 레코드는 {original_question, target_role, final_answer, source, prompt_version,
is_faq, owner_verified} 형태이고 dataset_version을 기록합니다.

TODO:
- export_training_data(workspace_id, dataset_version) -> JSONL 파일 경로 구현
- PII 스크러빙 로직 (이름/이메일/사내 식별정보 등)
- is_faq, owner_verified 등 OWNER_CONFIRMATIONS 연관 값은 Spring이 전달하는 데이터
  기준으로 채움 (이 레포가 OWNER_CONFIRMATIONS 테이블을 직접 조회하지 않음)
"""


async def export_training_data(workspace_id: str, dataset_version: str) -> str:
    """workspace_id 기준 LoRA 학습데이터 JSONL을 생성하고 경로를 반환. TODO: 구현."""
    raise NotImplementedError
