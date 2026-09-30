"""
workspace별 LoRA 학습데이터 export ETL.

입력 레코드는 Spring이 API payload로 전달합니다(POST /api/training-data/export).
이 레포는 CHAT_MESSAGES / OWNER_CONFIRMATIONS 등 Spring 소유 테이블을 조회하지 않으며,
user_id 등 개인 식별자는 받지도 기록하지도 않습니다.

- original_question / final_answer는 PII 스크러빙(app/core/utils/pii.py) 후 기록합니다.
- 출력: {training_data_dir}/{workspace_id}/{dataset_version}.jsonl (워크스페이스별 격리)
- 동일 workspace_id + dataset_version이 이미 있으면 덮어씁니다(재실행 시 멱등).
"""

import json
from pathlib import Path

from app.config import get_settings
from app.core.utils.pii import scrub_text
from app.schemas.training_data import (
    TrainingDataExportRequest,
    TrainingDataExportResponse,
)


def export_training_data(
    request: TrainingDataExportRequest,
) -> TrainingDataExportResponse:
    """레코드를 스크러빙해 JSONL로 저장하고 결과 요약을 반환합니다."""
    base = Path(get_settings().training_data_dir).resolve()
    out_dir = (base / request.workspace_id).resolve()
    if base not in out_dir.parents:  # 스키마 검증 외 방어선
        raise ValueError("invalid workspace_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{request.dataset_version}.jsonl"

    scrubbed_total = 0
    lines: list[str] = []
    for rec in request.records:
        question, n_q = scrub_text(rec.original_question)
        answer, n_a = scrub_text(rec.final_answer)
        scrubbed_total += n_q + n_a
        lines.append(
            json.dumps(
                {
                    "original_question": question,
                    "target_role": rec.target_role.value,
                    "final_answer": answer,
                    "source": rec.source,
                    "prompt_version": rec.prompt_version,
                    "is_faq": rec.is_faq,
                    "owner_verified": rec.owner_verified,
                    "dataset_version": request.dataset_version,
                    "workspace_id": request.workspace_id,
                },
                ensure_ascii=False,
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return TrainingDataExportResponse(
        workspace_id=request.workspace_id,
        dataset_version=request.dataset_version,
        path=str(path),
        record_count=len(lines),
        scrubbed_field_count=scrubbed_total,
    )
