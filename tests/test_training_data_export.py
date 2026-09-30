"""LoRA 학습데이터 export 파이프라인/엔드포인트 테스트. 출력 경로는 tmp_path로 격리."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.pipelines import export_training_data as pipeline
from app.schemas.training_data import TrainingDataExportRequest

client = TestClient(app)
URL = "/api/training-data/export"


@pytest.fixture(autouse=True)
def training_dir(monkeypatch, tmp_path):
    root = tmp_path / "training"
    monkeypatch.setattr(
        pipeline, "get_settings", lambda: SimpleNamespace(training_data_dir=str(root))
    )
    return root


def _rec(i=0, **kw):
    base = {
        "original_question": f"질문 {i}",
        "target_role": "developer",
        "final_answer": f"답변 {i}",
        "source": "chat",
        "prompt_version": "v1",
    }
    base.update(kw)
    return base


def _payload(records, ws="ws-1", ver="v1"):
    return {"workspace_id": ws, "dataset_version": ver, "records": records}


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f.read().splitlines()]


def test_ten_records_ten_lines(training_dir):
    req = TrainingDataExportRequest(**_payload([_rec(i) for i in range(10)]))
    res = pipeline.export_training_data(req)
    rows = _lines(res.path)
    assert res.record_count == 10 and len(rows) == 10
    assert set(rows[0]) == {
        "original_question", "target_role", "final_answer", "source",
        "prompt_version", "is_faq", "owner_verified", "dataset_version", "workspace_id",
    }
    assert rows[0]["target_role"] == "developer"
    assert res.path.endswith("v1.jsonl")


def test_pii_scrubbed_and_korean_preserved():
    req = TrainingDataExportRequest(
        **_payload([_rec(original_question="메일 a@b.com 입니다", final_answer="010-1234-5678 로 연락")])
    )
    res = pipeline.export_training_data(req)
    raw = open(res.path, encoding="utf-8").read()
    assert "a@b.com" not in raw and "010-1234-5678" not in raw
    assert "[EMAIL]" in raw and "[PHONE]" in raw and "메일" in raw
    assert res.scrubbed_field_count == 2


def test_workspace_isolation_and_overwrite(training_dir):
    a = pipeline.export_training_data(TrainingDataExportRequest(**_payload([_rec()], ws="ws-a")))
    b = pipeline.export_training_data(TrainingDataExportRequest(**_payload([_rec()], ws="ws-b")))
    assert a.path != b.path
    assert (training_dir / "ws-a" / "v1.jsonl").exists()
    assert (training_dir / "ws-b" / "v1.jsonl").exists()
    again = pipeline.export_training_data(
        TrainingDataExportRequest(**_payload([_rec(1), _rec(2)], ws="ws-a"))
    )
    assert len(_lines(again.path)) == 2


def test_endpoint_auth_ok_and_validation(training_dir):
    key = get_settings().internal_api_key
    body = _payload([_rec()])
    # 헤더 자체가 없으면 (Header(...) 필수) 422, 값이 틀리면 401
    assert client.post(URL, json=body).status_code == 422
    assert client.post(URL, json=body, headers={"X-Internal-Api-Key": "wrong"}).status_code == 401

    ok = client.post(URL, json=body, headers={"X-Internal-Api-Key": key})
    assert ok.status_code == 200
    assert ok.json()["record_count"] == 1

    empty = client.post(URL, json=_payload([]), headers={"X-Internal-Api-Key": key})
    assert empty.status_code == 422
    bad_ws = client.post(URL, json=_payload([_rec()], ws="../evil"), headers={"X-Internal-Api-Key": key})
    assert bad_ws.status_code == 422
    # 선두 점("." / "..")은 정규식만으로 막혀야 한다 (500이 아니라 422)
    for bad in ("..", ".", ".hidden"):
        assert client.post(URL, json=_payload([_rec()], ws=bad), headers={"X-Internal-Api-Key": key}).status_code == 422
        assert client.post(URL, json={**_payload([_rec()]), "dataset_version": bad}, headers={"X-Internal-Api-Key": key}).status_code == 422
