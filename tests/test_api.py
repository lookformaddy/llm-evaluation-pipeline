from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_eval_pipeline.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["dataset_cases"] == 12
    assert len(body["dataset_hash"]) == 64


def test_validate_dataset_endpoint():
    resp = client.post("/validate-dataset", json={
        "cases": [{"id": "a", "input": "q1"}, {"id": "b", "input": "q2"}]
    })
    assert resp.status_code == 200
    assert resp.json()["case_count"] == 2


def test_validate_dataset_duplicate_ids_422():
    resp = client.post("/validate-dataset", json={
        "cases": [{"id": "a", "input": "q1"}, {"id": "a", "input": "q2"}]
    })
    assert resp.status_code == 422


def test_score_endpoint(cases):
    case = cases[0]
    resp = client.post("/score", json={
        "case": case.model_dump(mode="json"),
        "output_text": case.expected_output,
        "citations": case.available_docs[:1],
    })
    assert resp.status_code == 200
    scores = resp.json()["scores"]
    assert set(scores) == {
        "regex_match", "semantic_similarity", "groundedness", "llm_judge"
    }
    assert scores["semantic_similarity"]["score"] > 0.8


def test_create_run_endpoint():
    resp = client.post("/runs", json={"config": {"name": "api-test"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"]["case_count"] == 12
    assert body["sut_config"]["name"] == "api-test"
    assert len(body["case_results"]) == 12


@pytest.mark.parametrize("path", ["/health", "/history"])
def test_get_endpoints_ok(path):
    assert client.get(path).status_code == 200
