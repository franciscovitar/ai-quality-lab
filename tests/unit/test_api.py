from fastapi.testclient import TestClient

from aql.api import app

client = TestClient(app)


def test_health_needs_no_database() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_rejects_invalid_fields_before_database() -> None:
    for body in [
        {"question": "no", "top_k": 5},
        {"question": "valid question", "top_k": 11},
        {"question": "valid question", "unexpected": "field"},
    ]:
        response = client.post("/v1/retrieval/search", json=body)
        assert response.status_code == 422
