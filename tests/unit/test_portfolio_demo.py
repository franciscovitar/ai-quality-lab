"""The recruiter demo must show exact frozen model text, not invented examples."""

from pathlib import Path

from fastapi.testclient import TestClient

from aql import portfolio_demo
from aql.api import app
from aql.generation_evaluation import GeneratedBatch, PromptBatch

client = TestClient(app)


def test_demo_preserves_frozen_model_answers_and_sources() -> None:
    data = portfolio_demo.load_portfolio_demo()
    root = portfolio_demo.project_root()
    generated = GeneratedBatch.model_validate_json(
        (root / "data/evals/frozen_generation_v1/generated-outputs.json").read_bytes()
    )
    prompts = PromptBatch.model_validate_json(
        (root / "data/evals/frozen_generation_v1/prompts.json").read_bytes()
    )
    original_answers = {case.case_id: case.text for case in generated.cases}
    original_context = {
        case.case_id: [{"id": s.citation_id, "quote": s.quote} for s in case.context]
        for case in prompts.cases
    }

    assert data["mode"] == "frozen_synthetic_research_demo"
    assert data["independent_reviewers"] == 0
    assert [case["id"] for case in data["examples"]] == ["R101", "R106", "R108", "R105", "R110"]
    for example in data["examples"]:
        assert example["model_answer"] == original_answers[example["id"]]
        assert example["source_passages"] == original_context[example["id"]]
        assert "provisional" in example["method_limit"]
        assert example["plain_explanation"].endswith(".")
        assert len(example["plain_explanation"]) > 45


def test_demo_routes_need_no_database_or_model_service() -> None:
    page = client.get("/demo")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Cinco respuestas bajo la lupa" in page.text
    assert 'id="case-answer"' in page.text
    assert page.text.index('id="case-simple"') < page.text.index('id="case-question"')
    assert page.text.count('id="case-observation"') == 1
    assert "Sin generación en vivo" in page.text

    for asset, mime in [("demo.css", "text/css"), ("demo.js", "javascript")]:
        resource = client.get(f"/demo/assets/{asset}")
        assert resource.status_code == 200
        assert mime in resource.headers["content-type"]

    api = client.get("/v1/demo/examples")
    assert api.status_code == 200
    assert api.json()["examples"][1]["category"] == "contradicted"


def test_demo_returns_generic_error_if_snapshot_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        portfolio_demo,
        "project_root",
        lambda: Path("/nonexistent/frozen-snapshot-for-test"),
    )
    response = client.get("/v1/demo/examples")
    assert response.status_code == 503
    assert response.json() == {"detail": "demo_snapshot_unavailable"}


def test_ui_renders_snapshot_as_text_not_html() -> None:
    javascript = client.get("/demo/assets/demo.js").text
    assert ".textContent =" in javascript
    assert "innerHTML" not in javascript
    assert "eval(" not in javascript
    assert 'fetch("/v1/demo/examples"' in javascript
    assert 'byId("case-simple").textContent = item.plain_explanation' in javascript
    assert 'cases.findIndex((item) => item.id === "R106")' in javascript
