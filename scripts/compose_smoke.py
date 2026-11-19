"""Bounded real-HTTP smoke against a freshly started local Compose stack."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:8001"


def get_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise AssertionError(f"Unexpected HTTP status: {response.status}")
        result = json.load(response)
        if not isinstance(result, dict):
            raise AssertionError("Expected a JSON object")
        return result


def wait_for_ready(timeout_seconds: int = 90) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if get_json("/ready") == {"status": "ready"}:
                return
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError(f"API did not become ready within {timeout_seconds}s")


def main() -> None:
    wait_for_ready()
    assert get_json("/health") == {"status": "ok"}

    documents = get_json("/v1/corpus/versions")["documents"]
    assert len(documents) == 6, "Expected six current synthetic documents"
    policy = [item for item in documents if item["slug"] == "webhook-delivery-policy"]
    assert len(policy) == 1 and policy[0]["version"] == 2

    search = get_json(
        "/v1/retrieval/search",
        {"question": "webhook delivery maximum budget five attempts", "top_k": 5},
    )
    assert search["strategy"] == "lexical"
    hits = search["hits"]
    assert hits, "Expected a source chunk"
    assert any(
        hit["document_slug"] == "webhook-delivery-policy" and hit["document_version"] == 2
        for hit in hits
    )
    assert all(hit["chunk_id"] and hit["content"] for hit in hits)

    demo = get_json("/v1/demo/examples")
    assert demo["mode"] == "frozen_synthetic_research_demo"
    assert demo["independent_reviewers"] == 0
    assert [case["id"] for case in demo["examples"]] == ["R101", "R106", "R108", "R105", "R110"]
    contradiction = demo["examples"][1]
    assert contradiction["author_claim_counts"]["contradicted"] == 1
    assert "3 days" in contradiction["model_answer"]
    assert any("thirty days" in passage["quote"] for passage in contradiction["source_passages"])
    for path, marker in [
        ("/demo", b'id="explorer-layout"'),
        ("/demo/assets/demo.js", b"selectCase"),
        ("/demo/assets/demo.css", b".explorer-layout"),
    ]:
        with urlopen(f"{BASE_URL}{path}", timeout=3) as response:
            assert response.status == 200
            assert marker in response.read()

    print(
        "Compose HTTP smoke passed: "
        f"health=200 ready=200 current_documents={len(documents)} "
        f"search_hits={len(hits)} current_policy=v2 "
        f"portfolio_demo={len(demo['examples'])} frozen_examples"
    )


if __name__ == "__main__":
    main()
