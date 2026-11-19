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
        except (HTTPError, URLError, TimeoutError, ValueError):
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

    print(
        "Compose HTTP smoke passed: "
        f"health=200 ready=200 current_documents={len(documents)} "
        f"search_hits={len(hits)} current_policy=v2"
    )


if __name__ == "__main__":
    main()
