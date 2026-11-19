"""The Compose smoke probe must tolerate connections reset during API startup."""

from scripts import compose_smoke


def test_wait_for_ready_retries_after_connection_reset(monkeypatch):
    calls = 0

    def transient_reset(path):
        nonlocal calls
        assert path == "/ready"
        calls += 1
        if calls == 1:
            raise ConnectionResetError("API still starting")
        return {"status": "ready"}

    monkeypatch.setattr(compose_smoke, "get_json", transient_reset)
    monkeypatch.setattr(compose_smoke.time, "sleep", lambda _: None)

    compose_smoke.wait_for_ready(timeout_seconds=3)
    assert calls == 2
