import pytest

from tests.conftest import validate_test_database_url


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pass@prod.example.com/ai_quality_test",
        "postgresql://user:pass@127.0.0.1/ai_quality",
        "postgresql://user:pass@localhost/ai_quality_test?options=-c%20search_path%3Dpublic",
        "postgresql://user:pass@localhost/ai_quality_test#fragment",
        "postgresql://user:pass@localhost/ai_quality_test/extra",
    ],
)
def test_database_gate_rejects_unsafe_targets(url: str) -> None:
    with pytest.raises(ValueError, match="dedicated local"):
        validate_test_database_url(url)


def test_database_gate_allows_only_disposable_database() -> None:
    validate_test_database_url("postgresql://user:pass@127.0.0.1:55435/ai_quality_test")
