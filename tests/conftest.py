"""Guard integration work against accidental use of a real or shared database."""

import os
from collections.abc import Iterator
from urllib.parse import urlsplit

import psycopg
import pytest
from psycopg import Connection
from psycopg.rows import dict_row


def validate_test_database_url(value: str) -> None:
    url = urlsplit(value)
    if (
        url.scheme not in {"postgresql", "postgres"}
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.path != "/ai_quality_test"
        or url.query
        or url.fragment
        or not url.username
        or not url.password
    ):
        raise ValueError(
            "TEST_DATABASE_URL must target the dedicated local ai_quality_test database"
        )


@pytest.fixture
def database() -> Iterator[Connection]:
    value = os.getenv("TEST_DATABASE_URL")
    if not value:
        if os.getenv("REQUIRE_INTEGRATION_TESTS") == "1":
            pytest.fail("TEST_DATABASE_URL is required by CI")
        pytest.skip("TEST_DATABASE_URL not set; no PostgreSQL evidence")
    validate_test_database_url(value)
    connection = psycopg.connect(value, connect_timeout=3, row_factory=dict_row)
    try:
        assert connection.execute("SELECT current_database() AS name").fetchone()["name"] == (
            "ai_quality_test"
        )
        yield connection
    finally:
        connection.rollback()
        connection.close()


def pytest_collection_finish(session: pytest.Session) -> None:
    if os.getenv("REQUIRE_INTEGRATION_TESTS") != "1":
        return
    integration_items = [item for item in session.items if item.get_closest_marker("integration")]
    if len(integration_items) < 8:
        raise pytest.UsageError("CI must collect at least eight PostgreSQL integration tests")
