"""Public, local-only HTTP surface for the first retrieval checkpoint."""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from psycopg import Error as PsycopgError
from pydantic import BaseModel, ConfigDict, Field

from aql.database import DatabaseNotConfigured, is_ready, open_database
from aql.portfolio_demo import load_portfolio_demo
from aql.retrieval import SearchHit, search_lexical

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Quality Lab", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/demo/assets", StaticFiles(directory=WEB_DIR), name="demo-assets")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
    strategy: str
    hits: list[SearchHit]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo", include_in_schema=False)
def portfolio_demo_page() -> FileResponse:
    """Static, local-first explainer; no generator, secret or database required."""
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/v1/demo/examples")
def portfolio_demo_examples() -> dict:
    """Existing SHA-bound experimental results, not an AI answer endpoint."""
    try:
        return load_portfolio_demo()
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Frozen portfolio demo unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="demo_snapshot_unavailable") from None


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        with open_database() as connection:
            if is_ready(connection):
                return {"status": "ready"}
    except (DatabaseNotConfigured, PsycopgError):
        logger.warning("Readiness probe cannot reach a migrated database")
    raise HTTPException(status_code=503, detail="database_not_ready")


@app.get("/v1/corpus/versions")
def corpus_versions() -> dict[str, list[dict[str, str | int]]]:
    try:
        with open_database() as connection:
            rows = connection.execute(
                """SELECT slug, version, title FROM documents
                   WHERE is_current ORDER BY slug"""
            ).fetchall()
    except (DatabaseNotConfigured, PsycopgError) as exc:
        logger.warning("Corpus query unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="database_unavailable") from None
    return {"documents": rows}


@app.post("/v1/retrieval/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        with open_database() as connection:
            hits = search_lexical(connection, request.question, request.top_k)
    except (DatabaseNotConfigured, PsycopgError) as exc:
        logger.warning("Retrieval unavailable: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="database_unavailable") from None
    return SearchResponse(strategy="lexical", hits=hits)
