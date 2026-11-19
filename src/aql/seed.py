"""CLI entry point for synthetic seed. Never deletes or updates existing corpus rows."""

import argparse
from pathlib import Path

from aql.corpus import seed_corpus
from aql.database import open_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=Path("data/synthetic_docs/corpus_v1.json"))
    args = parser.parse_args()
    with open_database() as connection:
        documents, chunks = seed_corpus(connection, args.file)
    print(f"Corpus seed verified: documents={documents}, chunks={chunks}")


if __name__ == "__main__":
    main()
