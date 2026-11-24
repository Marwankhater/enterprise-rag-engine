"""Ingest every document in examples/ with the access rules from examples/manifest.json.

python -m scripts.ingest_examples            # skips files that are already ingested
python -m scripts.ingest_examples --reset    # deletes all documents first
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import text

from app.core.config import AccessLevel, get_settings
from app.core.observability import configure_logging
from app.db.session import dispose, get_sessionmaker
from app.ingestion.pipeline import ingest_file
from app.retrieval.embeddings import build_embedder

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


async def main(reset: bool) -> None:
    settings = get_settings()
    configure_logging("WARNING")
    manifest = json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(settings)
    sessions = get_sessionmaker()
    if reset:
        async with sessions() as session:
            await session.execute(text("truncate documents cascade"))
            await session.commit()
    for name, rule in manifest.items():
        async with sessions() as session:
            result = await ingest_file(
                session,
                embedder,
                settings,
                EXAMPLES / name,
                department_id=rule.get("department_id"),
                access_level=int(AccessLevel[rule.get("access_level", "internal").upper()]),
            )
        state = "already ingested" if result.duplicate else f"{result.parents} parents, {result.children} children"
        print(f"{name}: {state}")
    await dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete all documents before ingesting")
    asyncio.run(main(parser.parse_args().reset))
