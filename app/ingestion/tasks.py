"""Celery tasks for ingestion, so parsing and embedding never block the API.

Each task opens its own engine with no pool: Celery runs every task in a fresh event loop and an
asyncpg connection cannot be reused across loops. The embedding model is loaded once per worker
process and reused.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.observability import configure_logging, configure_tracing, span
from app.db.models import AuditLog, IngestJob
from app.ingestion.pipeline import ingest_file
from app.retrieval.embeddings import Embedder, build_embedder

log = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery("rag", broker=settings.broker_url)
celery_app.conf.update(
    task_acks_late=True,  # a crash mid-task puts the job back on the queue
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # one heavy job at a time per process
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    task_always_eager=settings.ingest_eager,
    broker_connection_retry_on_startup=True,
)

CACHE_EPOCH_KEY = "cache:epoch"


@worker_process_init.connect
def _init_worker(**_: object) -> None:
    configure_logging(settings.log_level)
    configure_tracing(settings)


@lru_cache
def worker_embedder() -> Embedder:
    return build_embedder(settings)


async def run_job(job_id: uuid.UUID, embedder: Embedder | None = None) -> str:
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            job = await session.get(IngestJob, job_id)
            if job is None:
                log.error("ingest job not found", extra={"job_id": str(job_id)})
                return "missing"
            if job.status in ("done", "duplicate"):
                return job.status  # redelivered after success: nothing to do
            job.status = "running"
            await session.commit()
            path = Path(job.stored_path)
        with span("ingest.job", job_id=str(job_id), file=path.name):
            try:
                async with sessions() as session:
                    result = await ingest_file(
                        session,
                        embedder or worker_embedder(),
                        settings,
                        path,
                        source=job.filename,
                        department_id=job.department_id,
                        access_level=job.access_level,
                    )
                status, error = ("duplicate" if result.duplicate else "done"), None
            except Exception as exc:
                log.exception("ingest failed", extra={"job_id": str(job_id)})
                result, status, error = None, "failed", f"{type(exc).__name__}: {exc}"[:2000]
        async with sessions() as session:
            job = await session.get(IngestJob, job_id)
            assert job is not None
            job.status, job.error = status, error
            job.finished_at = datetime.now(UTC)
            if result is not None:
                job.document_id = uuid.UUID(result.document_id)
                job.parents, job.children = result.parents, result.children
            session.add(
                AuditLog(
                    principal=job.requested_by,
                    action="ingest",
                    detail={"job_id": str(job_id), "status": status, "file": job.filename},
                )
            )
            await session.commit()
        if status == "done":
            await _bump_cache_epoch()
            await asyncio.to_thread(path.unlink, missing_ok=True)
        return status
    finally:
        await engine.dispose()


async def _bump_cache_epoch() -> None:
    """New documents can change answers, so cached answers from before this point stop matching."""
    from redis.asyncio import Redis

    redis = Redis.from_url(settings.redis_url)
    try:
        await redis.incr(CACHE_EPOCH_KEY)
    except Exception:
        log.warning("could not bump cache epoch")
    finally:
        await redis.aclose()


@celery_app.task(name="ingest.process", bind=True, max_retries=3, default_retry_delay=30)
def process_ingest_job(self, job_id: str) -> str:
    try:
        return asyncio.run(run_job(uuid.UUID(job_id)))
    except OSError as exc:  # database or broker briefly unreachable
        raise self.retry(exc=exc) from exc
