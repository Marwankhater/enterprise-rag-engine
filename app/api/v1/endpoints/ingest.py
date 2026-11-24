"""Upload a document for background ingestion, then poll the job."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.v1.deps import Services, get_services
from app.core.config import AccessLevel, Principal
from app.core.observability import request_id_var
from app.core.security import rate_limited, require_ingest
from app.db.models import AuditLog, IngestJob
from app.ingestion.parser import SUPPORTED

router = APIRouter(prefix="/ingest", tags=["ingest"])

LevelName = Literal["public", "internal", "confidential", "restricted"]


class JobOut(BaseModel):
    job_id: str
    status: str
    filename: str
    department_id: str | None
    access_level: LevelName
    document_id: str | None = None
    parents: int = 0
    children: int = 0
    error: str | None = None


async def save_upload(file: UploadFile, target: Path, limit: int) -> int:
    """Stream the upload to disk off the event loop. Returns bytes written, or -1 if over the limit."""
    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
    out = await asyncio.to_thread(target.open, "wb")
    written = 0
    try:
        while chunk := await file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                break
            await asyncio.to_thread(out.write, chunk)
    finally:
        await asyncio.to_thread(out.close)
    if written > limit:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        return -1
    return written


def _out(job: IngestJob) -> JobOut:
    return JobOut(
        job_id=str(job.id),
        status=job.status,
        filename=job.filename,
        department_id=job.department_id,
        access_level=AccessLevel(job.access_level).name.lower(),  # type: ignore[arg-type]
        document_id=str(job.document_id) if job.document_id else None,
        parents=job.parents,
        children=job.children,
        error=job.error,
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobOut,
    summary="Queue a PDF, DOCX, Markdown or text file for ingestion",
)
async def ingest(
    principal: Annotated[Principal, Depends(require_ingest)],
    _: Annotated[Principal, Depends(rate_limited)],
    services: Annotated[Services, Depends(get_services)],
    file: Annotated[UploadFile, File()],
    access_level: Annotated[LevelName, Form()] = "internal",
    department_id: Annotated[str | None, Form(pattern=r"^[a-z0-9_\-]{2,64}$")] = None,
) -> JobOut:
    settings = services.settings
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Supported types: {', '.join(sorted(SUPPORTED))}")
    level = AccessLevel[access_level.upper()]
    # Nobody can publish a document they would not be allowed to read themselves.
    if level > principal.clearance:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Access level is above your own clearance")
    if department_id is not None and department_id not in principal.departments:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only ingest into your own departments")

    job_id = uuid.uuid4()
    target = Path(settings.upload_dir) / f"{job_id}{suffix}"
    written = await save_upload(file, target, settings.max_upload_mb * 1024 * 1024)
    if written < 0:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File is larger than {settings.max_upload_mb} MB"
        )

    async with services.sessions() as session:
        job = IngestJob(
            id=job_id,
            filename=(file.filename or target.name)[:500],
            stored_path=str(target),
            requested_by=principal.name,
            department_id=department_id,
            access_level=int(level),
        )
        session.add(job)
        session.add(
            AuditLog(
                principal=principal.name,
                action="ingest_requested",
                request_id=request_id_var.get(),
                detail={
                    "job_id": str(job_id),
                    "file": job.filename,
                    "bytes": written,
                    "access_level": access_level,
                    "department_id": department_id,
                },
            )
        )
        await session.commit()

    from app.ingestion.tasks import process_ingest_job, run_job

    if settings.ingest_eager:
        await run_job(job_id, embedder=services.embedder)
    else:
        process_ingest_job.delay(str(job_id))
    async with services.sessions() as session:
        fresh = await session.get(IngestJob, job_id)
        assert fresh is not None
        return _out(fresh)


@router.get("/{job_id}", response_model=JobOut, summary="Status of an ingestion job you started")
async def job_status(
    job_id: uuid.UUID,
    principal: Annotated[Principal, Depends(rate_limited)],
    services: Annotated[Services, Depends(get_services)],
) -> JobOut:
    async with services.sessions() as session:
        job = await session.get(IngestJob, job_id)
    if job is None or job.requested_by != principal.name:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return _out(job)
