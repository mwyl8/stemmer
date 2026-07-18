"""FastAPI: endpoints only — wires modules, no heavy logic. Thin switchboard.

Nothing in here does fetch/decode/separate/encode work itself: POST /jobs
does the minimal synchronous work an HTTP handler can't defer (stream an
upload to disk, or a cheap structural URL check) and hands the job straight
to the worker pool. All heavy lifting happens in pool.py's background
workers — the request handler never blocks on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend import jobs
from backend.config import DEFAULT_MODE, DEFAULT_TIER, MAX_UPLOAD_BYTES, MODES, TIERS
from backend.ingest.fetch import InvalidURLError, classify_platform, validate_url
from backend.pool import WorkerPool
from backend.storage import init_db

_UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    pool = WorkerPool()
    await pool.start()
    app.state.pool = pool
    purge_task = asyncio.create_task(jobs.run_purge_loop())
    try:
        yield
    finally:
        purge_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await purge_task
        await pool.shutdown()


app = FastAPI(title="Stemmer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/jobs", status_code=201)
async def submit_job(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        job_id = await _submit_file_job(request)
    else:
        job_id = await _submit_url_job(request)

    await request.app.state.pool.submit(job_id)
    return {"job_id": job_id}


async def _submit_file_job(request: Request) -> str:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(422, "multipart submission requires a 'file' field")
    mode, tier = _validate_mode_tier(form.get("mode"), form.get("tier"))

    upload_dir = Path(tempfile.mkdtemp(prefix="stemmer-upload-"))
    dest = upload_dir / (upload.filename or "upload")
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES} byte cap")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    finally:
        await upload.close()

    return jobs.create_job(mode, tier, "upload", str(dest))


async def _submit_url_job(request: Request) -> str:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(422, "expected a JSON body with a 'url' field") from exc

    url = body.get("url")
    if not url:
        raise HTTPException(422, "JSON submission requires a 'url' field")
    mode, tier = _validate_mode_tier(body.get("mode"), body.get("tier"))

    try:
        validate_url(url)
        source_type = classify_platform(url)
    except InvalidURLError as exc:
        raise HTTPException(400, str(exc)) from exc

    return jobs.create_job(mode, tier, source_type, url)


def _validate_mode_tier(mode: str | None, tier: str | None) -> tuple[str, str]:
    mode = mode or DEFAULT_MODE
    tier = tier or DEFAULT_TIER
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    if tier not in TIERS:
        raise HTTPException(422, f"tier must be one of {tuple(TIERS)}")
    return mode, tier


def _job_summary(job: jobs.Job) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "mode": job.mode,
        "tier": job.tier,
        "error": job.error,
        "created_at": job.created_at,
        "expires_at": job.expires_at,
        "stems": [{"name": s.name, "format": s.format, "duration": s.duration} for s in job.stems],
    }


def _get_job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_summary(_get_job_or_404(job_id))


@app.get("/jobs/{job_id}/stems")
def list_stems(job_id: str) -> list[dict]:
    job = _get_job_or_404(job_id)
    return [{"name": s.name, "format": s.format, "duration": s.duration} for s in job.stems]


@app.get("/jobs/{job_id}/stems/{name}")
def get_stem(job_id: str, name: str, format: str = "mp3") -> FileResponse:
    if format not in ("wav", "mp3"):
        raise HTTPException(422, "format must be 'wav' or 'mp3'")
    job = _get_job_or_404(job_id)
    stem = next((s for s in job.stems if s.name == name and s.format == format), None)
    if stem is None:
        raise HTTPException(404, f"no {format} stem named {name!r} for this job")
    media_type = "audio/wav" if format == "wav" else "audio/mpeg"
    return FileResponse(stem.path, media_type=media_type, filename=f"{name}.{format}")


@app.get("/jobs/{job_id}/download")
def download_all(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, f"job is not done yet (status={job.status})")
    wav_stems = [s for s in job.stems if s.format == "wav"]
    if not wav_stems:
        raise HTTPException(404, "no stems available for this job")

    fd, zip_name = tempfile.mkstemp(suffix=".zip", prefix="stemmer-download-")
    os.close(fd)  # mkstemp hands back an open fd; ZipFile below reopens by path
    zip_path = Path(zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem in wav_stems:
            zf.write(stem.path, arcname=f"{stem.name}.wav")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{job_id}.zip",
        background=BackgroundTask(lambda: zip_path.unlink(missing_ok=True)),
    )


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> Response:
    if not jobs.purge_job(job_id):
        raise HTTPException(404, "job not found")
    return Response(status_code=204)
