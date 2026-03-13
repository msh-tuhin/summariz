"""FastAPI route handlers."""

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from ..cache import get_video_cache_id, get_cache_dir
from ..jobs import create_job, get_job, JobStatus
from ..pipeline import PipelineConfig, run_pipeline
from .models import (
    SummarizeRequest,
    SummarizeResponse,
    JobStatusResponse,
    SummaryResponse,
    HealthResponse,
)

logger = logging.getLogger("content_summary.api")

router = APIRouter(prefix="/api/v1")


def run_summarization_job(job_id: str) -> None:
    """Background task to run the pipeline.

    Args:
        job_id: The job ID to process.
    """
    job = get_job(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return

    try:
        job.mark_processing()

        # Save PDF to cache directory for web service
        output_dir = get_cache_dir(job.video_id)

        config = PipelineConfig(
            url=job.url,
            llm=job.config.get("llm", "ollama"),
            model=job.config.get("model"),
            transcriber=job.config.get("transcriber", "whisperx"),
            output_dir=output_dir,
            force_audio=job.config.get("force_audio", False),
            use_cache=not job.config.get("no_cache", False),
        )

        result = run_pipeline(
            config,
            progress_callback=job.update_progress,
        )

        job.mark_completed(
            video_title=result.video_info.title,
            pdf_path=result.pdf_path,
            summary=result.summary,
        )

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job.mark_failed(str(e))


@router.post("/summarize", response_model=SummarizeResponse, status_code=202)
async def submit_job(
    request: SummarizeRequest,
    background_tasks: BackgroundTasks,
):
    """Submit a new summarization job."""
    try:
        video_id = get_video_cache_id(request.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        url=request.url,
        video_id=video_id,
        config={
            "llm": request.llm,
            "model": request.model,
            "transcriber": request.transcriber,
            "force_audio": request.force_audio,
            "no_cache": request.no_cache,
        },
    )

    background_tasks.add_task(run_summarization_job, job.job_id)

    return SummarizeResponse(
        job_id=job.job_id,
        status=job.status.value,
        video_id=video_id,
        status_url=f"/api/v1/jobs/{job.job_id}",
        message="Job submitted successfully",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    pdf_url = None
    summary_url = None
    if job.status == JobStatus.COMPLETED:
        pdf_url = f"/api/v1/jobs/{job_id}/pdf"
        summary_url = f"/api/v1/jobs/{job_id}/summary"

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        video_id=job.video_id,
        video_title=job.video_title,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        pdf_url=pdf_url,
        summary_url=summary_url,
        error=job.error,
    )


@router.get("/jobs/{job_id}/pdf")
async def download_pdf(job_id: str):
    """Download the generated PDF."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    pdf_path = Path(job.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.get("/jobs/{job_id}/summary", response_model=SummaryResponse)
async def get_summary(job_id: str):
    """Get the summary text."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    return SummaryResponse(
        job_id=job.job_id,
        video_title=job.video_title,
        summary=job.summary,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )
