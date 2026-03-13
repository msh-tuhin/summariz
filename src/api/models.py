"""Pydantic models for API request/response."""

from typing import Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """Request body for POST /summarize."""

    url: str = Field(..., description="YouTube video URL")
    llm: Literal["openai", "anthropic", "ollama"] = "ollama"
    model: str | None = None
    transcriber: Literal["whisperx", "gladia"] = "whisperx"
    force_audio: bool = False
    no_cache: bool = False


class SummarizeResponse(BaseModel):
    """Response for POST /summarize."""

    job_id: str
    status: str
    video_id: str
    status_url: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for GET /jobs/{job_id}."""

    job_id: str
    status: str
    video_id: str
    video_title: str | None
    progress: str
    created_at: str
    started_at: str | None
    completed_at: str | None
    pdf_url: str | None
    summary_url: str | None
    error: str | None


class SummaryResponse(BaseModel):
    """Response for GET /jobs/{job_id}/summary."""

    job_id: str
    video_title: str
    summary: str


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
