"""Background job management for web service."""

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .cache import CACHE_DIR


class JobStatus(str, Enum):
    """Status of a summarization job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


JOBS_DIR = CACHE_DIR / "jobs"


@dataclass
class Job:
    """Represents a summarization job."""

    job_id: str
    video_id: str
    url: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    progress: str = "Waiting to start..."
    video_title: str | None = None
    pdf_path: str | None = None
    summary: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    @property
    def job_dir(self) -> Path:
        """Directory for this job's state."""
        return JOBS_DIR / self.job_id

    @property
    def state_file(self) -> Path:
        """Path to the job state file."""
        return self.job_dir / "job.json"

    def save(self) -> None:
        """Persist job state to disk."""
        self.job_dir.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["status"] = self.status.value
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, job_id: str) -> "Job | None":
        """Load job from disk.

        Args:
            job_id: The job ID to load.

        Returns:
            Job instance or None if not found.
        """
        state_file = JOBS_DIR / job_id / "job.json"
        if not state_file.exists():
            return None
        data = json.loads(state_file.read_text(encoding="utf-8"))
        data["status"] = JobStatus(data["status"])
        return cls(**data)

    def update_progress(self, message: str) -> None:
        """Update progress message and save.

        Args:
            message: Progress message to display.
        """
        self.progress = message
        self.save()

    def mark_processing(self) -> None:
        """Mark job as processing."""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def mark_completed(self, video_title: str, pdf_path: Path, summary: str) -> None:
        """Mark job as completed.

        Args:
            video_title: Title of the video.
            pdf_path: Path to the generated PDF.
            summary: The generated summary text.
        """
        self.status = JobStatus.COMPLETED
        self.video_title = video_title
        self.pdf_path = str(pdf_path)
        self.summary = summary
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.progress = "Complete"
        self.save()

    def mark_failed(self, error: str) -> None:
        """Mark job as failed.

        Args:
            error: Error message describing the failure.
        """
        self.status = JobStatus.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.progress = "Failed"
        self.save()


def create_job(url: str, video_id: str, config: dict[str, Any]) -> Job:
    """Create a new job.

    Args:
        url: YouTube video URL.
        video_id: Extracted video ID.
        config: Job configuration dict.

    Returns:
        The created Job instance.
    """
    job = Job(
        job_id=uuid.uuid4().hex[:12],
        video_id=video_id,
        url=url,
        config=config,
    )
    job.save()
    return job


def get_job(job_id: str) -> Job | None:
    """Get job by ID.

    Args:
        job_id: The job ID to retrieve.

    Returns:
        Job instance or None if not found.
    """
    return Job.load(job_id)
