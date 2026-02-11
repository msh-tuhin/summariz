"""Caching system for resumable pipeline execution."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .utils import extract_video_id

logger = logging.getLogger("content_summary")

# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "content_summary"


def get_cache_dir(video_id: str) -> Path:
    """Get the cache directory for a specific video.

    Args:
        video_id: The YouTube video ID.

    Returns:
        Path to the video's cache directory.
    """
    cache_dir = CACHE_DIR / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_video_cache_id(url: str) -> str:
    """Extract video ID from URL for caching.

    Args:
        url: The YouTube URL.

    Returns:
        The video ID.

    Raises:
        ValueError: If video ID cannot be extracted.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return video_id


class PipelineCache:
    """Cache manager for the summarization pipeline."""

    def __init__(self, video_id: str):
        """Initialize cache for a video.

        Args:
            video_id: The YouTube video ID.
        """
        self.video_id = video_id
        self.cache_dir = get_cache_dir(video_id)
        logger.debug(f"Cache directory: {self.cache_dir}")

    @property
    def info_path(self) -> Path:
        """Path to cached video info."""
        return self.cache_dir / "info.json"

    @property
    def content_path(self) -> Path:
        """Path to cached text content (subtitles or transcription)."""
        return self.cache_dir / "content.txt"

    @property
    def audio_path(self) -> Path:
        """Path to cached audio file."""
        return self.cache_dir / "audio.mp3"

    def summary_path(self, provider: str, model: str) -> Path:
        """Path to cached summary for a specific provider/model.

        Args:
            provider: LLM provider name.
            model: Model name.

        Returns:
            Path to the summary file.
        """
        safe_model = model.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"summary_{provider}_{safe_model}.md"

    # Video info caching
    def has_video_info(self) -> bool:
        """Check if video info is cached."""
        return self.info_path.exists()

    def save_video_info(self, video_info) -> None:
        """Save video info to cache.

        Args:
            video_info: VideoInfo dataclass instance.
        """
        data = asdict(video_info)
        self.info_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug(f"Cached video info: {self.info_path}")

    def load_video_info(self):
        """Load video info from cache.

        Returns:
            VideoInfo instance or None if not cached.
        """
        if not self.has_video_info():
            return None

        from .youtube import VideoInfo

        data = json.loads(self.info_path.read_text(encoding="utf-8"))
        logger.info("Loaded video info from cache")
        return VideoInfo(**data)

    # Content caching (subtitles or transcription)
    def has_content(self) -> bool:
        """Check if text content is cached."""
        return self.content_path.exists()

    def save_content(self, content: str) -> None:
        """Save text content to cache.

        Args:
            content: The text content (subtitles or transcription).
        """
        self.content_path.write_text(content, encoding="utf-8")
        logger.debug(f"Cached content: {self.content_path}")

    def load_content(self) -> str | None:
        """Load text content from cache.

        Returns:
            The cached content or None if not cached.
        """
        if not self.has_content():
            return None

        logger.info("Loaded content from cache")
        return self.content_path.read_text(encoding="utf-8")

    # Audio caching
    def has_audio(self) -> bool:
        """Check if audio file is cached."""
        return self.audio_path.exists()

    def get_audio_path(self) -> Path | None:
        """Get path to cached audio file.

        Returns:
            Path to audio file or None if not cached.
        """
        if self.has_audio():
            logger.info("Using cached audio file")
            return self.audio_path
        return None

    def save_audio(self, source_path: Path) -> Path:
        """Copy audio file to cache.

        Args:
            source_path: Path to the source audio file.

        Returns:
            Path to the cached audio file.
        """
        import shutil

        shutil.copy2(source_path, self.audio_path)
        logger.debug(f"Cached audio: {self.audio_path}")
        return self.audio_path

    # Summary caching
    def has_summary(self, provider: str, model: str) -> bool:
        """Check if summary is cached for provider/model.

        Args:
            provider: LLM provider name.
            model: Model name.

        Returns:
            True if summary is cached.
        """
        return self.summary_path(provider, model).exists()

    def save_summary(self, summary: str, provider: str, model: str) -> None:
        """Save summary to cache.

        Args:
            summary: The generated summary.
            provider: LLM provider name.
            model: Model name.
        """
        path = self.summary_path(provider, model)
        path.write_text(summary, encoding="utf-8")
        logger.debug(f"Cached summary: {path}")

    def load_summary(self, provider: str, model: str) -> str | None:
        """Load summary from cache.

        Args:
            provider: LLM provider name.
            model: Model name.

        Returns:
            The cached summary or None if not cached.
        """
        if not self.has_summary(provider, model):
            return None

        logger.info(f"Loaded summary from cache ({provider}/{model})")
        return self.summary_path(provider, model).read_text(encoding="utf-8")

    def clear(self) -> None:
        """Clear all cached data for this video."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache for video: {self.video_id}")
