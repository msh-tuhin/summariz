"""Shared utilities for the YouTube Content Summarizer."""

import logging
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO.

    Returns:
        Configured logger instance.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("content_summary")


def get_api_key(key_name: str) -> str:
    """Get an API key from environment variables.

    Args:
        key_name: The name of the environment variable.

    Returns:
        The API key value.

    Raises:
        ValueError: If the API key is not set.
    """
    key = os.getenv(key_name)
    if not key:
        raise ValueError(
            f"{key_name} not found. Please set it in your environment or .env file."
        )
    return key


def get_temp_dir() -> Path:
    """Get the temporary directory for downloaded files.

    Returns:
        Path to the temporary directory.
    """
    temp_dir = Path(tempfile.gettempdir()) / "content_summary"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename.

    Args:
        name: The original string.

    Returns:
        A sanitized filename-safe string.
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized


def extract_video_id(url: str) -> str | None:
    """Extract the video ID from a YouTube URL.

    Args:
        url: The YouTube URL.

    Returns:
        The video ID, or None if not found.
    """
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def format_duration(seconds: int) -> str:
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s".
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)
