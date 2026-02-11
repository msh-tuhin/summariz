"""Shared utilities for the YouTube Content Summarizer."""

import logging
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import markdown
from xhtml2pdf import pisa

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


def save_summary_to_pdf(summary: str, video_id: str, video_title: str, output_dir: Path | None = None) -> Path:
    """Save a markdown summary to a PDF file.

    Args:
        summary: The markdown summary text to save.
        video_id: The YouTube video ID.
        video_title: The title of the video.
        output_dir: Directory to save the PDF. Defaults to current directory.

    Returns:
        Path to the saved PDF file.
    """
    output_dir = output_dir or Path.cwd()
    output_path = output_dir / f"youtube_{video_id}.pdf"

    # Convert markdown to HTML
    html_content = markdown.markdown(summary, extensions=['extra', 'codehilite'])

    # Wrap in HTML template with styling
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.6;
                margin: 40px;
            }}
            h1 {{
                color: #333;
                font-size: 20px;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #444;
                font-size: 16px;
                margin-top: 20px;
            }}
            h3 {{
                color: #555;
                font-size: 14px;
            }}
            .video-id {{
                color: #888;
                font-style: italic;
                font-size: 10px;
                margin-bottom: 20px;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                font-family: monospace;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 3px solid #ccc;
                margin-left: 0;
                padding-left: 15px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>{video_title}</h1>
        <div class="video-id">Video ID: {video_id}</div>
        {html_content}
    </body>
    </html>
    """

    with open(output_path, "wb") as pdf_file:
        pisa.CreatePDF(html_template, dest=pdf_file)

    return output_path
