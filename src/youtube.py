"""YouTube subtitle and audio download functionality."""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv

from .utils import get_temp_dir, sanitize_filename

# Load environment variables
load_dotenv()

logger = logging.getLogger("content_summary")


def _get_yt_dlp_base_opts() -> dict:
    """Get base yt-dlp options with cookie and runtime support.

    Configurable via environment variables:
        YT_COOKIES_FILE: Path to cookies.txt file
        YT_NO_CHECK_CERTS: Set to "1" to skip SSL verification

    Note: JS runtime is handled separately via subprocess when needed.
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
    }

    # Cookie support
    cookies_file = os.environ.get("YT_COOKIES_FILE")
    if cookies_file and Path(cookies_file).exists():
        opts["cookiefile"] = cookies_file
        logger.debug(f"Using cookies file: {cookies_file}")
    elif cookies_file:
        logger.warning(f"Cookies file not found: {cookies_file}")

    # SSL verification
    if os.environ.get("YT_NO_CHECK_CERTS") == "1":
        opts["nocheckcertificate"] = True
        logger.debug("SSL certificate verification disabled")

    return opts


def _build_yt_dlp_command(url: str, extra_args: list[str] = None) -> list[str]:
    """Build yt-dlp command with environment-based options.

    Args:
        url: The YouTube URL.
        extra_args: Additional command line arguments.

    Returns:
        List of command arguments.
    """
    cmd = ["yt-dlp"]

    # Cookie support
    cookies_file = os.environ.get("YT_COOKIES_FILE")
    if cookies_file and Path(cookies_file).exists():
        cmd.extend(["--cookies", cookies_file])

    # JS runtime
    js_runtime = os.environ.get("YT_JS_RUNTIME")
    if js_runtime:
        cmd.extend(["--js-runtimes", f"node:{js_runtime}"])
        cmd.extend(["--remote-components", "ejs:github"])

    # SSL verification
    if os.environ.get("YT_NO_CHECK_CERTS") == "1":
        cmd.append("--no-check-certificates")

    if extra_args:
        cmd.extend(extra_args)

    cmd.append(url)
    return cmd


def _needs_js_runtime() -> bool:
    """Check if JS runtime is configured."""
    return bool(os.environ.get("YT_JS_RUNTIME"))


@dataclass
class VideoInfo:
    """Information about a YouTube video."""

    id: str
    title: str
    duration: int  # seconds
    channel: str
    description: str
    manual_subtitles: dict[str, list]  # language -> list of subtitle formats
    automatic_subtitles: dict[str, list]  # language -> list of subtitle formats

    @property
    def has_manual_subtitles(self) -> bool:
        """Check if the video has any manual subtitles."""
        return bool(self.manual_subtitles)

    def get_best_manual_subtitle_lang(self, preferred: list[str] | None = None) -> str | None:
        """Get the best available manual subtitle language.

        Args:
            preferred: List of preferred language codes in order.

        Returns:
            The best available language code, or None if no manual subtitles.
        """
        if not self.manual_subtitles:
            return None

        preferred = preferred or ["en", "en-US", "en-GB"]

        # Check preferred languages first
        for lang in preferred:
            if lang in self.manual_subtitles:
                return lang

        # Return first available language
        return next(iter(self.manual_subtitles.keys()))


def get_video_info(url: str) -> VideoInfo:
    """Fetch video metadata from YouTube.

    Args:
        url: The YouTube video URL.

    Returns:
        VideoInfo object with video metadata.

    Raises:
        ValueError: If the video cannot be found or accessed.
    """
    logger.info(f"Fetching video info for: {url}")

    if _needs_js_runtime():
        # Use subprocess for JS runtime support
        info = _get_video_info_subprocess(url)
    else:
        # Use Python API
        ydl_opts = _get_yt_dlp_base_opts()
        ydl_opts["extract_flat"] = False

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
            except yt_dlp.utils.DownloadError as e:
                raise ValueError(f"Failed to fetch video info: {e}") from e

    video_info = VideoInfo(
        id=info.get("id", ""),
        title=info.get("title", "Unknown Title"),
        duration=info.get("duration", 0),
        channel=info.get("channel", info.get("uploader", "Unknown Channel")),
        description=info.get("description", ""),
        manual_subtitles=info.get("subtitles", {}),
        automatic_subtitles=info.get("automatic_captions", {}),
    )

    logger.info(f"Video: {video_info.title}")
    logger.info(f"Duration: {video_info.duration}s")
    logger.info(f"Manual subtitles available: {list(video_info.manual_subtitles.keys())}")

    return video_info


def _get_video_info_subprocess(url: str) -> dict:
    """Fetch video info using subprocess (supports JS runtime).

    Args:
        url: The YouTube URL.

    Returns:
        Video info dict.

    Raises:
        ValueError: If fetching fails.
    """
    cmd = _build_yt_dlp_command(url, ["--dump-json", "--no-download"])
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to fetch video info: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse video info: {e}") from e


def download_subtitles(url: str, video_info: VideoInfo | None = None, lang: str | None = None) -> Path | None:
    """Download manual subtitles for a video.

    Args:
        url: The YouTube video URL.
        video_info: Optional pre-fetched video info.
        lang: Specific language to download. If None, uses best available.

    Returns:
        Path to the downloaded subtitle file, or None if no manual subtitles.
    """
    if video_info is None:
        video_info = get_video_info(url)

    if not video_info.has_manual_subtitles:
        logger.info("No manual subtitles available")
        return None

    # Determine language to download
    target_lang = lang or video_info.get_best_manual_subtitle_lang()
    if not target_lang:
        logger.info("No suitable subtitle language found")
        return None

    logger.info(f"Downloading subtitles in language: {target_lang}")

    temp_dir = get_temp_dir()
    output_template = str(temp_dir / f"{sanitize_filename(video_info.title)}")

    if _needs_js_runtime():
        # Use subprocess for JS runtime support
        cmd = _build_yt_dlp_command(url, [
            "--skip-download",
            "--write-subs",
            "--sub-langs", target_lang,
            "--sub-format", "vtt/srt/best",
            "-o", output_template,
        ])
        logger.debug(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    else:
        ydl_opts = _get_yt_dlp_base_opts()
        ydl_opts.update({
            "skip_download": True,
            "writesubtitles": True,
            "subtitleslangs": [target_lang],
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": output_template,
        })
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    # Find the downloaded subtitle file
    for ext in [".vtt", ".srt"]:
        subtitle_path = Path(f"{output_template}.{target_lang}{ext}")
        if subtitle_path.exists():
            logger.info(f"Subtitles downloaded to: {subtitle_path}")
            return subtitle_path

    logger.warning("Subtitle download completed but file not found")
    return None


def download_audio(url: str, video_info: VideoInfo | None = None) -> Path:
    """Download audio from a YouTube video.

    Args:
        url: The YouTube video URL.
        video_info: Optional pre-fetched video info.

    Returns:
        Path to the downloaded audio file.

    Raises:
        RuntimeError: If the download fails.
    """
    if video_info is None:
        video_info = get_video_info(url)

    logger.info(f"Downloading audio for: {video_info.title}")

    temp_dir = get_temp_dir()
    output_template = str(temp_dir / sanitize_filename(video_info.title))

    if _needs_js_runtime():
        # Use subprocess for JS runtime support
        cmd = _build_yt_dlp_command(url, [
            "-f", "bestaudio/best",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "192K",
            "-o", output_template,
        ])
        logger.debug(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio download failed: {result.stderr}")
    else:
        ydl_opts = _get_yt_dlp_base_opts()
        ydl_opts.update({
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": output_template,
        })
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    audio_path = Path(f"{output_template}.mp3")
    if audio_path.exists():
        logger.info(f"Audio downloaded to: {audio_path}")
        return audio_path

    raise RuntimeError("Audio download completed but file not found")


def extract_text_from_subtitles(file_path: Path) -> str:
    """Parse a subtitle file and extract plain text.

    Args:
        file_path: Path to the .vtt or .srt subtitle file.

    Returns:
        Plain text content of the subtitles.
    """
    logger.info(f"Extracting text from: {file_path}")

    content = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() == ".vtt":
        text = _parse_vtt(content)
    elif file_path.suffix.lower() == ".srt":
        text = _parse_srt(content)
    else:
        raise ValueError(f"Unsupported subtitle format: {file_path.suffix}")

    logger.info(f"Extracted {len(text)} characters of text")
    return text


def _parse_vtt(content: str) -> str:
    """Parse WebVTT subtitle content to plain text."""
    lines = []

    # Skip header and metadata
    in_cue = False
    for line in content.split("\n"):
        line = line.strip()

        # Skip empty lines and WebVTT header
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE"):
            in_cue = False
            continue

        # Skip timestamp lines
        if "-->" in line:
            in_cue = True
            continue

        # Skip cue identifiers (numeric or named)
        if not in_cue:
            continue

        # Remove VTT formatting tags
        text = re.sub(r"<[^>]+>", "", line)
        text = re.sub(r"\{[^}]+\}", "", text)  # Remove positioning

        if text:
            lines.append(text)

    # Remove consecutive duplicates (common in VTT)
    deduplicated = []
    for line in lines:
        if not deduplicated or line != deduplicated[-1]:
            deduplicated.append(line)

    return " ".join(deduplicated)


def _parse_srt(content: str) -> str:
    """Parse SRT subtitle content to plain text."""
    lines = []

    # SRT format: index, timestamp, text, blank line
    for block in content.split("\n\n"):
        block_lines = block.strip().split("\n")
        if len(block_lines) >= 3:
            # Skip index (first line) and timestamp (second line)
            text_lines = block_lines[2:]
            for line in text_lines:
                # Remove HTML-like tags
                text = re.sub(r"<[^>]+>", "", line)
                if text.strip():
                    lines.append(text.strip())

    return " ".join(lines)
