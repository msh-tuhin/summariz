"""Shared pipeline logic for CLI and web service."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cache import PipelineCache, get_video_cache_id
from .youtube import (
    get_video_info,
    download_subtitles,
    download_audio,
    extract_text_from_subtitles,
    VideoInfo,
)
from .transcriber import transcribe_audio
from .summarizer import summarize
from .utils import save_summary_to_pdf

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "deepseek-v3.2",  # Cloud model, falls back to llama3.1 locally
}


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    url: str
    llm: str = "ollama"
    model: str | None = None
    transcriber: str = "whisperx"
    output_dir: Path | None = None
    force_audio: bool = False
    use_cache: bool = True
    clear_cache: bool = False


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    video_info: VideoInfo
    summary: str
    pdf_path: Path


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the full summarization pipeline.

    Args:
        config: Pipeline configuration.
        progress_callback: Optional callback for progress updates.

    Returns:
        PipelineResult with video info, summary, and PDF path.

    Raises:
        ValueError: For invalid input.
        RuntimeError: For pipeline failures.
    """

    def update_progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Resolve model
    model = config.model or DEFAULT_MODELS.get(config.llm, "")

    # Initialize cache
    video_id = get_video_cache_id(config.url)
    cache = PipelineCache(video_id)

    if config.clear_cache:
        cache.clear()

    # Step 1: Get video info
    update_progress("Fetching video information...")
    video_info = None
    if config.use_cache:
        video_info = cache.load_video_info()

    if not video_info:
        video_info = get_video_info(config.url)
        if config.use_cache:
            cache.save_video_info(video_info)

    # Step 2: Get content
    update_progress("Getting video content...")
    text_content = None

    if config.use_cache:
        text_content = cache.load_content()

    if text_content is None:
        if not config.force_audio and video_info.has_manual_subtitles:
            update_progress("Downloading subtitles...")
            subtitle_path = download_subtitles(config.url, video_info)
            if subtitle_path:
                text_content = extract_text_from_subtitles(subtitle_path)
                if config.use_cache:
                    cache.save_content(text_content)

        if text_content is None:
            update_progress("Downloading audio...")
            audio_path = cache.get_audio_path() if config.use_cache else None

            if not audio_path:
                audio_path = download_audio(config.url, video_info)
                if config.use_cache:
                    audio_path = cache.save_audio(audio_path)

            update_progress(f"Transcribing with {config.transcriber}...")
            text_content = transcribe_audio(audio_path, backend=config.transcriber)
            if config.use_cache:
                cache.save_content(text_content)

    # Step 3: Generate summary
    update_progress(f"Generating summary with {config.llm}/{model}...")
    summary = None

    if config.use_cache:
        summary = cache.load_summary(config.llm, model)

    if summary is None:
        summary = summarize(
            text_content, video_info.title, provider=config.llm, model=model
        )
        if config.use_cache:
            cache.save_summary(summary, config.llm, model)

    # Step 4: Save PDF
    update_progress("Generating PDF...")
    pdf_path = save_summary_to_pdf(
        summary, video_info.id, video_info.title, config.output_dir
    )

    update_progress("Complete")

    return PipelineResult(
        video_info=video_info,
        summary=summary,
        pdf_path=pdf_path,
    )
