#!/usr/bin/env python3
"""CLI entry point for the YouTube Content Summarizer."""

import sys
from pathlib import Path

import click

from src.cache import PipelineCache, get_video_cache_id
from src.youtube import get_video_info, download_subtitles, download_audio, extract_text_from_subtitles
from src.transcriber import transcribe_audio
from src.summarizer import summarize
from src.utils import setup_logging, format_duration

# Default models per provider
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.1",
}


@click.command()
@click.argument("url")
@click.option(
    "--llm",
    type=click.Choice(["openai", "anthropic", "ollama"]),
    default="ollama",
    help="LLM provider to use for summarization.",
)
@click.option(
    "--model",
    default=None,
    help="Specific model name (e.g., gpt-4o, claude-sonnet-4-20250514, llama3.1).",
)
@click.option(
    "--transcriber",
    type=click.Choice(["whisperx", "gladia"]),
    default="whisperx",
    help="Transcription backend to use.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, prints to stdout.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed progress information.",
)
@click.option(
    "--force-audio",
    is_flag=True,
    help="Skip subtitle check and always use audio transcription.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching, run all steps fresh.",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear cached data for this video before running.",
)
def main(url: str, llm: str, model: str | None, transcriber: str, output: str | None, verbose: bool, force_audio: bool, no_cache: bool, clear_cache: bool):
    """Summarize a YouTube video.

    URL: The YouTube video URL to summarize.

    Examples:

        python main.py "https://youtube.com/watch?v=..."

        python main.py "URL" --llm ollama --model llama3.1

        python main.py "URL" --llm anthropic --model claude-sonnet-4-20250514

        python main.py "URL" --transcriber gladia --llm openai

        python main.py "URL" --output summary.md --verbose

        python main.py "URL" --clear-cache  # Re-run from scratch
    """
    logger = setup_logging(verbose)

    # Resolve model name
    model = model or DEFAULT_MODELS.get(llm, "")

    try:
        # Initialize cache
        video_id = get_video_cache_id(url)
        cache = PipelineCache(video_id)

        if clear_cache:
            cache.clear()
            click.echo("Cache cleared.")

        use_cache = not no_cache

        # Step 1: Get video info
        video_info = None
        if use_cache:
            video_info = cache.load_video_info()

        if video_info:
            click.echo(f"[cached] Video info loaded from cache")
        else:
            click.echo(f"Fetching video information...")
            video_info = get_video_info(url)
            if use_cache:
                cache.save_video_info(video_info)

        click.echo(f"Title: {video_info.title}")
        click.echo(f"Channel: {video_info.channel}")
        click.echo(f"Duration: {format_duration(video_info.duration)}")
        click.echo()

        # Step 2: Get content (subtitles or audio transcription)
        text_content = None

        if use_cache:
            text_content = cache.load_content()
            if text_content:
                click.echo(f"[cached] Content loaded from cache ({len(text_content):,} characters)")

        if text_content is None:
            if not force_audio and video_info.has_manual_subtitles:
                click.echo("Manual subtitles available. Downloading...")
                subtitle_path = download_subtitles(url, video_info)

                if subtitle_path:
                    text_content = extract_text_from_subtitles(subtitle_path)
                    click.echo(f"Extracted {len(text_content):,} characters from subtitles.")
                    if use_cache:
                        cache.save_content(text_content)

            if text_content is None:
                if force_audio:
                    click.echo("Force audio mode. Downloading audio...")
                else:
                    click.echo("No manual subtitles. Downloading audio for transcription...")

                # Check for cached audio
                audio_path = cache.get_audio_path() if use_cache else None

                if audio_path:
                    click.echo(f"[cached] Using cached audio file")
                else:
                    audio_path = download_audio(url, video_info)
                    click.echo(f"Audio downloaded.")
                    if use_cache:
                        audio_path = cache.save_audio(audio_path)

                click.echo(f"Starting transcription with {transcriber}...")
                text_content = transcribe_audio(audio_path, backend=transcriber)
                click.echo(f"Transcription complete: {len(text_content):,} characters.")
                if use_cache:
                    cache.save_content(text_content)

        click.echo()

        # Step 3: Generate summary
        summary = None

        if use_cache:
            summary = cache.load_summary(llm, model)
            if summary:
                click.echo(f"[cached] Summary loaded from cache ({llm}/{model})")

        if summary is None:
            click.echo(f"Generating summary using {llm}/{model}...")
            summary = summarize(text_content, video_info.title, provider=llm, model=model)
            if use_cache:
                cache.save_summary(summary, llm, model)

        # Step 4: Output
        if output:
            output_path = Path(output)
            output_path.write_text(summary, encoding="utf-8")
            click.echo(f"\nSummary saved to: {output_path}")
        else:
            click.echo("\n" + "=" * 60)
            click.echo("SUMMARY")
            click.echo("=" * 60 + "\n")
            click.echo(summary)

        click.echo("\nDone!")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
