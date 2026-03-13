#!/usr/bin/env python3
"""CLI entry point for the YouTube Content Summarizer."""

import sys
from pathlib import Path

import click

from src.pipeline import PipelineConfig, run_pipeline
from src.utils import setup_logging, format_duration


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
    help="Output directory for PDF. Defaults to current directory.",
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
def main(
    url: str,
    llm: str,
    model: str | None,
    transcriber: str,
    output: str | None,
    verbose: bool,
    force_audio: bool,
    no_cache: bool,
    clear_cache: bool,
):
    """Summarize a YouTube video.

    URL: The YouTube video URL to summarize.

    Examples:

        python main.py "https://youtube.com/watch?v=..."

        python main.py "URL" --llm ollama --model llama3.1

        python main.py "URL" --llm anthropic --model claude-sonnet-4-20250514

        python main.py "URL" --transcriber gladia --llm openai

        python main.py "URL" --output ./summaries --verbose

        python main.py "URL" --clear-cache  # Re-run from scratch
    """
    logger = setup_logging(verbose)

    def print_progress(msg: str):
        click.echo(msg)

    try:
        config = PipelineConfig(
            url=url,
            llm=llm,
            model=model,
            transcriber=transcriber,
            output_dir=Path(output) if output else None,
            force_audio=force_audio,
            use_cache=not no_cache,
            clear_cache=clear_cache,
        )

        result = run_pipeline(config, progress_callback=print_progress)

        # Print video info
        click.echo(f"\nTitle: {result.video_info.title}")
        click.echo(f"Channel: {result.video_info.channel}")
        click.echo(f"Duration: {format_duration(result.video_info.duration)}")

        # Output summary
        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60 + "\n")
        click.echo(result.summary)
        click.echo(f"\nSummary saved to: {result.pdf_path}")
        click.echo("\nDone!")

    except (ValueError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
