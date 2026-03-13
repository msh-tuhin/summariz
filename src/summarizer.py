"""LLM-based summarization with multiple provider support."""

import logging
from abc import ABC, abstractmethod

import anthropic
import ollama
import openai

from .utils import get_api_key

logger = logging.getLogger("content_summary")

SYSTEM_PROMPT = """You are an expert content summarizer. Your task is to create clear, well-structured summaries of video transcripts.

Analyze the content and determine the most appropriate summary format based on the content type:

- For educational content with distinct points/lessons: Use an intro paragraph, numbered list with brief explanations, and conclusion
- For interviews/conversations: Highlight key topics discussed with the most interesting insights
- For tutorials/how-tos: Provide step-by-step overview with key takeaways
- For narrative/documentary content: Summarize the main story arc and key themes
- For news/analysis: Present main arguments and supporting points

Guidelines:
- Start with a brief overview that captures the essence of the content
- Use clear, concise language
- Preserve important details, quotes, or statistics
- End with key takeaways or a brief conclusion
- Match the tone to the content (formal for educational, conversational for casual content)
- Keep the summary comprehensive but not overly long (aim for 10-20% of original length)"""


def summarize(
    text: str,
    video_title: str,
    provider: str = "openai",
    model: str | None = None,
) -> str:
    """Generate a content-aware summary of the provided text.

    Args:
        text: The text content to summarize (transcript or subtitles).
        video_title: The title of the video for context.
        provider: LLM provider ('openai' or 'anthropic').
        model: Specific model name. Uses default for provider if not specified.

    Returns:
        The generated summary.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    logger.info(f"Generating summary using {provider}")

    summarizer = _get_summarizer(provider)
    summary = summarizer.summarize(text, video_title, model)

    logger.info(f"Summary generated: {len(summary)} characters")
    return summary


def _get_summarizer(provider: str) -> "BaseSummarizer":
    """Get the appropriate summarizer for the specified provider.

    Args:
        provider: The LLM provider name.

    Returns:
        A summarizer instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    providers = {
        "openai": OpenAISummarizer,
        "anthropic": AnthropicSummarizer,
        "ollama": OllamaSummarizer,
    }

    if provider not in providers:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {list(providers.keys())}")

    return providers[provider]()


class BaseSummarizer(ABC):
    """Base class for LLM summarizers."""

    default_model: str

    @abstractmethod
    def summarize(self, text: str, video_title: str, model: str | None = None) -> str:
        """Generate a summary of the text.

        Args:
            text: The text to summarize.
            video_title: The title of the video.
            model: Optional specific model to use.

        Returns:
            The generated summary.
        """
        pass

    def _build_user_prompt(self, text: str, video_title: str) -> str:
        """Build the user prompt for summarization.

        Args:
            text: The text to summarize.
            video_title: The title of the video.

        Returns:
            The formatted user prompt.
        """
        return f"""Please summarize the following video transcript.

Video Title: {video_title}

Transcript:
{text}

Provide a well-structured summary appropriate for this content type."""


class OpenAISummarizer(BaseSummarizer):
    """Summarizer using OpenAI's API."""

    default_model = "gpt-4o-mini"

    def __init__(self):
        api_key = get_api_key("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)

    def summarize(self, text: str, video_title: str, model: str | None = None) -> str:
        model = model or self.default_model
        logger.debug(f"Using OpenAI model: {model}")

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(text, video_title)},
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        return response.choices[0].message.content


class AnthropicSummarizer(BaseSummarizer):
    """Summarizer using Anthropic's API."""

    default_model = "claude-sonnet-4-20250514"

    def __init__(self):
        api_key = get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def summarize(self, text: str, video_title: str, model: str | None = None) -> str:
        model = model or self.default_model
        logger.debug(f"Using Anthropic model: {model}")

        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": self._build_user_prompt(text, video_title)},
            ],
        )

        return response.content[0].text


class OllamaSummarizer(BaseSummarizer):
    """Summarizer using Ollama with cloud fallback to local."""

    default_cloud_model = "deepseek-v3.2"
    default_local_model = "llama3.1"
    default_model = default_cloud_model  # For compatibility

    def __init__(self):
        # Check for cloud API key
        try:
            self.api_key = get_api_key("OLLAMA_API_KEY")
            self.cloud_available = True
            logger.debug("Ollama cloud API key found")
        except (ValueError, KeyError):
            self.api_key = None
            self.cloud_available = False
            logger.debug("No Ollama cloud API key, using local only")

    def summarize(self, text: str, video_title: str, model: str | None = None) -> str:
        # Combine system prompt and user prompt for Ollama
        full_prompt = f"""{SYSTEM_PROMPT}

{self._build_user_prompt(text, video_title)}"""

        messages = [{"role": "user", "content": full_prompt}]

        # Try cloud first if available
        if self.cloud_available:
            cloud_model = model or self.default_cloud_model
            try:
                return self._call_cloud(cloud_model, messages)
            except Exception as e:
                logger.warning(f"Ollama cloud failed: {e}. Falling back to local.")

        # Fall back to local
        local_model = model or self.default_local_model
        return self._call_local(local_model, messages)

    def _call_cloud(self, model: str, messages: list) -> str:
        """Call Ollama cloud API."""
        logger.debug(f"Using Ollama cloud model: {model}")

        client = ollama.Client(
            host="https://api.ollama.com",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        response = client.chat(model=model, messages=messages)
        return self._extract_response(response)

    def _call_local(self, model: str, messages: list) -> str:
        """Call local Ollama instance."""
        logger.debug(f"Using Ollama local model: {model}")

        response = ollama.chat(model=model, messages=messages)
        return self._extract_response(response)

    def _extract_response(self, response) -> str:
        """Extract text from Ollama response, handling reasoning models."""
        message = response.get("message", {})

        # For dict-style response
        if isinstance(message, dict):
            content = message.get("content", "")
            thinking = message.get("thinking", "")
        else:
            # For object-style response (newer ollama library)
            content = getattr(message, "content", "") or ""
            thinking = getattr(message, "thinking", "") or ""

        # Prefer content, fall back to thinking for reasoning models
        result = content.strip() if content.strip() else thinking.strip()

        if not result:
            logger.warning(f"Empty response from Ollama: {response}")

        return result
