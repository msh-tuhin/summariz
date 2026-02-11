"""Audio transcription via Gladia API or local WhisperX."""

import logging
import time
from pathlib import Path

import requests

from .utils import get_api_key

logger = logging.getLogger("content_summary")

GLADIA_UPLOAD_URL = "https://api.gladia.io/v2/upload"
GLADIA_TRANSCRIPTION_URL = "https://api.gladia.io/v2/transcription"


def transcribe_audio(
    file_path: Path,
    backend: str = "whisperx",
    poll_interval: float = 5.0,
) -> str:
    """Transcribe an audio file.

    Args:
        file_path: Path to the audio file.
        backend: Transcription backend ('whisperx' or 'gladia').
        poll_interval: Seconds to wait between status checks (Gladia only).

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If transcription fails.
        ValueError: If an unsupported backend is specified.
    """
    if backend == "whisperx":
        return transcribe_with_whisperx(file_path)
    elif backend == "gladia":
        return transcribe_with_gladia(file_path, poll_interval)
    else:
        raise ValueError(f"Unsupported transcription backend: {backend}. Use 'whisperx' or 'gladia'.")


def transcribe_with_whisperx(file_path: Path, model_size: str = "base") -> str:
    """Transcribe an audio file using local WhisperX.

    Args:
        file_path: Path to the audio file.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2').

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If transcription fails.
    """
    import torch
    import whisperx

    # Fix for PyTorch 2.6+ compatibility with pyannote models
    # See: https://github.com/m-bain/whisperX/issues/1016
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
    ])

    logger.info(f"Transcribing with WhisperX (model: {model_size})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.debug(f"Using device: {device}, compute_type: {compute_type}")

    # Load model
    model = whisperx.load_model(model_size, device, compute_type=compute_type)

    # Load audio
    audio = whisperx.load_audio(str(file_path))

    # Transcribe
    result = model.transcribe(audio, batch_size=16)

    # Extract text from segments
    segments = result.get("segments", [])
    if not segments:
        raise RuntimeError("WhisperX transcription returned no segments")

    transcript = " ".join(seg.get("text", "").strip() for seg in segments)

    logger.info(f"WhisperX transcription complete: {len(transcript)} characters")
    return transcript


def transcribe_with_gladia(file_path: Path, poll_interval: float = 5.0) -> str:
    """Transcribe an audio file using the Gladia API.

    Args:
        file_path: Path to the audio file.
        poll_interval: Seconds to wait between status checks.

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If transcription fails.
    """
    api_key = get_api_key("GLADIA_API_KEY")
    headers = {"x-gladia-key": api_key}

    # Step 1: Upload the audio file
    logger.info(f"Uploading audio file to Gladia: {file_path}")
    audio_url = _upload_audio(file_path, headers)

    # Step 2: Request transcription
    logger.info("Requesting Gladia transcription...")
    result_url = _request_transcription(audio_url, headers)

    # Step 3: Poll for completion
    logger.info("Waiting for Gladia transcription to complete...")
    transcript = _poll_for_result(result_url, headers, poll_interval)

    logger.info(f"Gladia transcription complete: {len(transcript)} characters")
    return transcript


def _upload_audio(file_path: Path, headers: dict) -> str:
    """Upload an audio file to Gladia and return the URL.

    Args:
        file_path: Path to the audio file.
        headers: API headers with authentication.

    Returns:
        URL of the uploaded audio.

    Raises:
        RuntimeError: If upload fails.
    """
    with open(file_path, "rb") as f:
        files = {"audio": (file_path.name, f, "audio/mpeg")}
        response = requests.post(GLADIA_UPLOAD_URL, headers=headers, files=files)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to upload audio: {response.status_code} - {response.text}")

    data = response.json()
    audio_url = data.get("audio_url")

    if not audio_url:
        raise RuntimeError(f"No audio URL in response: {data}")

    logger.debug(f"Audio uploaded: {audio_url}")
    return audio_url


def _request_transcription(audio_url: str, headers: dict) -> str:
    """Request transcription for an uploaded audio file.

    Args:
        audio_url: URL of the uploaded audio.
        headers: API headers with authentication.

    Returns:
        URL to poll for transcription results.

    Raises:
        RuntimeError: If the request fails.
    """
    payload = {
        "audio_url": audio_url,
        "detect_language": True,
    }

    headers_with_content = {**headers, "Content-Type": "application/json"}
    response = requests.post(
        GLADIA_TRANSCRIPTION_URL,
        headers=headers_with_content,
        json=payload,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(f"Failed to request transcription: {response.status_code} - {response.text}")

    data = response.json()
    result_url = data.get("result_url")

    if not result_url:
        raise RuntimeError(f"No result URL in response: {data}")

    logger.debug(f"Transcription requested: {result_url}")
    return result_url


def _poll_for_result(result_url: str, headers: dict, poll_interval: float) -> str:
    """Poll for transcription completion and return the transcript.

    Args:
        result_url: URL to poll for results.
        headers: API headers with authentication.
        poll_interval: Seconds to wait between polls.

    Returns:
        The transcribed text.

    Raises:
        RuntimeError: If transcription fails or times out.
    """
    max_attempts = 120  # 10 minutes with 5 second intervals

    for attempt in range(max_attempts):
        response = requests.get(result_url, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to get transcription status: {response.status_code} - {response.text}")

        data = response.json()
        status = data.get("status")

        if status == "done":
            # Extract full transcript
            result = data.get("result", {})
            transcription = result.get("transcription", {})
            full_transcript = transcription.get("full_transcript", "")

            if not full_transcript:
                # Try alternative structure
                utterances = transcription.get("utterances", [])
                if utterances:
                    full_transcript = " ".join(u.get("text", "") for u in utterances)

            if not full_transcript:
                raise RuntimeError(f"Transcription completed but no text found: {data}")

            return full_transcript

        elif status == "error":
            error_msg = data.get("error", "Unknown error")
            raise RuntimeError(f"Transcription failed: {error_msg}")

        elif status in ("queued", "processing"):
            logger.debug(f"Transcription status: {status} (attempt {attempt + 1})")
            time.sleep(poll_interval)

        else:
            logger.warning(f"Unknown transcription status: {status}")
            time.sleep(poll_interval)

    raise RuntimeError("Transcription timed out after maximum attempts")
