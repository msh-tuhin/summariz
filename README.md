# YouTube Content Summarizer

A tool that summarizes YouTube videos using AI. It extracts content from manual subtitles or audio transcription, then generates a structured summary using your choice of LLM provider.

Available as both a **CLI tool** and a **REST API** with background job processing.

## Features

- Extracts text from manual subtitles when available
- Falls back to audio transcription using WhisperX or Gladia
- Supports multiple LLM providers: OpenAI, Anthropic, and Ollama (local)
- Caches intermediate results to avoid redundant processing
- Outputs summary to both stdout and PDF file

## Requirements

- Python 3.12
- FFmpeg (for audio processing)
- CUDA-compatible GPU (recommended for WhisperX)

## Installation

1. Clone the repository and create a virtual environment:
```bash
git clone <repository-url>
cd content_summary
py -3.12 -m venv venv312
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **Important: Reinstall PyTorch with CUDA support**

WhisperX installation replaces PyTorch with the CPU-only version. You must reinstall the CUDA version for GPU acceleration:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

Replace `cu121` with your CUDA version if different (e.g., `cu118` for CUDA 11.8).

4. Set up API keys (optional, only needed for cloud LLM providers):

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GLADIA_API_KEY=your-gladia-key
```

## Usage

Basic usage (uses local Ollama with llama3.1):
```bash
python main.py "https://youtube.com/watch?v=VIDEO_ID"
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--llm` | LLM provider (`openai`, `anthropic`, `ollama`) | `ollama` |
| `--model` | Specific model name | Provider default |
| `--transcriber` | Transcription backend (`whisperx`, `gladia`) | `whisperx` |
| `--output`, `-o` | Output directory for PDF | Current directory |
| `--verbose`, `-v` | Show detailed progress | Off |
| `--force-audio` | Skip subtitles, always use audio transcription | Off |
| `--no-cache` | Disable caching, run all steps fresh | Off |
| `--clear-cache` | Clear cached data for this video before running | Off |

### Default Models

| Provider | Default Model |
|----------|---------------|
| OpenAI | gpt-4o-mini |
| Anthropic | claude-sonnet-4-20250514 |
| Ollama | llama3.1 |

### Examples

Using OpenAI:
```bash
python main.py "https://youtube.com/watch?v=..." --llm openai
```

Using Anthropic with a specific model:
```bash
python main.py "https://youtube.com/watch?v=..." --llm anthropic --model claude-sonnet-4-20250514
```

Using Gladia for transcription with verbose output:
```bash
python main.py "https://youtube.com/watch?v=..." --transcriber gladia --verbose
```

Save PDF to a specific directory:
```bash
python main.py "https://youtube.com/watch?v=..." --output ./summaries
```

Force audio transcription (skip subtitles):
```bash
python main.py "https://youtube.com/watch?v=..." --force-audio
```

## Web API

The project includes a FastAPI-based REST API for integration with web applications.

### Running the Server

```bash
# Development
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/summarize` | Submit a new summarization job |
| GET | `/api/v1/jobs/{job_id}` | Get job status and progress |
| GET | `/api/v1/jobs/{job_id}/pdf` | Download the generated PDF |
| GET | `/api/v1/jobs/{job_id}/summary` | Get the summary text |
| GET | `/api/v1/health` | Health check |

### Example Usage

**Submit a job:**
```bash
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtube.com/watch?v=VIDEO_ID", "llm": "ollama"}'
```

Response:
```json
{
  "job_id": "abc123",
  "status": "pending",
  "video_id": "VIDEO_ID",
  "status_url": "/api/v1/jobs/abc123",
  "message": "Job submitted successfully"
}
```

**Check job status:**
```bash
curl http://localhost:8000/api/v1/jobs/abc123
```

**Request body options for `/api/v1/summarize`:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | required | YouTube video URL |
| `llm` | string | `"ollama"` | LLM provider: `openai`, `anthropic`, `ollama` |
| `model` | string | null | Specific model name |
| `transcriber` | string | `"whisperx"` | Transcription backend: `whisperx`, `gladia` |
| `force_audio` | boolean | `false` | Skip subtitles, use audio |
| `no_cache` | boolean | `false` | Disable caching |

## Output

The tool outputs:
1. Summary printed to stdout
2. PDF file saved as `youtube_{video_id}.pdf` in the output directory

## Caching

The tool caches:
- Video metadata
- Extracted text content
- Downloaded audio files
- Generated summaries (per LLM/model combination)

Use `--no-cache` to bypass caching or `--clear-cache` to delete cached data for a video.

## Troubleshooting

### WhisperX is slow or using CPU
Ensure PyTorch CUDA is installed correctly:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If this prints `False`, reinstall PyTorch with CUDA support (see installation step 3).

### FFmpeg not found
Install FFmpeg and ensure it's in your system PATH:
- Windows: `winget install ffmpeg` or download from https://ffmpeg.org/
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
