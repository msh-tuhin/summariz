"""FastAPI web application for YouTube Content Summarizer."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.utils import setup_logging

# Setup logging
setup_logging(verbose=False)

app = FastAPI(
    title="YouTube Content Summarizer",
    description="API for summarizing YouTube videos",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
