# src/main.py
"""
Application entry point.
Provides CLI and programmatic startup for the Financial Statement Analyzer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import structlog
import uvicorn

from src.config import get_settings

# Ensure src is on the Python path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = structlog.get_logger(__name__)


def configure_logging():
    """Set up structured logging for the application."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.environment == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.stdlib._NAME_TO_LEVEL.get(settings.log_level.lower(), 20)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def run_server():
    """Start the FastAPI server."""
    settings = get_settings()
    configure_logging()

    logger.info(
        "starting_server",
        host=settings.api_host,
        port=settings.api_port,
        environment=settings.environment.value,
        model=settings.openai.model,
    )

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment == "development",
        workers=1 if settings.environment == "development" else 4,
        log_level=settings.log_level.lower(),
        access_log=settings.environment == "development",
    )


if __name__ == "__main__":
    run_server()