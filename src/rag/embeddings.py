# src/rag/embeddings.py
"""
Embedding model configuration with support for OpenAI and Azure OpenAI.
"""
from __future__ import annotations

from functools import lru_cache

import structlog
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache
def get_embedding_model():
    """Return the configured embedding model."""
    settings = get_settings()

    if settings.azure.search_endpoint:
        logger.info("using_azure_openai_embeddings")
        return AzureOpenAIEmbeddings(
            model=settings.openai.embedding_model,
            azure_deployment="text-embedding-3-large",
            openai_api_version="2024-02-01",
        )

    logger.info(
        "using_openai_embeddings",
        model=settings.openai.embedding_model,
    )
    return OpenAIEmbeddings(model=settings.openai.embedding_model)