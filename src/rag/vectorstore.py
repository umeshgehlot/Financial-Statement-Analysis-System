# src/rag/vectorstore.py
"""
Vector store abstraction supporting ChromaDB (local) and Azure AI Search (cloud).
Provides a unified interface for indexing and retrieving financial documents.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import structlog
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from src.config import get_settings
from src.rag.embeddings import get_embedding_model

logger = structlog.get_logger(__name__)


class VectorStoreManager:
    """Manages vector store lifecycle — creation, indexing, and retrieval."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = get_embedding_model()
        self._store = None

    def get_or_create_store(self):
        """Return the configured vector store, creating if needed."""
        if self._store is not None:
            return self._store

        if self.settings.rag.use_azure_search:
            self._store = self._create_azure_search_store()
        else:
            self._store = self._create_chroma_store()

        logger.info(
            "vectorstore_ready",
            store_type=type(self._store).__name__,
        )
        return self._store

    def _create_chroma_store(self) -> Chroma:
        """Create a persistent ChromaDB store."""
        persist_dir = Path("./data/chroma_db")
        persist_dir.mkdir(parents=True, exist_ok=True)

        return Chroma(
            collection_name=self.settings.rag.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )

    def _create_azure_search_store(self):
        """Create an Azure AI Search vector store."""
        from langchain_community.vectorstores import AzureSearch

        return AzureSearch(
            azure_search_endpoint=self.settings.azure.search_endpoint,
            azure_search_key=self.settings.azure.search_api_key,
            index_name=self.settings.rag.collection_name,
            embedding_function=self.embeddings,
        )

    def index_documents(self, documents: list[Document]) -> list[str]:
        """Index a batch of documents and return their IDs."""
        store = self.get_or_create_store()

        ids = [str(uuid.uuid4()) for _ in documents]

        store.add_documents(documents=documents, ids=ids)

        logger.info(
            "documents_indexed",
            count=len(documents),
            store_type=type(store).__name__,
        )
        return ids

    def as_retriever(self, search_kwargs: dict | None = None):
        """Return the store as a LangChain retriever."""
        store = self.get_or_create_store()
        kwargs = search_kwargs or {"k": self.settings.rag.top_k_results}
        return store.as_retriever(search_kwargs=kwargs)

    def delete_collection(self):
        """Delete the entire collection (for testing/reset)."""
        store = self.get_or_create_store()
        if hasattr(store, "delete_collection"):
            store.delete_collection()
        self._store = None
        logger.info("collection_deleted")