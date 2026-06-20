# src/rag/retriever.py
"""
Hybrid retriever combining semantic search, metadata filtering,
and optional BM25 for maximum recall on financial queries.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever

from src.config import get_settings
from src.rag.vectorstore import VectorStoreManager

logger = structlog.get_logger(__name__)


class FinancialRetriever(BaseRetriever):
    """
    Domain-aware retriever for financial documents.

    Combines:
    - Semantic search (via vector store)
    - BM25 keyword search (exact amount/date matching)
    - Metadata filtering (by account, date range, category)
    """

    vector_store_manager: VectorStoreManager
    bm25_retriever: BM25Retriever | None = None
    top_k: int = 5
    use_ensemble: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_retrievers(self):
        """Build the ensemble of retrievers."""
        vector_retriever = self.vector_store_manager.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        if self.use_ensemble and self.bm25_retriever:
            return [vector_retriever, self.bm25_retriever], [0.6, 0.4]

        return [vector_retriever], [1.0]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        LangChain BaseRetriever requires this method name.
        Retrieves relevant documents for a financial query.
        """
        retrievers, weights = self._get_retrievers()

        if len(retrievers) > 1:
            ensemble = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
            )
            docs = ensemble.invoke(query)
        else:
            docs = retrievers[0].invoke(query)

        logger.info(
            "retrieval_complete",
            query=query[:100],
            results=len(docs),
        )
        return docs[:self.top_k]

    def retrieve_with_filters(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Public method for filtered retrieval."""
        docs = self._get_relevant_documents(
            query,
            run_manager=CallbackManagerForRetrieverRun(
                run_id=None, handlers=[], inheritable_handlers=[]
            ),
        )

        if filters:
            docs = self._apply_filters(docs, filters)

        logger.info(
            "filtered_retrieval_complete",
            query=query[:100],
            results=len(docs),
            filters=filters,
        )
        return docs

    def _apply_filters(
        self, docs: list[Document], filters: dict
    ) -> list[Document]:
        """Apply metadata filters to retrieved documents."""
        filtered = docs

        if "date_start" in filters:
            start = filters["date_start"]
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            filtered = [
                d for d in filtered
                if d.metadata.get("date", "") >= start.isoformat()
            ]

        if "date_end" in filters:
            end = filters["date_end"]
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            filtered = [
                d for d in filtered
                if d.metadata.get("date", "") <= end.isoformat()
            ]

        if "category" in filters:
            cat = filters["category"].lower()
            filtered = [
                d for d in filtered
                if d.metadata.get("category", "").lower() == cat
            ]

        if "doc_type" in filters:
            filtered = [
                d for d in filtered
                if d.metadata.get("doc_type") == filters["doc_type"]
            ]

        return filtered

    def build_bm25_index(self, documents: list[Document]):
        """Build BM25 index from a corpus of documents."""
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = self.top_k
        logger.info("bm25_index_built", doc_count=len(documents))


class QueryRewriter:
    """
    Rewrites user queries for better retrieval.
    Transforms natural language financial questions into more precise search queries.
    """

    REWRITE_PROMPT = """You are a financial query optimizer. Given a user question about bank statements
or financial transactions, rewrite it to maximize retrieval relevance. Include specific financial
terms, transaction types, and time references.

Original question: {question}

Rewritten query (respond with only the rewritten query, nothing else):"""

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(self.REWRITE_PROMPT)
        chain = prompt | self.llm
        result = chain.invoke({"question": question})
        rewritten = result.content.strip()
        logger.info(
            "query_rewritten",
            original=question[:50],
            rewritten=rewritten[:50],
        )
        return rewritten