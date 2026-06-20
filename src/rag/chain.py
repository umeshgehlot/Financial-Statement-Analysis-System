# src/rag/chain.py
"""
RAG chain for financial document Q&A using LangChain.
Implements retrieval with context-aware prompting and citation tracking.
"""
from __future__ import annotations

import structlog
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.rag.retriever import FinancialRetriever, QueryRewriter

logger = structlog.get_logger(__name__)

FINANCIAL_QA_SYSTEM_PROMPT = """You are a senior financial analyst assistant specializing in
bank statement analysis. You have access to transaction data extracted from bank statements.

Your responsibilities:
1. Answer questions about transactions, balances, spending patterns, and financial activity accurately.
2. Always cite specific transactions, dates, and amounts from the provided context.
3. When analyzing spending, break down by category, time period, and merchant.
4. Flag unusual or potentially suspicious transactions when asked.
5. Provide precise numerical summaries — never estimate when exact data is available.
6. If the context does not contain enough information to answer, say so clearly.

IMPORTANT RULES:
- Base ALL answers strictly on the provided context. Never fabricate transactions or amounts.
- When displaying amounts, always include the dollar sign and two decimal places.
- Distinguish between credits (money in) and debits (money out).
- For date ranges, be explicit about which dates the data covers.

Context from retrieved documents:
{context}

Additional metadata:
{metadata}"""

FINANCIAL_QA_USER_PROMPT = """Question: {question}

Please provide a detailed, well-structured answer based on the financial data in the context above.
If performing calculations, show your work."""


class FinancialRAGChain:
    """
    End-to-end RAG chain for financial statement Q&A.

    Architecture:
    1. Query rewriting (optional) for better retrieval
    2. Hybrid retrieval (semantic + BM25)
    3. Context formatting with metadata enrichment
    4. LLM generation with financial analyst prompt
    5. Source citation tracking
    """

    def __init__(
        self,
        retriever: FinancialRetriever,
        model_name: str | None = None,
        enable_query_rewriting: bool = True,
    ):
        settings = get_settings()
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=model_name or settings.openai.model,
            temperature=settings.openai.temperature,
            max_tokens=settings.openai.max_tokens,
        )
        self.query_rewriter = (
            QueryRewriter(self.llm) if enable_query_rewriting else None
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", FINANCIAL_QA_SYSTEM_PROMPT),
            ("human", FINANCIAL_QA_USER_PROMPT),
        ])

        self._chain = self._build_chain()

    def _format_docs(self, docs: list[Document]) -> str:
        """Format retrieved documents into a structured context string."""
        if not docs:
            return "No relevant documents found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("doc_type", "unknown")
            formatted.append(
                f"[Document {i}] (Source: {source}, Type: {doc_type})\n"
                f"{doc.page_content}\n"
            )
        return "\n---\n".join(formatted)

    def _extract_metadata(self, docs: list[Document]) -> str:
        """Extract and summarize metadata from retrieved documents."""
        sources = set()
        dates = set()
        categories = set()

        for doc in docs:
            meta = doc.metadata
            if meta.get("source"):
                sources.add(meta["source"])
            if meta.get("date"):
                dates.add(meta["date"][:10])
            if meta.get("category"):
                categories.add(meta["category"])

        parts = []
        if sources:
            parts.append(f"Sources: {', '.join(sources)}")
        if dates:
            sorted_dates = sorted(dates)
            parts.append(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        if categories:
            parts.append(f"Categories: {', '.join(sorted(categories))}")

        return "\n".join(parts) if parts else "No additional metadata available."

    def _build_chain(self):
        """Build the LangChain RAG pipeline."""
        return (
            RunnableParallel({
                "context": lambda x: self._format_docs(
                    self.retriever.invoke(x["question"])
                ),
                "metadata": lambda x: self._extract_metadata(
                    self.retriever.invoke(x["question"])
                ),
                "question": lambda x: x["question"],
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(
        self,
        question: str,
        filters: dict | None = None,
    ) -> dict:
        """
        Answer a financial question using the RAG pipeline.

        Returns:
            dict with 'answer', 'sources', and 'retrieved_docs' keys.
        """
        # Optionally rewrite query for better retrieval
        effective_question = question
        if self.query_rewriter:
            try:
                effective_question = self.query_rewriter.rewrite(question)
            except Exception as e:
                logger.warning("query_rewrite_failed", error=str(e))

        # Retrieve relevant documents
        if filters:
            retrieved_docs = self.retriever.retrieve_with_filters(
                effective_question, filters=filters
            )
        else:
            retrieved_docs = self.retriever.invoke(effective_question)

        # Generate answer
        context = self._format_docs(retrieved_docs)
        metadata = self._extract_metadata(retrieved_docs)

        answer = self._chain.invoke({"question": question})

        # Track sources
        sources = list({
            doc.metadata.get("source", "unknown")
            for doc in retrieved_docs
        })

        logger.info(
            "rag_query_complete",
            question=question[:80],
            docs_retrieved=len(retrieved_docs),
            sources=sources,
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "metadata": {
                "context_docs": len(retrieved_docs),
                "filters_applied": filters,
            },
        }

    def stream(self, question: str, filters: dict | None = None):
        """Stream the answer token by token."""
        effective_question = question
        if self.query_rewriter:
            try:
                effective_question = self.query_rewriter.rewrite(question)
            except Exception:
                pass

        if filters:
            retrieved_docs = self.retriever.retrieve_with_filters(
                effective_question, filters=filters
            )
        else:
            retrieved_docs = self.retriever.invoke(effective_question)

        context = self._format_docs(retrieved_docs)
        metadata = self._extract_metadata(retrieved_docs)

        chain = self.prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({
            "question": question,
            "context": context,
            "metadata": metadata,
        }):
            yield chunk