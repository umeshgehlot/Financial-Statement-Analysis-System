# src/document_processor/chunker.py
"""
Financial-aware text chunking that preserves transaction boundaries
and maintains semantic coherence for downstream RAG retrieval.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_transactions_per_chunk: int = 20  # ← fixed spelling


class FinancialChunker:
    """
    Domain-aware chunker for financial documents.

    Rather than blindly splitting text, this chunker:
    1. Preserves individual transaction boundaries (never splits mid-transaction)
    2. Groups transactions by date or time period when possible
    3. Includes metadata context in each chunk for better retrieval
    4. Creates summary chunks for high-level queries
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " | ", " ", ""],
            length_function=len,
        )

    def chunk_documents(
        self, documents: list[dict[str, any]]
    ) -> list[Document]:
        """
        Chunk a list of documents (from ParsedStatement.to_documents())
        into LangChain Document objects suitable for vector indexing.
        """
        all_chunks: list[Document] = []
        transaction_buffer: list[dict] = []
        summary_docs: list[dict] = []

        for doc in documents:
            if doc["metadata"].get("doc_type") == "statement_summary":
                summary_docs.append(doc)
            elif doc["metadata"].get("doc_type") == "transaction":
                transaction_buffer.append(doc)

        # Summary chunks — keep these intact
        for doc in summary_docs:
            chunk_id = self._generate_chunk_id(doc["page_content"])
            all_chunks.append(Document(
                page_content=doc["page_content"],
                metadata={
                    **doc["metadata"],
                    "chunk_id": chunk_id,
                    "chunk_type": "summary",
                },
            ))

        # Transaction chunks — group into manageable batches
        if transaction_buffer:
            transaction_chunks = self._group_transactions(transaction_buffer)
            all_chunks.extend(transaction_chunks)

        logger.info(
            "chunking_complete",
            input_docs=len(documents),
            output_chunks=len(all_chunks),
        )
        return all_chunks

    def _group_transactions(
        self, transactions: list[dict]
    ) -> list[Document]:
        """Group transactions into chunks while preserving boundaries."""
        chunks: list[Document] = []
        current_batch: list[dict] = []
        current_text_length = 0

        for txn in transactions:
            txn_text = txn["page_content"]

            # If adding this transaction exceeds chunk size, flush the batch
            if (
                current_text_length + len(txn_text) > self.config.chunk_size
                and current_batch
            ):
                chunks.append(self._create_transaction_chunk(current_batch))
                current_batch = []
                current_text_length = 0

            current_batch.append(txn)
            current_text_length += len(txn_text) + 1  # +1 for separator

            # Hard limit on transactions per chunk
            if len(current_batch) >= self.config.max_transactions_per_chunk:  # ← fixed
                chunks.append(self._create_transaction_chunk(current_batch))
                current_batch = []
                current_text_length = 0

        if current_batch:
            chunks.append(self._create_transaction_chunk(current_batch))

        return chunks

    def _create_transaction_chunk(self, batch: list[dict]) -> Document:
        """Create a single Document from a batch of transactions."""
        texts = [txn["page_content"] for txn in batch]
        combined_text = "\n".join(texts)

        # Include date range in content for better retrieval
        dates = [
            txn["metadata"].get("date", "")
            for txn in batch
            if txn["metadata"].get("date")
        ]
        date_range = ""
        if dates:
            date_range = f"Transactions from {dates[0]} to {dates[-1]}\n"

        page_content = date_range + combined_text
        chunk_id = self._generate_chunk_id(page_content)

        # Aggregate metadata
        amounts = []
        for txn in batch:
            try:
                amounts.append(float(txn["metadata"].get("amount", 0)))
            except (ValueError, TypeError):
                pass

        return Document(
            page_content=page_content,
            metadata={
                "source": batch[0]["metadata"].get("source", ""),
                "doc_type": "transaction_batch",
                "chunk_id": chunk_id,
                "chunk_type": "transactions",
                "transaction_count": len(batch),
                "date_start": dates[0] if dates else None,
                "date_end": dates[-1] if dates else None,
                "total_amount": str(sum(amounts)) if amounts else "0",
            },
        )

    @staticmethod
    def _generate_chunk_id(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]