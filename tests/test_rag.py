# tests/test_rag.py
"""Tests for the RAG pipeline components."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.document_processor.chunker import ChunkingConfig, FinancialChunker
from src.document_processor.normalizer import TransactionNormalizer
from src.document_processor.parser import Transaction


class TestFinancialChunker:
    def test_chunker_preserves_transaction_boundaries(self):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = FinancialChunker(config)

        documents = [
            {
                "page_content": "Bank Statement Summary\nTotal: $1000",
                "metadata": {
                    "source": "test.pdf",
                    "doc_type": "statement_summary",
                },
            },
            *[
                {
                    "page_content": (
                        f"Date: 2024-01-{i:02d} | Description: Test Txn {i} | "
                        f"Amount: ${i * 10}.00"
                    ),
                    "metadata": {
                        "source": "test.pdf",
                        "doc_type": "transaction",
                        "date": f"2024-01-{i:02d}",
                        "amount": str(i * 10),
                    },
                }
                for i in range(1, 21)
            ],
        ]

        chunks = chunker.chunk_documents(documents)

        assert len(chunks) >= 2  # At least summary + transaction batch
        summary_chunks = [
            c for c in chunks if c.metadata.get("chunk_type") == "summary"
        ]
        assert len(summary_chunks) == 1

    def test_chunker_empty_input(self):
        chunker = FinancialChunker()
        chunks = chunker.chunk_documents([])
        assert chunks == []

    def test_chunk_metadata_preserved(self):
        config = ChunkingConfig(chunk_size=2000)
        chunker = FinancialChunker(config)

        documents = [
            {
                "page_content": "Date: 2024-01-15 | Test",
                "metadata": {
                    "source": "test.pdf",
                    "doc_type": "transaction",
                    "date": "2024-01-15",
                    "amount": "100.00",
                },
            },
        ]

        chunks = chunker.chunk_documents(documents)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.pdf"

    def test_summary_chunk_intact(self):
        """Summary chunks should never be split."""
        config = ChunkingConfig(chunk_size=200)
        chunker = FinancialChunker(config)

        long_summary = "Bank Statement Summary\n" + "\n".join(
            [f"Line {i}: {'x' * 80}" for i in range(20)]
        )
        documents = [
            {
                "page_content": long_summary,
                "metadata": {
                    "source": "test.pdf",
                    "doc_type": "statement_summary",
                },
            },
        ]

        chunks = chunker.chunk_documents(documents)
        summary_chunks = [
            c for c in chunks if c.metadata.get("chunk_type") == "summary"
        ]
        # Summary should stay as one chunk regardless of size
        assert len(summary_chunks) == 1


class TestTransactionNormalizer:
    def test_categorize_payroll(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 5),
            description="DIRECT DEPOSIT - ACME CORP PAYROLL",
            amount=Decimal("5200.00"),
        )
        result = normalizer.normalize([txn])
        assert result[0].category == "Income/Salary"

    def test_categorize_grocery(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 7),
            description="WHOLE FOODS MARKET #10423",
            amount=Decimal("-127.43"),
        )
        result = normalizer.normalize([txn])
        assert result[0].category == "Food/Groceries"

    def test_categorize_rent(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 10),
            description="MONTHLY RENT PAYMENT",
            amount=Decimal("-2100.00"),
        )
        result = normalizer.normalize([txn])
        assert result[0].category == "Housing/Rent"

    def test_categorize_transportation(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 18),
            description="UBER TRIP HELP.UBER.COM",
            amount=Decimal("-24.50"),
        )
        result = normalizer.normalize([txn])
        assert result[0].category == "Transportation"

    def test_uncategorized_fallback(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 20),
            description="RANDOM MERCHANT XYZ123",
            amount=Decimal("-15.00"),
        )
        result = normalizer.normalize([txn])
        assert result[0].category == "Uncategorized"

    def test_amount_normalization(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 5),
            description="Test",
            amount=Decimal("100.1"),
        )
        result = normalizer.normalize([txn])
        assert result[0].amount == Decimal("100.10")

    def test_description_cleaned(self):
        normalizer = TransactionNormalizer()
        txn = Transaction(
            date=datetime(2024, 1, 5),
            description="  WHOLE   FOODS  #12345  ",
            amount=Decimal("-50.00"),
        )
        result = normalizer.normalize([txn])
        assert "  " not in result[0].description