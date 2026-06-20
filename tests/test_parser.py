# tests/test_parser.py
"""Tests for the document parsing pipeline."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.document_processor.parser import (
    CSVParser,
    DocumentParserFactory,
    DocumentFormat,
    PDFParser,
    ParsedStatement,
    Transaction,
)


class TestDocumentParserFactory:
    def test_detect_pdf_format(self, tmp_path):
        pdf_file = tmp_path / "statement.pdf"
        pdf_file.touch()
        assert DocumentParserFactory.detect_format(pdf_file) == DocumentFormat.PDF

    def test_detect_csv_format(self, tmp_path):
        csv_file = tmp_path / "statement.csv"
        csv_file.touch()
        assert DocumentParserFactory.detect_format(csv_file) == DocumentFormat.CSV

    def test_detect_xlsx_format(self, tmp_path):
        xlsx_file = tmp_path / "statement.xlsx"
        xlsx_file.touch()
        assert DocumentParserFactory.detect_format(xlsx_file) == DocumentFormat.XLSX

    def test_detect_unknown_format(self, tmp_path):
        unknown_file = tmp_path / "statement.xyz"
        unknown_file.touch()
        assert DocumentParserFactory.detect_format(unknown_file) == DocumentFormat.UNKNOWN

    def test_parse_unsupported_raises(self, tmp_path):
        file = tmp_path / "data.xyz"
        file.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            DocumentParserFactory.parse(file)


class TestTransaction:
    def test_to_dict(self):
        txn = Transaction(
            date="2024-01-15T00:00:00",
            description="Test Transaction",
            amount=Decimal("-50.00"),
            balance=Decimal("1000.00"),
            transaction_type="debit",
            category="Shopping",
        )
        d = txn.to_dict()
        assert d["amount"] == "-50.00"
        assert d["transaction_type"] == "debit"
        assert d["category"] == "Shopping"

    def test_to_text(self):
        txn = Transaction(
            date="2024-01-15T00:00:00",
            description="WHOLE FOODS",
            amount=Decimal("-127.43"),
            balance=Decimal("8322.57"),
            transaction_type="debit",
        )
        text = txn.to_text()
        assert "WHOLE FOODS" in text
        assert "$127.43" in text


class TestParsedStatement:
    def test_total_credits(self, sample_parsed_statement):
        total = sample_parsed_statement.total_credits
        assert total == Decimal("5700.00")

    def test_total_debits(self, sample_parsed_statement):
        total = sample_parsed_statement.total_debits
        assert total == Decimal("-2493.67")

    def test_transaction_count(self, sample_parsed_statement):
        assert sample_parsed_statement.transaction_count == 8

    def test_to_documents(self, sample_parsed_statement):
        docs = sample_parsed_statement.to_documents()
        assert len(docs) == 9  # 1 summary + 8 transactions
        assert docs[0]["metadata"]["doc_type"] == "statement_summary"
        assert all(
            d["metadata"]["doc_type"] == "transaction" for d in docs[1:]
        )


class TestPDFParser:
    def test_detect_bank(self):
        parser = PDFParser()
        assert parser._detect_bank("Welcome to Chase Bank statement") == "Chase"
        assert parser._detect_bank("Bank of America account summary") == "Bank of America"
        assert parser._detect_bank("Some random text") is None

    def test_parse_date(self):
        parser = PDFParser()
        assert parser._parse_date("01/15/2024") is not None
        assert parser._parse_date("2024-01-15") is not None
        assert parser._parse_date("Jan 15, 2024") is not None
        assert parser._parse_date("invalid date") is None