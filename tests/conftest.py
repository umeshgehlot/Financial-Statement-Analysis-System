# tests/conftest.py
"""Shared test fixtures."""
from __future__ import annotations

import io
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from src.document_processor.parser import ParsedStatement, Transaction


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    """Generate a realistic set of sample transactions."""
    return [
        Transaction(
            date=datetime(2024, 1, 5),
            description="DIRECT DEPOSIT - ACME CORP PAYROLL",
            amount=Decimal("5200.00"),
            balance=Decimal("8450.00"),
            transaction_type="credit",
        ),
        Transaction(
            date=datetime(2024, 1, 7),
            description="WHOLE FOODS MARKET #10423",
            amount=Decimal("-127.43"),
            balance=Decimal("8322.57"),
            transaction_type="debit",
        ),
        Transaction(
            date=datetime(2024, 1, 10),
            description="MONTHLY RENT PAYMENT - APT 4B",
            amount=Decimal("-2100.00"),
            balance=Decimal("6222.57"),
            transaction_type="debit",
        ),
        Transaction(
            date=datetime(2024, 1, 12),
            description="STARBUCKS STORE #14523",
            amount=Decimal("-6.75"),
            balance=Decimal("6215.82"),
            transaction_type="debit",
        ),
        Transaction(
            date=datetime(2024, 1, 15),
            description="AMAZON.COM AMZN.COM/BILL",
            amount=Decimal("-89.99"),
            balance=Decimal("6125.83"),
            transaction_type="debit",
        ),
        Transaction(
            date=datetime(2024, 1, 18),
            description="UBER TRIP HELP.UBER.COM",
            amount=Decimal("-24.50"),
            balance=Decimal("6101.33"),
            transaction_type="debit",
        ),
        Transaction(
            date=datetime(2024, 1, 20),
            description="TRANSFER FROM SAVINGS",
            amount=Decimal("500.00"),
            balance=Decimal("6601.33"),
            transaction_type="credit",
        ),
        Transaction(
            date=datetime(2024, 1, 25),
            description="ELECTRIC COMPANY AUTOPAY",
            amount=Decimal("-145.00"),
            balance=Decimal("6456.33"),
            transaction_type="debit",
        ),
    ]


@pytest.fixture
def sample_parsed_statement(sample_transactions) -> ParsedStatement:
    """Create a sample parsed statement."""
    return ParsedStatement(
        source_file="test_statement.pdf",
        account_number="****1234",
        account_holder="John Doe",
        bank_name="Chase",
        statement_period_start=datetime(2024, 1, 1),
        statement_period_end=datetime(2024, 1, 31),
        opening_balance=Decimal("5350.00"),
        closing_balance=Decimal("6456.33"),
        transactions=sample_transactions,
        raw_text="Sample raw text",
    )


@pytest.fixture
def sample_transaction_dicts(sample_transactions) -> list[dict]:
    """Convert sample transactions to dict format."""
    return [t.to_dict() for t in sample_transactions]