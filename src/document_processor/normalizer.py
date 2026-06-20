# src/document_processor/normalizer.py
"""
Transaction categorization and normalization using rule-based + ML hybrid approach.
"""
from __future__ import annotations

import re
from decimal import Decimal

import structlog

from .parser import Transaction

logger = structlog.get_logger(__name__)

# Category mapping rules — easily extensible
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Income/Salary", [
        r"(?i)payroll|direct\s*dep|salary|wage|income",
    ]),
    ("Income/Transfer", [
        r"(?i)transfer\s*(?:from|in)|zelle\s*(?:from|receive)|venmo\s*(?:receive|credit)",
    ]),
    ("Housing/Rent", [
        r"(?i)rent|mortgage|hoa\s*(?:fee|dues)|property\s*tax",
    ]),
    ("Housing/Utilities", [
        r"(?i)electric|gas\s*(?:bill|co)|water\s*(?:bill|sewer)|internet|comcast|verizon|at&t|spectrum",
    ]),
    ("Food/Groceries", [
        r"(?i)grocery|whole\s*foods|trader\s*joe|kroger|safeway|walmart\s*(?:grocery|super)|costco|aldi|publix",
    ]),
    ("Food/Dining", [
        r"(?i)restaurant|starbucks|mcdonald|chipotle|doordash|ubereats|grubhub|pizza|cafe|coffee",
    ]),
    ("Transportation", [
        r"(?i)uber(?!\s*eats)|lyft|gas\s*station|shell|chevron|exxon|parking|toll|metro\s*transit|auto\s*(?:pay|insur)",
    ]),
    ("Shopping", [
        r"(?i)amazon|target|walmart(?!\s*(?:grocery|super))|best\s*buy|nordstrom|macys|ebay",
    ]),
    ("Healthcare", [
        r"(?i)pharmacy|cvs|walgreen|hospital|medical|dental|vision|optum|anthem|blue\s*cross",
    ]),
    ("Insurance", [
        r"(?i)insurance|allstate|geico|state\s*farm|progressive|premium",
    ]),
    ("Entertainment", [
        r"(?i)netflix|spotify|hulu|disney|movie|theater|gaming|steam|playstation|xbox|apple\s*(?:music|tv)",
    ]),
    ("Financial/Fees", [
        r"(?i)fee|service\s*charge|overdraft|interest\s*(?:charge|paid)|atm\s*(?:fee|surcharge)|annual\s*fee",
    ]),
    ("Financial/Investment", [
        r"(?i)investment|brokerage|fidelity|vanguard|schwab|robinhood|crypto|coinbase",
    ]),
    ("Subscription", [
        r"(?i)subscription|membership|annual\s*(?:fee|renewal)|monthly\s*(?:fee|plan)",
    ]),
    ("ATM/Cash", [
        r"(?i)atm\s*(?:withdrawal|cash)|cash\s*withdrawal",
    ]),
]


class TransactionNormalizer:
    """Cleans, categorizes, and normalizes parsed transactions."""

    def __init__(self):
        self._compiled_rules = [
            (category, [re.compile(p) for p in patterns])
            for category, patterns in CATEGORY_RULES
        ]

    def normalize(self, transactions: list[Transaction]) -> list[Transaction]:
        """Apply normalization pipeline to all transactions."""
        normalized = []
        for txn in transactions:
            txn = self._clean_description(txn)
            txn = self._categorize(txn)
            txn = self._normalize_amount(txn)
            normalized.append(txn)

        logger.info(
            "normalization_complete",
            count=len(normalized),
            categorized=sum(1 for t in normalized if t.category),
        )
        return normalized

    def _clean_description(self, txn: Transaction) -> Transaction:
        """Clean up messy bank statement descriptions."""
        desc = txn.description
        # Remove reference numbers, extra whitespace
        desc = re.sub(r"\s+#\d+", "", desc)
        desc = re.sub(r"\s{2,}", " ", desc)
        desc = re.sub(r"[^\w\s\-/&'.]", "", desc)
        desc = desc.strip().title()
        txn.description = desc
        return txn

    def _categorize(self, txn: Transaction) -> Transaction:
        """Assign a category using rule-based pattern matching."""
        if txn.category:
            return txn

        desc = txn.description
        for category, patterns in self._compiled_rules:
            for pattern in patterns:
                if pattern.search(desc):
                    txn.category = category
                    return txn

        txn.category = "Uncategorized"
        return txn

    def _normalize_amount(self, txn: Transaction) -> Transaction:
        """Ensure consistent amount representation."""
        if not isinstance(txn.amount, Decimal):
            txn.amount = Decimal(str(txn.amount))
        txn.amount = txn.amount.quantize(Decimal("0.01"))
        return txn