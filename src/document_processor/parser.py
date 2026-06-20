# src/document_processor/parser.py
"""
Multi-format document parser supporting PDF, CSV, Excel, and OFX bank statements.
Uses PyMuPDF for PDFs, pandas for tabular data, and custom rules for structured formats.
"""
from __future__ import annotations

import csv
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from pypdf import PdfReader

logger = structlog.get_logger(__name__)


class DocumentFormat(str, Enum):
    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    OFX = "ofx"
    QFX = "qfx"
    UNKNOWN = "unknown"


@dataclass
class Transaction:
    date: datetime
    description: str
    amount: Decimal
    balance: Decimal | None = None
    category: str | None = None
    transaction_type: str | None = None  # debit | credit
    check_number: str | None = None
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "description": self.description,
            "amount": str(self.amount),
            "balance": str(self.balance) if self.balance else None,
            "category": self.category,
            "transaction_type": self.transaction_type,
            "check_number": self.check_number,
            "reference": self.reference,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """Convert transaction to human-readable text for embedding."""
        parts = [
            f"Date: {self.date.strftime('%Y-%m-%d')}",
            f"Description: {self.description}",
            f"Amount: ${self.amount:,.2f}",
        ]
        if self.balance is not None:
            parts.append(f"Balance: ${self.balance:,.2f}")
        if self.transaction_type:
            parts.append(f"Type: {self.transaction_type}")
        if self.category:
            parts.append(f"Category: {self.category}")
        return " | ".join(parts)


@dataclass
class ParsedStatement:
    """Structured output of a parsed bank statement."""
    source_file: str
    account_number: str | None
    account_holder: str | None
    bank_name: str | None
    statement_period_start: datetime | None
    statement_period_end: datetime | None
    opening_balance: Decimal | None
    closing_balance: Decimal | None
    transactions: list[Transaction]
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_credits(self) -> Decimal:
        return sum(
            (t.amount for t in self.transactions if t.amount > 0),
            Decimal("0"),
        )

    @property
    def total_debits(self) -> Decimal:
        return sum(
            (t.amount for t in self.transactions if t.amount < 0),
            Decimal("0"),
        )

    @property
    def transaction_count(self) -> int:
        return len(self.transactions)

    def to_documents(self) -> list[dict[str, Any]]:
        """Convert parsed statement into document chunks for RAG indexing."""
        docs = []

        # Statement-level summary
        summary = (
            f"Bank Statement from {self.bank_name or 'Unknown Bank'}\n"
            f"Account: {self.account_number or 'N/A'}\n"
            f"Holder: {self.account_holder or 'N/A'}\n"
            f"Period: {self.statement_period_start} to {self.statement_period_end}\n"
            f"Opening Balance: ${self.opening_balance or 0:,.2f}\n"
            f"Closing Balance: ${self.closing_balance or 0:,.2f}\n"
            f"Total Credits: ${self.total_credits:,.2f}\n"
            f"Total Debits: ${self.total_debits:,.2f}\n"
            f"Transaction Count: {self.transaction_count}"
        )
        docs.append({
            "page_content": summary,
            "metadata": {
                "source": self.source_file,
                "doc_type": "statement_summary",
                "bank_name": self.bank_name,
                "account_number": self.account_number,
            },
        })

        # Individual transactions
        for i, txn in enumerate(self.transactions):
            docs.append({
                "page_content": txn.to_text(),
                "metadata": {
                    "source": self.source_file,
                    "doc_type": "transaction",
                    "transaction_index": i,
                    "date": txn.date.isoformat(),
                    "amount": str(txn.amount),
                    "transaction_type": txn.transaction_type,
                    "category": txn.category,
                },
            })

        return docs


class BaseParser(ABC):
    """Abstract base parser for bank statement formats."""

    @abstractmethod
    def parse(self, file_path: Path, content: bytes | None = None) -> ParsedStatement:
        ...

    @staticmethod
    def _detect_amount(text: str) -> Decimal | None:
        """Extract monetary amount from text."""
        patterns = [
            r"\$?([\d,]+\.\d{2})",
            r"$$([\d,]+\.\d{2})$$",  # Parentheses = negative
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    amount = Decimal(match.group(1).replace(",", ""))
                    if "(" in text and ")" in text:
                        amount = -amount
                    return amount
                except InvalidOperation:
                    continue
        return None

    @staticmethod
    def _parse_date(text: str) -> datetime | None:
        """Try multiple date formats."""
        formats = [
            "%m/%d/%Y",
            "%m/%d/%y",
            "%Y-%m-%d",
            "%d-%b-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
        ]
        text = text.strip()
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None


class PDFParser(BaseParser):
    """Parser for PDF bank statements using PyMuPDF and pattern matching."""

    # Common patterns found across major US bank statements
    HEADER_PATTERNS = {
        "account_number": [
            r"Account\s*(?:Number|#)[:\s]*(\*{0,4}\d{4,})",
            r"Account[:\s]*(\d{6,})",
            r"A/C\s*(?:No|#)[:\s]*(\d+)",
        ],
        "account_holder": [
            r"(?:Account\s*Holder|Name)[:\s]*(.+?)(?:\n|$)",
            r"^([A-Z][A-Za-z\s]+?)(?:\n)",
        ],
        "statement_period": [
            r"(?:Statement|Period)[:\s]*(\w+\s+\d{1,2},?\s+\d{4})\s*(?:to|through|-)\s*(\w+\s+\d{1,2},?\s+\d{4})",
            r"(\d{2}/\d{2}/\d{4})\s*(?:to|-)\s*(\d{2}/\d{2}/\d{4})",
        ],
        "opening_balance": [
            r"(?:Opening|Beginning|Previous)\s*(?:Balance)[:\s]*\$?([\d,]+\.\d{2})",
            r"(?:Balance\s*Forward|Brought\s*Forward)[:\s]*\$?([\d,]+\.\d{2})",
        ],
        "closing_balance": [
            r"(?:Closing|Ending|New)\s*(?:Balance)[:\s]*\$?([\d,]+\.\d{2})",
            r"(?:Current|Available)\s*(?:Balance)[:\s]*\$?([\d,]+\.\d{2})",
        ],
    }

    TRANSACTION_PATTERNS = [
        # MM/DD  Description  Amount  Balance
        re.compile(
            r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+"
            r"(.+?)\s+"
            r"(-?\$?[\d,]+\.\d{2})\s+"
            r"(\$?[\d,]+\.\d{2})?"
        ),
        # MM/DD  Description  Debit  Credit  Balance
        re.compile(
            r"(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+"
            r"(.+?)\s+"
            r"(-?\$?[\d,]+\.\d{2}|-)\s+"
            r"(-?\$?[\d,]+\.\d{2}|-)\s+"
            r"(\$?[\d,]+\.\d{2})?"
        ),
    ]

    def parse(self, file_path: Path, content: bytes | None = None) -> ParsedStatement:
        logger.info("parsing_pdf", file=str(file_path))

        if content:
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
        else:
            import fitz
            doc = fitz.open(str(file_path))

        raw_text = ""
        for page in doc:
            raw_text += page.get_text() + "\n"
        doc.close()

        return self._extract_statement(file_path.name, raw_text)

    def _extract_statement(self, filename: str, text: str) -> ParsedStatement:
        """Extract structured data from raw PDF text."""
        bank_name = self._detect_bank(text)
        account_number = self._extract_field("account_number", text)
        account_holder = self._extract_field("account_holder", text)

        period_start, period_end = None, None
        for pattern in self.HEADER_PATTERNS["statement_period"]:
            match = pattern.search(text)
            if match:
                period_start = self._parse_date(match.group(1))
                period_end = self._parse_date(match.group(2))
                break

        opening = self._extract_decimal_field("opening_balance", text)
        closing = self._extract_decimal_field("closing_balance", text)

        transactions = self._extract_transactions(text, period_start.year if period_start else None)

        return ParsedStatement(
            source_file=filename,
            account_number=account_number,
            account_holder=account_holder,
            bank_name=bank_name,
            statement_period_start=period_start,
            statement_period_end=period_end,
            opening_balance=opening,
            closing_balance=closing,
            transactions=transactions,
            raw_text=text,
        )

    def _detect_bank(self, text: str) -> str | None:
        banks = {
            "Chase": r"JPMorgan\s*Chase|Chase\s*Bank",
            "Bank of America": r"Bank\s*of\s*America|BofA",
            "Wells Fargo": r"Wells\s*Fargo",
            "Citibank": r"Citi(?:bank)?",
            "Capital One": r"Capital\s*One",
            "US Bank": r"U\.?S\.?\s*Bank",
            "PNC": r"PNC\s*Bank",
            "TD Bank": r"TD\s*Bank",
            "Truist": r"Truist",
            "Ally": r"Ally\s*Bank",
        }
        text_upper = text[:2000]  # Header region
        for name, pattern in banks.items():
            if re.search(pattern, text_upper, re.IGNORECASE):
                return name
        return None

    def _extract_field(self, field_name: str, text: str) -> str | None:
        for pattern in self.HEADER_PATTERNS.get(field_name, []):
            match = pattern.search(text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_decimal_field(self, field_name: str, text: str) -> Decimal | None:
        for pattern in self.HEADER_PATTERNS.get(field_name, []):
            match = pattern.search(text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return Decimal(match.group(1).replace(",", ""))
                except InvalidOperation:
                    continue
        return None

    def _extract_transactions(
        self, text: str, default_year: int | None
    ) -> list[Transaction]:
        transactions = []
        seen = set()

        for pattern in self.TRANSACTION_PATTERNS:
            for match in pattern.finditer(text):
                date_str = match.group(1)
                desc = match.group(2).strip()

                txn_date = self._parse_date(date_str)
                if not txn_date and default_year:
                    try:
                        txn_date = datetime.strptime(
                            f"{date_str}/{default_year}", "%m/%d/%Y"
                        )
                    except ValueError:
                        continue

                if not txn_date:
                    continue

                amount = self._detect_amount(match.group(3))
                if amount is None:
                    continue

                balance = None
                if len(match.groups()) >= 5 and match.group(5):
                    balance = self._detect_amount(match.group(5))
                elif len(match.groups()) >= 4 and match.group(4):
                    bal_candidate = self._detect_amount(match.group(4))
                    if bal_candidate and abs(bal_candidate) > abs(amount) * 10:
                        balance = bal_candidate

                key = (txn_date.isoformat(), str(amount), desc[:50])
                if key in seen:
                    continue
                seen.add(key)

                txn_type = "credit" if amount > 0 else "debit"

                transactions.append(Transaction(
                    date=txn_date,
                    description=desc,
                    amount=amount,
                    balance=balance,
                    transaction_type=txn_type,
                ))

        transactions.sort(key=lambda t: t.date)
        logger.info("extracted_transactions", count=len(transactions))
        return transactions


class CSVParser(BaseParser):
    """Parser for CSV/Excel bank statement exports."""

    # Column name mappings across common bank export formats
    COLUMN_MAPPINGS = {
        "date": [
            "date", "transaction date", "trans date", "post date",
            "posting date", "settlement date", "booking date",
        ],
        "description": [
            "description", "memo", "narrative", "details",
            "transaction description", "payee", "name",
            "transaction details", "particulars",
        ],
        "amount": [
            "amount", "transaction amount", "value", "sum",
            "debit/credit", "debit credit",
        ],
        "debit": [
            "debit", "debit amount", "withdrawal", "withdrawals",
            "money out", "payments",
        ],
        "credit": [
            "credit", "credit amount", "deposit", "deposits",
            "money in", "receipts",
        ],
        "balance": [
            "balance", "running balance", "available balance",
            "ledger balance", "closing balance",
        ],
        "category": [
            "category", "type", "transaction type", "classification",
        ],
        "check_number": [
            "check number", "check #", "cheque number", "cheque no",
            "reference number", "ref no",
        ],
    }

    def parse(self, file_path: Path, content: bytes | None = None) -> ParsedStatement:
        logger.info("parsing_csv", file=str(file_path))

        if file_path.suffix.lower() in (".xlsx", ".xls"):
            df = self._read_excel(file_path, content)
        else:
            df = self._read_csv(file_path, content)

        df = self._normalize_columns(df)
        transactions = self._extract_transactions(df)

        account_number = None
        if content:
            text = content.decode("utf-8", errors="replace")[:5000]
        else:
            text = file_path.read_text(errors="replace")[:5000]

        acct_match = re.search(r"Account[:\s#]*(\d{4,})", text)
        if acct_match:
            account_number = acct_match.group(1)

        return ParsedStatement(
            source_file=file_path.name,
            account_number=account_number,
            account_holder=None,
            bank_name=None,
            statement_period_start=transactions[0].date if transactions else None,
            statement_period_end=transactions[-1].date if transactions else None,
            opening_balance=transactions[0].balance if transactions else None,
            closing_balance=transactions[-1].balance if transactions else None,
            transactions=transactions,
            raw_text=df.to_string(),
            metadata={"row_count": len(df), "columns": list(df.columns)},
        )

    def _read_csv(self, file_path: Path, content: bytes | None) -> pd.DataFrame:
        if content:
            return pd.read_csv(io.BytesIO(content), on_bad_lines="skip")
        return pd.read_csv(file_path, on_bad_lines="skip")

    def _read_excel(self, file_path: Path, content: bytes | None) -> pd.DataFrame:
        if content:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        return pd.read_excel(file_path, engine="openpyxl")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map heterogeneous column names to standard names."""
        col_mapping = {}
        df_cols_lower = {col: col.lower().strip() for col in df.columns}

        for standard_name, aliases in self.COLUMN_MAPPINGS.items():
            for orig_col, lower_col in df_cols_lower.items():
                if lower_col in aliases:
                    col_mapping[orig_col] = standard_name
                    break

        df = df.rename(columns=col_mapping)
        return df

    def _extract_transactions(self, df: pd.DataFrame) -> list[Transaction]:
        transactions = []

        has_separate_dc = "debit" in df.columns and "credit" in df.columns

        for _, row in df.iterrows():
            date_val = row.get("date")
            if pd.isna(date_val):
                continue

            txn_date = pd.to_datetime(date_val).to_pydatetime()
            description = str(row.get("description", "")).strip()
            if not description:
                continue

            if has_separate_dc:
                debit = self._to_decimal(row.get("debit", 0))
                credit = self._to_decimal(row.get("credit", 0))
                amount = credit - debit if credit else -debit
            else:
                amount = self._to_decimal(row.get("amount", 0))

            balance = self._to_decimal(row.get("balance"))
            category = str(row.get("category", "")) or None
            txn_type = "credit" if amount > 0 else "debit"

            transactions.append(Transaction(
                date=txn_date,
                description=description,
                amount=amount,
                balance=balance,
                transaction_type=txn_type,
                category=category,
            ))

        transactions.sort(key=lambda t: t.date)
        return transactions

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if pd.isna(value) or value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.replace("$", "").replace(",", "").strip()
                if value.startswith("(") and value.endswith(")"):
                    value = "-" + value[1:-1]
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None


class DocumentParserFactory:
    """Factory that detects format and dispatches to the correct parser."""

    PARSERS = {
        DocumentFormat.PDF: PDFParser(),
        DocumentFormat.CSV: CSVParser(),
        DocumentFormat.XLSX: CSVParser(),
        DocumentFormat.XLS: CSVParser(),
    }

    @classmethod
    def detect_format(cls, file_path: Path) -> DocumentFormat:
        suffix = file_path.suffix.lower().lstrip(".")
        try:
            return DocumentFormat(suffix)
        except ValueError:
            return DocumentFormat.UNKNOWN

    @classmethod
    def parse(
        cls, file_path: Path, content: bytes | None = None
    ) -> ParsedStatement:
        fmt = cls.detect_format(file_path)
        if fmt == DocumentFormat.UNKNOWN:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        parser = cls.PARSERS[fmt]
        return parser.parse(file_path, content)