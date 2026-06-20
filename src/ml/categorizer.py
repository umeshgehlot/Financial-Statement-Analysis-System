# src/ml/categorizer.py
"""
ML-enhanced transaction categorizer using few-shot classification.
"""
from __future__ import annotations

from typing import Any

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_settings

logger = structlog.get_logger(__name__)


CATEGORIZATION_PROMPT = """You are a financial transaction categorizer.
Given a list of bank transactions, assign each one a category from this list:

- Income/Salary
- Income/Transfer
- Housing/Rent
- Housing/Utilities
- Food/Groceries
- Food/Dining
- Transportation
- Shopping
- Healthcare
- Insurance
- Entertainment
- Financial/Fees
- Financial/Investment
- Subscription
- ATM/Cash
- Transfer
- Other

Respond ONLY with a JSON array of objects like:
[{{"index": 0, "category": "Food/Dining", "confidence": 0.95}}, ...]

Transactions to categorize:
{transactions}"""


class TransactionCategorizer:
    """
    LLM-based transaction categorizer for transactions that
    couldn't be categorized by the rule-based normalizer.
    Uses batch inference for efficiency.
    """

    def __init__(self, batch_size: int = 50):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai.model,
            temperature=0,
            max_tokens=4096,
        )
        self.batch_size = batch_size
        self.prompt = ChatPromptTemplate.from_template(CATEGORIZATION_PROMPT)

    def categorize(
        self, transactions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Categorize a list of transactions using LLM inference."""
        uncategorized = [
            (i, t) for i, t in enumerate(transactions)
            if not t.get("category") or t["category"] == "Uncategorized"
        ]

        if not uncategorized:
            return transactions

        logger.info(
            "categorizing_transactions",
            total=len(transactions),
            uncategorized=len(uncategorized),
        )

        # Process in batches
        for batch_start in range(0, len(uncategorized), self.batch_size):
            batch = uncategorized[batch_start:batch_start + self.batch_size]
            batch_transactions = [
                {
                    "index": i,
                    "description": t.get("description", ""),
                    "amount": str(t.get("amount", 0)),
                    "date": t.get("date", ""),
                }
                for i, t in batch
            ]

            try:
                chain = self.prompt | self.llm
                result = chain.invoke({
                    "transactions": str(batch_transactions)
                })

                import json
                categories = json.loads(result.content)

                for cat_info in categories:
                    idx = cat_info["index"]
                    category = cat_info["category"]
                    confidence = cat_info.get("confidence", 0.5)

                    if confidence >= 0.6:
                        transactions[idx]["category"] = category
                        transactions[idx]["category_confidence"] = confidence

            except Exception as e:
                logger.error("categorization_batch_error", error=str(e))

        categorized_count = sum(
            1 for t in transactions
            if t.get("category") and t["category"] != "Uncategorized"
        )
        logger.info(
            "categorization_complete",
            categorized=categorized_count,
            total=len(transactions),
        )

        return transactions