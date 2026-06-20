# src/agents/tools.py
"""
LangGraph tools for financial analysis operations.
Each tool wraps a specific analytical capability and returns structured output.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Annotated, Any

import pandas as pd
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)


class FinancialToolKit:
    """Collection of financial analysis tools used by the LangGraph agent."""

    def __init__(self, transactions: list[dict[str, Any]] | None = None):
        self._transactions = transactions or []
        self._df: pd.DataFrame | None = None
        if self._transactions:
            self._build_dataframe()

    def _build_dataframe(self):
        """Convert transactions to a pandas DataFrame for analysis."""
        df = pd.DataFrame(self._transactions)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.to_period("M")
            df["weekday"] = df["date"].dt.day_name()
        if "amount" in df.columns:
            df["amount"] = df["amount"].apply(
                lambda x: Decimal(str(x)) if pd.notna(x) else Decimal("0")
            )
            df["amount_float"] = df["amount"].apply(float)
        self._df = df

    def set_transactions(self, transactions: list[dict[str, Any]]):
        """Update the transaction dataset."""
        self._transactions = transactions
        self._build_dataframe()

    def get_tools(self) -> list:
        """Return all tools bound to this toolkit instance."""
        return [
            self._analyze_spending_by_category,
            self._analyze_spending_over_time,
            self._find_large_transactions,
            self._calculate_monthly_summary,
            self._detect_recurring_transactions,
            self._analyze_cash_flow,
            self._find_anomalous_transactions,
        ]

    @tool
    def _analyze_spending_by_category(
        self,
        start_date: Annotated[str, "Start date in YYYY-MM-DD format"] = "",
        end_date: Annotated[str, "End date in YYYY-MM-DD format"] = "",
    ) -> str:
        """Analyze total spending broken down by category within a date range."""
        if self._df is None or self._df.empty:
            return "No transaction data available for analysis."

        df = self._df.copy()
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]

        debits = df[df["amount_float"] < 0].copy()
        if debits.empty:
            return "No spending (debit) transactions found in the specified period."

        category_totals = (
            debits.groupby("category")["amount_float"]
            .sum()
            .abs()
            .sort_values(ascending=False)
        )

        total_spending = category_totals.sum()
        lines = ["Spending by Category:"]
        for cat, amount in category_totals.items():
            pct = (amount / total_spending * 100) if total_spending else 0
            lines.append(f"  {cat}: ${amount:,.2f} ({pct:.1f}%)")
        lines.append(f"\nTotal Spending: ${total_spending:,.2f}")

        result = "\n".join(lines)
        logger.info("spending_by_category", total=str(total_spending))
        return result

    @tool
    def _analyze_spending_over_time(
        self,
        period: Annotated[str, "Grouping period: 'daily', 'weekly', or 'monthly'"] = "monthly",
    ) -> str:
        """Analyze spending trends over time, grouped by the specified period."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        df = self._df[self._df["amount_float"] < 0].copy()
        if df.empty:
            return "No spending transactions found."

        if period == "monthly":
            grouped = df.groupby(df["date"].dt.to_period("M"))["amount_float"].sum().abs()
        elif period == "weekly":
            grouped = df.groupby(df["date"].dt.to_period("W"))["amount_float"].sum().abs()
        else:
            grouped = df.groupby(df["date"].dt.date)["amount_float"].sum().abs()

        lines = [f"Spending Over Time ({period}):"]
        for period_label, amount in grouped.items():
            lines.append(f"  {period_label}: ${amount:,.2f}")

        if len(grouped) > 1:
            avg = grouped.mean()
            trend = "increasing" if grouped.iloc[-1] > grouped.iloc[0] else "decreasing"
            lines.append(f"\nAverage per period: ${avg:,.2f}")
            lines.append(f"Overall trend: {trend}")

        return "\n".join(lines)

    @tool
    def _find_large_transactions(
        self,
        threshold: Annotated[float, "Minimum absolute amount to flag"] = 500.0,
        limit: Annotated[int, "Maximum number of results"] = 20,
    ) -> str:
        """Find transactions above a specified dollar threshold."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        large = self._df[self._df["amount_float"].abs() >= threshold].copy()
        large = large.sort_values("amount_float", key=abs, ascending=False).head(limit)

        if large.empty:
            return f"No transactions found above ${threshold:,.2f}."

        lines = [f"Transactions above ${threshold:,.2f}:"]
        for _, row in large.iterrows():
            sign = "+" if row["amount_float"] > 0 else ""
            lines.append(
                f"  {row['date'].strftime('%Y-%m-%d')} | "
                f"{row.get('description', 'N/A')[:50]} | "
                f"{sign}${row['amount_float']:,.2f} | "
                f"{row.get('category', 'N/A')}"
            )

        return "\n".join(lines)

    @tool
    def _calculate_monthly_summary(
        self,
        year: Annotated[int, "Year to summarize, e.g. 2024"] = 0,
    ) -> str:
        """Calculate a monthly financial summary including income, expenses, and net flow."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        df = self._df.copy()
        if year:
            df = df[df["date"].dt.year == year]

        if df.empty:
            return "No transactions found for the specified period."

        monthly = df.groupby(df["date"].dt.to_period("M")).agg(
            total_credits=("amount_float", lambda x: x[x > 0].sum()),
            total_debits=("amount_float", lambda x: x[x < 0].sum().abs()),
            net_flow=("amount_float", "sum"),
            transaction_count=("amount_float", "count"),
        )

        lines = ["Monthly Financial Summary:"]
        lines.append("-" * 70)
        lines.append(f"{'Month':<12} {'Income':>12} {'Expenses':>12} {'Net':>12} {'Txns':>6}")
        lines.append("-" * 70)

        for month, row in monthly.iterrows():
            lines.append(
                f"{str(month):<12} "
                f"${row['total_credits']:>10,.2f} "
                f"${row['total_debits']:>10,.2f} "
                f"${row['net_flow']:>10,.2f} "
                f"{int(row['transaction_count']):>5}"
            )

        totals = monthly.sum()
        lines.append("-" * 70)
        lines.append(
            f"{'TOTAL':<12} "
            f"${totals['total_credits']:>10,.2f} "
            f"${totals['total_debits']:>10,.2f} "
            f"${totals['net_flow']:>10,.2f} "
            f"{int(totals['transaction_count']):>5}"
        )

        return "\n".join(lines)

    @tool
    def _detect_recurring_transactions(
        self,
        min_occurrences: Annotated[int, "Minimum times a transaction must appear"] = 3,
    ) -> str:
        """Identify recurring transactions (subscriptions, regular payments)."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        df = self._df.copy()
        df["desc_normalized"] = df["description"].str.lower().str.strip()
        df["amount_rounded"] = df["amount_float"].round(2)

        recurring = (
            df.groupby(["desc_normalized", "amount_rounded"])
            .agg(
                count=("date", "count"),
                first_seen=("date", "min"),
                last_seen=("date", "max"),
                total=("amount_float", "sum"),
            )
            .reset_index()
        )

        recurring = recurring[recurring["count"] >= min_occurrences]
        recurring = recurring.sort_values("count", ascending=False)

        if recurring.empty:
            return f"No recurring transactions found (min {min_occurrences} occurrences)."

        lines = ["Recurring Transactions Detected:"]
        for _, row in recurring.iterrows():
            monthly_est = abs(row["total"]) / max(
                (row["last_seen"] - row["first_seen"]).days / 30, 1
            )
            lines.append(
                f"  {row['desc_normalized'][:40]:<40} "
                f"${row['amount_rounded']:>10,.2f} x{int(row['count'])} times "
                f"(~${monthly_est:,.2f}/mo)"
            )

        return "\n".join(lines)

    @tool
    def _analyze_cash_flow(
        self,
    ) -> str:
        """Analyze overall cash flow: total income, total expenses, savings rate."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        total_income = self._df[self._df["amount_float"] > 0]["amount_float"].sum()
        total_expenses = self._df[self._df["amount_float"] < 0]["amount_float"].sum().abs()
        net_flow = total_income - total_expenses
        savings_rate = (net_flow / total_income * 100) if total_income > 0 else 0

        lines = [
            "Cash Flow Analysis:",
            f"  Total Income:    ${total_income:>12,.2f}",
            f"  Total Expenses:  ${total_expenses:>12,.2f}",
            f"  Net Cash Flow:   ${net_flow:>12,.2f}",
            f"  Savings Rate:    {savings_rate:>11.1f}%",
        ]

        if savings_rate < 0:
            lines.append("\n  WARNING: Spending exceeds income in this period.")
        elif savings_rate < 10:
            lines.append("\n  NOTE: Savings rate is below the recommended 20% target.")

        return "\n".join(lines)

    @tool
    def _find_anomalous_transactions(
        self,
        zscore_threshold: Annotated[float, "Z-score threshold for anomaly detection"] = 2.5,
    ) -> str:
        """Find statistically unusual transactions using z-score analysis."""
        if self._df is None or self._df.empty:
            return "No transaction data available."

        amounts = self._df["amount_float"]
        mean = amounts.mean()
        std = amounts.std()

        if std == 0:
            return "All transactions have the same amount — no anomalies to detect."

        self._df["zscore"] = (amounts - mean).abs() / std
        anomalies = self._df[self._df["zscore"] >= zscore_threshold].copy()
        anomalies = anomalies.sort_values("zscore", ascending=False)

        if anomalies.empty:
            return f"No anomalous transactions found (threshold: {zscore_threshold} std devs)."

        lines = [
            f"Anomalous Transactions (z-score >= {zscore_threshold}):",
            f"  Based on mean=${mean:,.2f}, std=${std:,.2f}",
            "",
        ]
        for _, row in anomalies.head(15).iterrows():
            lines.append(
                f"  {row['date'].strftime('%Y-%m-%d')} | "
                f"{row.get('description', '')[:40]:<40} | "
                f"${row['amount_float']:>10,.2f} | "
                f"z={row['zscore']:.1f}"
            )

        return "\n".join(lines)