# tests/test_agents.py
"""Tests for the LangGraph agent and analytical tools."""
from __future__ import annotations

import pytest

from src.agents.tools import FinancialToolKit


class TestFinancialToolKit:
    @pytest.fixture
    def toolkit(self, sample_transaction_dicts):
        toolkit = FinancialToolKit()
        toolkit.set_transactions(sample_transaction_dicts)
        return toolkit

    def test_spending_by_category(self, toolkit):
        tool = [t for t in toolkit.get_tools() if t.name == "_analyze_spending_by_category"][0]
        result = tool.invoke({})
        assert "Spending by Category" in result
        assert "$" in result

    def test_find_large_transactions(self, toolkit):
        tool = [t for t in toolkit.get_tools() if t.name == "_find_large_transactions"][0]
        result = tool.invoke({"threshold": 100.0})
        assert "RENT" in result or "DEPOSIT" in result

    def test_monthly_summary(self, toolkit):
        tool = [t for t in toolkit.get_tools() if t.name == "_calculate_monthly_summary"][0]
        result = tool.invoke({})
        assert "Monthly Financial Summary" in result
        assert "$" in result

    def test_cash_flow(self, toolkit):
        tool = [t for t in toolkit.get_tools() if t.name == "_analyze_cash_flow"][0]
        result = tool.invoke({})
        assert "Total Income" in result
        assert "Total Expenses" in result
        assert "Savings Rate" in result

    def test_recurring_transactions(self, toolkit):
        tool = [t for t in toolkit.get_tools() if t.name == "_detect_recurring_transactions"][0]
        result = tool.invoke({"min_occurrences": 2})
        assert isinstance(result, str)

    def test_empty_toolkit(self):
        toolkit = FinancialToolKit()
        tool = [t for t in toolkit.get_tools() if t.name == "_analyze_spending_by_category"][0]
        result = tool.invoke({})
        assert "No transaction data" in result