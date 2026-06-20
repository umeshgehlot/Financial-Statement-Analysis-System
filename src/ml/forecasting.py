# src/ml/forecasting.py
"""
Time series forecasting for spending patterns and balance projections.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


@dataclass
class ForecastResult:
    metric: str
    current_value: float
    forecast_values: list[dict[str, Any]]
    confidence_interval: list[dict[str, Any]]
    trend: str  # increasing, decreasing, stable
    summary: str


class SpendingForecaster:
    """
    Spending and balance forecasting using statistical methods.

    Uses exponential smoothing and linear regression for projections.
    No heavy ML dependencies needed for basic financial forecasting.
    """

    def __init__(self, forecast_periods: int = 3):
        self.forecast_periods = forecast_periods  # months ahead

    def forecast(
        self, transactions: list[dict[str, Any]]
    ) -> dict[str, ForecastResult]:
        """Generate spending and balance forecasts from transaction history."""
        df = self._prepare_data(transactions)

        if len(df) < 3:
            logger.warning("insufficient_data_for_forecasting", rows=len(df))
            return {}

        results = {}
        results["spending"] = self._forecast_spending(df)
        results["income"] = self._forecast_income(df)
        results["balance"] = self._forecast_balance(df)
        results["category_forecasts"] = self._forecast_by_category(df)

        logger.info("forecasting_complete", metrics=list(results.keys()))
        return results

    def _prepare_data(
        self, transactions: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert transactions to monthly aggregated DataFrame."""
        df = pd.DataFrame(transactions)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

        monthly = df.groupby(df["date"].dt.to_period("M")).agg(
            total_spending=("amount", lambda x: x[x < 0].sum()),
            total_income=("amount", lambda x: x[x > 0].sum()),
            net_flow=("amount", "sum"),
            transaction_count=("amount", "count"),
        )

        monthly.index = monthly.index.to_timestamp()
        return monthly

    def _forecast_spending(self, df: pd.DataFrame) -> ForecastResult:
        """Forecast future monthly spending."""
        spending = df["total_spending"].abs()
        return self._generate_forecast(
            spending, "Monthly Spending", invert_trend=True
        )

    def _forecast_income(self, df: pd.DataFrame) -> ForecastResult:
        """Forecast future monthly income."""
        income = df["total_income"]
        return self._generate_forecast(income, "Monthly Income")

    def _forecast_balance(self, df: pd.DataFrame) -> ForecastResult:
        """Project balance based on net flow trends."""
        net = df["net_flow"]
        return self._generate_forecast(net, "Net Cash Flow")

    def _forecast_by_category(
        self, df: pd.DataFrame
    ) -> dict[str, ForecastResult]:
        """Per-category spending forecasts (requires category data)."""
        # This would need transaction-level category data
        return {}

    def _generate_forecast(
        self,
        series: pd.Series,
        metric_name: str,
        invert_trend: bool = False,
    ) -> ForecastResult:
        """Generate forecast using exponential weighted moving average + regression."""
        values = series.values.astype(float)
        n = len(values)

        # Linear regression for trend
        x = np.arange(n).reshape(-1, 1)
        slope, intercept, r_value, _, _ = stats.linregress(x.flatten(), values)

        # Determine trend direction
        if invert_trend:
            trend = "increasing" if slope < 0 else "decreasing"
        else:
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")

        # Exponential smoothing
        alpha = 0.3
        smoothed = [values[0]]
        for i in range(1, n):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])

        # Forecast
        last_date = series.index[-1]
        forecast_values = []
        confidence_interval = []

        residual_std = np.std(values - np.array(smoothed[:n]))

        for i in range(1, self.forecast_periods + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            forecast_val = slope * (n + i - 1) + intercept

            # Blend with exponential smoothing
            blend_val = 0.6 * forecast_val + 0.4 * smoothed[-1]

            forecast_values.append({
                "date": forecast_date.strftime("%Y-%m"),
                "value": round(blend_val, 2),
            })

            z = 1.96  # 95% CI
            ci_width = z * residual_std * np.sqrt(i)
            confidence_interval.append({
                "date": forecast_date.strftime("%Y-%m"),
                "lower": round(blend_val - ci_width, 2),
                "upper": round(blend_val + ci_width, 2),
            })

        current_value = float(values[-1])

        summary_parts = [
            f"{metric_name} is trending {trend}.",
            f"Current: ${current_value:,.2f}.",
        ]
        if forecast_values:
            summary_parts.append(
                f"Projected next month: ${forecast_values[0]['value']:,.2f}."
            )

        return ForecastResult(
            metric=metric_name,
            current_value=current_value,
            forecast_values=forecast_values,
            confidence_interval=confidence_interval,
            trend=trend,
            summary=" ".join(summary_parts),
        )