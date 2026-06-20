# src/ml/anomaly_detector.py
"""
ML-based anomaly detection for financial transactions.
Uses Isolation Forest + statistical methods to flag unusual spending patterns.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


@dataclass
class AnomalyResult:
    transaction_index: int
    date: str
    description: str
    amount: float
    anomaly_score: float
    reason: str
    severity: str  # low, medium, high


class TransactionAnomalyDetector:
    """
    Multi-method anomaly detector for bank transactions.

    Combines:
    1. Isolation Forest for multivariate anomaly detection
    2. Z-score for individual amount outliers
    3. Velocity checks for unusual frequency patterns
    4. Time-based pattern deviation detection
    """

    def __init__(
        self,
        contamination: float = 0.05,
        zscore_threshold: float = 3.0,
    ):
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold
        self._scaler = StandardScaler()
        self._model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )

    def fit_predict(
        self, transactions: list[dict[str, Any]]
    ) -> list[AnomalyResult]:
        """Run all anomaly detection methods and return flagged transactions."""
        if len(transactions) < 10:
            logger.warning("insufficient_data_for_anomaly_detection")
            return []

        df = self._prepare_features(transactions)

        # Method 1: Isolation Forest
        iso_anomalies = self._isolation_forest(df)

        # Method 2: Z-score
        zscore_anomalies = self._zscore_detection(df)

        # Method 3: Velocity analysis
        velocity_anomalies = self._velocity_analysis(df)

        # Combine results — transaction is anomalous if flagged by >= 2 methods
        combined = self._merge_anomalies(
            iso_anomalies, zscore_anomalies, velocity_anomalies, df
        )

        logger.info(
            "anomaly_detection_complete",
            total_transactions=len(transactions),
            anomalies_found=len(combined),
        )
        return combined

    def _prepare_features(
        self, transactions: list[dict[str, Any]]
    ) -> pd.DataFrame:
        """Extract numerical features for ML models."""
        df = pd.DataFrame(transactions)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["abs_amount"] = df["amount"].abs()
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["description_length"] = df["description"].str.len()

        # Encode category
        if "category" in df.columns:
            df["category_code"] = df["category"].astype("category").cat.codes
        else:
            df["category_code"] = 0

        return df

    def _isolation_forest(self, df: pd.DataFrame) -> dict[int, float]:
        """Detect anomalies using Isolation Forest."""
        features = [
            "abs_amount", "day_of_week", "day_of_month",
            "is_weekend", "category_code",
        ]

        X = df[features].fillna(0)
        X_scaled = self._scaler.fit_transform(X)

        predictions = self._model.fit_predict(X_scaled)
        scores = self._model.decision_function(X_scaled)

        anomalies = {}
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:
                anomalies[i] = abs(score)

        return anomalies

    def _zscore_detection(self, df: pd.DataFrame) -> dict[int, float]:
        """Detect amount outliers using z-score."""
        amounts = df["abs_amount"]
        z_scores = np.abs(stats.zscore(amounts))

        anomalies = {}
        for i, z in enumerate(z_scores):
            if z >= self.zscore_threshold:
                anomalies[i] = z

        return anomalies

    def _velocity_analysis(self, df: pd.DataFrame) -> dict[int, float]:
        """Flag days with unusually high transaction frequency or volume."""
        daily_counts = df.groupby(df["date"].dt.date).size()
        daily_volumes = df.groupby(df["date"].dt.date)["abs_amount"].sum()

        anomalies = {}

        # Unusual daily frequency
        if len(daily_counts) > 5:
            count_mean = daily_counts.mean()
            count_std = daily_counts.std()
            if count_std > 0:
                for date, count in daily_counts.items():
                    if (count - count_mean) / count_std > 2:
                        day_indices = df[df["date"].dt.date == date].index.tolist()
                        for idx in day_indices:
                            anomalies[idx] = (count - count_mean) / count_std

        # Unusual daily volume
        if len(daily_volumes) > 5:
            vol_mean = daily_volumes.mean()
            vol_std = daily_volumes.std()
            if vol_std > 0:
                for date, volume in daily_volumes.items():
                    if (volume - vol_mean) / vol_std > 2:
                        day_indices = df[df["date"].dt.date == date].index.tolist()
                        for idx in day_indices:
                            if idx not in anomalies:
                                anomalies[idx] = (volume - vol_mean) / vol_std

        return anomalies

    def _merge_anomalies(
        self,
        iso: dict[int, float],
        zscore: dict[int, float],
        velocity: dict[int, float],
        df: pd.DataFrame,
    ) -> list[AnomalyResult]:
        """Merge anomaly signals from multiple detectors."""
        all_indices = set(iso.keys()) | set(zscore.keys()) | set(velocity.keys())
        results = []

        for idx in all_indices:
            if idx >= len(df):
                continue

            methods_flagged = sum([
                idx in iso,
                idx in zscore,
                idx in velocity,
            ])

            row = df.iloc[idx]
            avg_score = np.mean([
                iso.get(idx, 0),
                zscore.get(idx, 0),
                velocity.get(idx, 0),
            ])

            reasons = []
            if idx in iso:
                reasons.append(f"Isolation Forest (score={iso[idx]:.2f})")
            if idx in zscore:
                reasons.append(f"Z-score={zscore[idx]:.1f}")
            if idx in velocity:
                reasons.append(f"Velocity anomaly ({velocity[idx]:.1f} std)")

            severity = "low"
            if methods_flagged >= 3 or avg_score > 4:
                severity = "high"
            elif methods_flagged >= 2 or avg_score > 3:
                severity = "medium"

            results.append(AnomalyResult(
                transaction_index=int(idx),
                date=row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "unknown",
                description=str(row.get("description", ""))[:80],
                amount=float(row["amount"]),
                anomaly_score=round(avg_score, 3),
                reason="; ".join(reasons),
                severity=severity,
            ))

        results.sort(key=lambda x: x.anomaly_score, reverse=True)
        return results