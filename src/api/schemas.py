# src/api/schemas.py
"""API request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    transactions_extracted: int
    processing_time_ms: float
    statement_summary: dict[str, Any]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    filters: dict[str, Any] | None = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    metadata: dict[str, Any]
    query_time_ms: float


class AnalysisRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    analysis_type: str = Field(
        "comprehensive",
        pattern="^(comprehensive|spending|anomaly|forecast|cashflow)$",
    )


class AnalysisResponse(BaseModel):
    analysis: str
    tools_used: list[str]
    metadata: dict[str, Any]
    execution_time_ms: float


class TransactionResponse(BaseModel):
    date: str
    description: str
    amount: str
    balance: str | None
    category: str | None
    transaction_type: str | None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    vectorstore_status: str
    documents_indexed: int


class MetricsResponse(BaseModel):
    total_queries: int
    total_documents: int
    avg_query_time_ms: float
    total_anomalies_detected: int