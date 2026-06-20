# src/monitoring/metrics.py
"""
Application metrics collection using Prometheus and OpenTelemetry.
Tracks RAG performance, latency, error rates, and business metrics.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import structlog
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = structlog.get_logger(__name__)

# Create a custom registry
REGISTRY = CollectorRegistry()

# --- RAG Metrics ---
RAG_QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "Time spent processing RAG queries",
    ["query_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=REGISTRY,
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["status", "query_type"],
    registry=REGISTRY,
)

RAG_RETRIEVAL_DOCS = Histogram(
    "rag_retrieval_documents",
    "Number of documents retrieved per query",
    buckets=[1, 3, 5, 10, 20, 50],
    registry=REGISTRY,
)

# --- Document Processing Metrics ---
DOCUMENTS_PROCESSED = Counter(
    "documents_processed_total",
    "Total documents processed",
    ["format", "status"],
    registry=REGISTRY,
)

TRANSACTIONS_EXTRACTED = Counter(
    "transactions_extracted_total",
    "Total transactions extracted",
    ["source_format"],
    registry=REGISTRY,
)

PROCESSING_DURATION = Histogram(
    "processing_duration_seconds",
    "Document processing time",
    ["stage"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# --- Agent Metrics ---
AGENT_STEPS = Histogram(
    "agent_steps_total",
    "Number of reasoning steps per agent invocation",
    buckets=[1, 2, 3, 5, 8, 10, 15],
    registry=REGISTRY,
)

TOOL_CALLS = Counter(
    "tool_calls_total",
    "Total tool invocations",
    ["tool_name", "status"],
    registry=REGISTRY,
)

# --- System Metrics ---
ACTIVE_UPLOADS = Gauge(
    "active_uploads",
    "Number of currently processing uploads",
    registry=REGISTRY,
)

VECTORSTORE_SIZE = Gauge(
    "vectorstore_documents_total",
    "Total documents in vector store",
    registry=REGISTRY,
)


class MetricsCollector:
    """Central metrics collection and reporting."""

    @staticmethod
    @contextmanager
    def track_query(query_type: str = "general"):
        """Context manager to track RAG query metrics."""
        start = time.perf_counter()
        try:
            yield
            RAG_QUERIES_TOTAL.labels(status="success", query_type=query_type).inc()
        except Exception:
            RAG_QUERIES_TOTAL.labels(status="error", query_type=query_type).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            RAG_QUERY_DURATION.labels(query_type=query_type).observe(duration)

    @staticmethod
    @contextmanager
    def track_processing(stage: str):
        """Context manager to track document processing metrics."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            PROCESSING_DURATION.labels(stage=stage).observe(duration)

    @staticmethod
    def record_tool_call(tool_name: str, success: bool):
        """Record a tool invocation."""
        status = "success" if success else "error"
        TOOL_CALLS.labels(tool_name=tool_name, status=status).inc()

    @staticmethod
    def record_document_processed(fmt: str, success: bool):
        """Record document processing completion."""
        status = "success" if success else "error"
        DOCUMENTS_PROCESSED.labels(format=fmt, status=status).inc()

    @staticmethod
    def get_metrics() -> bytes:
        """Generate current metrics in Prometheus format."""
        return generate_latest(REGISTRY)

    @staticmethod
    def get_metrics_content_type() -> str:
        return CONTENT_TYPE_LATEST