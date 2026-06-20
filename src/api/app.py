# src/api/app.py
"""
FastAPI application for the Financial Statement Analyzer.
Provides REST endpoints for document upload, querying, and analysis.
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_core.messages import HumanMessage

from src.agents.graph import build_financial_analysis_graph, create_analysis_input
from src.agents.tools import FinancialToolKit
from src.api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from src.config import get_settings
from src.document_processor.chunker import FinancialChunker
from src.document_processor.normalizer import TransactionNormalizer
from src.document_processor.parser import DocumentParserFactory
from src.ml.anomaly_detector import TransactionAnomalyDetector
from src.ml.forecasting import SpendingForecaster
from src.monitoring.metrics import MetricsCollector
from src.rag.chain import FinancialRAGChain
from src.rag.retriever import FinancialRetriever
from src.rag.vectorstore import VectorStoreManager

logger = structlog.get_logger(__name__)

# Application state
_app_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    settings = get_settings()
    logger.info(
        "application_starting",
        environment=settings.environment,
        model=settings.openai.model,
    )

    # Initialize core components
    vector_store_manager = VectorStoreManager()
    retriever = FinancialRetriever(
        vector_store_manager=vector_store_manager,
        top_k=settings.rag.top_k_results,
    )
    rag_chain = FinancialRAGChain(retriever=retriever)
    toolkit = FinancialToolKit()
    graph = build_financial_analysis_graph(rag_chain, toolkit)

    _app_state.update({
        "settings": settings,
        "vector_store_manager": vector_store_manager,
        "retriever": retriever,
        "rag_chain": rag_chain,
        "toolkit": toolkit,
        "graph": graph,
        "normalizer": TransactionNormalizer(),
        "chunker": FinancialChunker(),
        "anomaly_detector": TransactionAnomalyDetector(),
        "forecaster": SpendingForecaster(),
        "transactions": [],
        "files_processed": 0,
    })

    logger.info("application_ready")
    yield
    logger.info("application_shutting_down")


app = FastAPI(
    title="Financial Statement Analyzer",
    description="ML-powered bank statement analysis with RAG + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    settings = _app_state["settings"]
    vsm = _app_state["vector_store_manager"]

    try:
        store = vsm.get_or_create_store()
        vs_status = "connected"
    except Exception:
        vs_status = "disconnected"

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.environment,
        vectorstore_status=vs_status,
        documents_indexed=_app_state.get("files_processed", 0),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_statement(file: UploadFile = File(...)):
    """
    Upload and process a bank statement (PDF, CSV, Excel).

    Steps:
    1. Parse document to extract transactions
    2. Normalize and categorize transactions
    3. Chunk for RAG indexing
    4. Index in vector store
    """
    settings = _app_state["settings"]
    max_size = settings.max_upload_size_mb * 1024 * 1024

    # Validate file
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            413, f"File exceeds {settings.max_upload_size_mb}MB limit"
        )

    start_time = time.perf_counter()
    file_path = Path(file.filename)

    try:
        with MetricsCollector.track_processing("parse"):
            statement = DocumentParserFactory.parse(file_path, content)

        with MetricsCollector.track_processing("normalize"):
            normalizer: TransactionNormalizer = _app_state["normalizer"]
            statement.transactions = normalizer.normalize(statement.transactions)

        with MetricsCollector.track_processing("chunk"):
            chunker: FinancialChunker = _app_state["chunker"]
            docs = statement.to_documents()
            chunks = chunker.chunk_documents(docs)

        with MetricsCollector.track_processing("index"):
            vsm: VectorStoreManager = _app_state["vector_store_manager"]
            vsm.index_documents(chunks)

        # Update global transaction store
        txn_dicts = [t.to_dict() for t in statement.transactions]
        _app_state["transactions"].extend(txn_dicts)
        _app_state["toolkit"].set_transactions(_app_state["transactions"])
        _app_state["files_processed"] += 1

        processing_time = (time.perf_counter() - start_time) * 1000
        MetricsCollector.record_document_processed(
            file_path.suffix.lstrip("."), True
        )

        logger.info(
            "statement_uploaded",
            filename=file.filename,
            transactions=len(statement.transactions),
            time_ms=round(processing_time, 1),
        )

        return UploadResponse(
            file_id=str(uuid.uuid4()),
            filename=file.filename,
            transactions_extracted=len(statement.transactions),
            processing_time_ms=round(processing_time, 1),
            statement_summary={
                "bank_name": statement.bank_name,
                "account_number": statement.account_number,
                "period_start": str(statement.statement_period_start),
                "period_end": str(statement.statement_period_end),
                "opening_balance": str(statement.opening_balance),
                "closing_balance": str(statement.closing_balance),
                "total_credits": str(statement.total_credits),
                "total_debits": str(statement.total_debits),
            },
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        MetricsCollector.record_document_processed("unknown", False)
        logger.error("upload_error", error=str(e))
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_statements(request: QueryRequest):
    """
    Ask a natural language question about uploaded bank statements.
    Uses RAG to retrieve relevant context and generate an answer.
    """
    rag_chain: FinancialRAGChain = _app_state["rag_chain"]

    try:
        with MetricsCollector.track_query("rag"):
            start = time.perf_counter()
            result = rag_chain.invoke(
                question=request.question,
                filters=request.filters,
            )
            query_time = (time.perf_counter() - start) * 1000

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"],
            query_time_ms=round(query_time, 1),
        )

    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(500, f"Query error: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_statements(request: AnalysisRequest):
    """
    Run advanced analysis using the LangGraph agent with tool orchestration.
    Supports: comprehensive, spending, anomaly, forecast, cashflow.
    """
    graph = _app_state["graph"]
    toolkit: FinancialToolKit = _app_state["toolkit"]

    if not _app_state["transactions"]:
        raise HTTPException(
            400, "No statements uploaded. Upload a bank statement first."
        )

    # Enhance question based on analysis type
    enhanced_question = request.question
    if request.analysis_type == "anomaly":
        enhanced_question = (
            f"Using the find_anomalous_transactions tool, {request.question}"
        )
    elif request.analysis_type == "forecast":
        enhanced_question = (
            f"Based on historical spending data, {request.question}"
        )
    elif request.analysis_type == "cashflow":
        enhanced_question = (
            f"Using the analyze_cash_flow tool, {request.question}"
        )

    try:
        with MetricsCollector.track_query("agent"):
            start = time.perf_counter()

            input_state = create_analysis_input(enhanced_question, toolkit)
            result = graph.invoke(input_state)

            exec_time = (time.perf_counter() - start) * 1000

        final_answer = result.get("final_answer", "")
        if not final_answer and result.get("messages"):
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    final_answer = msg.content
                    break

        # Track which tools were used
        tools_used = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    tools_used.append(tc["name"])

        return AnalysisResponse(
            analysis=final_answer,
            tools_used=list(set(tools_used)),
            metadata={
                "analysis_type": request.analysis_type,
                "transaction_count": len(_app_state["transactions"]),
                "agent_steps": len(tools_used),
            },
            execution_time_ms=round(exec_time, 1),
        )

    except Exception as e:
        logger.error("analysis_error", error=str(e))
        raise HTTPException(500, f"Analysis error: {str(e)}")


@app.post("/anomalies")
async def detect_anomalies(zscore_threshold: float = 2.5):
    """Run anomaly detection on all uploaded transactions."""
    detector: TransactionAnomalyDetector = _app_state["anomaly_detector"]
    transactions = _app_state["transactions"]

    if not transactions:
        raise HTTPException(400, "No transactions available.")

    anomalies = detector.fit_predict(transactions)
    return {
        "anomalies": [
            {
                "date": a.date,
                "description": a.description,
                "amount": a.amount,
                "score": a.anomaly_score,
                "reason": a.reason,
                "severity": a.severity,
            }
            for a in anomalies
        ],
        "total_transactions": len(transactions),
        "anomaly_count": len(anomalies),
    }


@app.post("/forecast")
async def forecast_spending(periods: int = 3):
    """Generate spending and income forecasts."""
    forecaster = SpendingForecaster(forecast_periods=periods)
    transactions = _app_state["transactions"]

    if not transactions:
        raise HTTPException(400, "No transactions available.")

    results = forecaster.forecast(transactions)
    return {
        metric: {
            "current_value": f.current_value,
            "trend": f.trend,
            "forecast": f.forecast_values,
            "confidence_interval": f.confidence_interval,
            "summary": f.summary,
        }
        for metric, f in results.items()
        if hasattr(f, "current_value")
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=MetricsCollector.get_metrics(),
        media_type=MetricsCollector.get_metrics_content_type(),
    )


@app.get("/transactions")
async def list_transactions(
    limit: int = 100,
    offset: int = 0,
    category: str | None = None,
):
    """List processed transactions with optional filtering."""
    transactions = _app_state["transactions"]

    if category:
        transactions = [
            t for t in transactions
            if t.get("category", "").lower() == category.lower()
        ]

    return {
        "total": len(transactions),
        "offset": offset,
        "limit": limit,
        "transactions": transactions[offset:offset + limit],
    }