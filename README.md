# Financial Statement Analysis System

ML-powered bank statement analysis platform that combines Retrieval-Augmented Generation (RAG) with a LangGraph agent to let users upload bank statements and ask natural-language questions about their financial activity — spending breakdowns, cash flow, recurring charges, anomalies, and forecasts.

Built with FastAPI, LangChain/LangGraph, and a hybrid semantic + keyword retriever, with first-class support for local development (Chroma, Docker Compose) and production deployment on Azure (AKS, Azure AI Search, Cosmos DB, Blob Storage) via Terraform and Kubernetes manifests.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Observability](#observability)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Contributing](#contributing)

## Overview

A user uploads a bank statement (PDF, CSV, or Excel). The system parses it into structured transactions, normalizes and categorizes each one, indexes them for semantic search, and exposes two ways to interrogate the data:

1. **`/query`** — a direct RAG pipeline for grounded Q&A over the indexed statement text.
2. **`/analyze`** — a LangGraph agent that reasons over the question, selects from a toolkit of financial analysis functions (spending by category, cash flow, recurring transactions, anomaly detection, etc.), and synthesizes a final answer.

Anomaly detection and spending forecasts are also available as standalone ML-driven endpoints.

## Features

- **Multi-format statement parsing** — PDF (via PyMuPDF/pypdf), CSV, and Excel (XLSX/XLS), with currency, parentheses-as-negative, and date-format normalization handled per parser.
- **Rule-based + LLM-hybrid categorization** — transactions are first matched against an extensible set of regex category rules (income, housing, food, transport, subscriptions, etc.) with an LLM-based few-shot classifier available for ambiguous cases.
- **Domain-aware chunking** — a custom chunker that never splits a transaction across chunks, groups transactions by time period, and generates dedicated summary chunks for high-level statement queries.
- **Hybrid retrieval** — combines vector similarity search with BM25 keyword search so that exact amounts, dates, and merchant names are retrieved reliably alongside semantic matches.
- **Pluggable vector store** — Chroma for local development, Azure AI Search for production, behind a single `VectorStoreManager` interface.
- **LangGraph agent with 7 financial tools** — spending by category, spending over time, large-transaction detection, monthly summaries, recurring-transaction detection, cash flow analysis, and z-score anomaly detection, orchestrated in a retrieval → analyst → tools → synthesis graph.
- **ML-based anomaly detection** — Isolation Forest plus statistical z-score and transaction-velocity checks, with low/medium/high severity scoring.
- **Spending forecasts** — exponential smoothing and trend regression to project upcoming spending, income, and balance trends with confidence intervals.
- **Full observability** — Prometheus metrics, structured JSON logging via `structlog`, OpenTelemetry export, and LangSmith tracing/evaluation (faithfulness, relevance, groundedness, and numerical-accuracy checks).
- **Cloud-ready infrastructure** — Dockerfile, Docker Compose (API + Chroma + Prometheus + Grafana), Kubernetes manifests, and Terraform for Azure (AI Search, Storage, Cosmos DB, AKS).

## Architecture

```
                         ┌────────────────────┐
   Upload (PDF/CSV/XLSX) │   FastAPI Service   │
   ─────────────────────►│                     │
                         │  1. Parser          │
                         │  2. Normalizer      │
                         │  3. Chunker         │
                         │  4. Vector Indexer  │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │      Vector Store        │
                       │  Chroma (local) /         │
                       │  Azure AI Search (cloud)  │
                       └────────────┬────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                        │
   /query (RAG)                                            /analyze (Agent)
        │                                                        │
┌───────▼────────┐                              ┌────────────────▼───────────────┐
│ Hybrid Retriever│                              │       LangGraph Agent          │
│ (Vector + BM25) │                              │                                 │
│       │         │                              │  START → retrieval → analyst   │
│  RAG Chain (LLM)│                              │            ↕                   │
│  + citations    │                              │          tools                 │
└─────────────────┘                              │            ↓                   │
                                                  │        synthesis → END         │
                                                  └─────────────────────────────────┘
```

The agent's tool layer (`FinancialToolKit`) operates on a pandas DataFrame built from normalized transactions, while `/anomalies` and `/forecast` call the ML modules (`TransactionAnomalyDetector`, `SpendingForecaster`) directly for standalone analysis.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic v2 |
| LLM orchestration | LangChain, LangGraph, LangSmith |
| LLM / Embeddings | OpenAI (`gpt-4o`, `text-embedding-3-large`) / Azure OpenAI |
| Vector store | ChromaDB (local), Azure AI Search (cloud) |
| Document parsing | PyMuPDF, pypdf, pandas, openpyxl, python-docx, camelot-py, tabula-py |
| ML | scikit-learn (Isolation Forest), SciPy, statsmodels |
| Cloud | Azure Blob Storage, Cosmos DB, Azure AI Search, AKS |
| IaC | Terraform (`azurerm`), Kubernetes manifests |
| Observability | Prometheus, Grafana, OpenTelemetry, structlog, LangSmith |
| CI/CD | GitHub Actions (lint, test, security scan, build, deploy) |

## Project Structure

```
.
├── src/
│   ├── api/                  # FastAPI app, routes, request/response schemas
│   ├── agent/                 # LangGraph graph, nodes, and financial tool definitions
│   ├── document_processor/   # Parsers, normalizer/categorizer, financial-aware chunker
│   ├── ml/                   # Anomaly detection, forecasting, LLM-based categorizer
│   ├── rag/                  # Embeddings, vector store, hybrid retriever, RAG chain
│   ├── monitoring/           # Prometheus metrics, LangSmith evaluation
│   ├── config.py             # Pydantic settings (OpenAI, Azure, LangSmith, RAG, app)
│   └── main.py                # CLI / programmatic entry point (uvicorn launcher)
├── tests/                    # pytest suite (agents, parser, RAG pipeline)
├── infra/
│   ├── terraform/             # Azure infrastructure (AI Search, Storage, Cosmos, AKS)
│   ├── k8s/                  # Deployment, Service, Ingress manifests
│   └── prometheus.yml
├── .github/workflows/        # CI (lint/test/security) and CD (build/deploy) pipelines
├── Dockerfile
├── docker-compose.yml        # API + Chroma + Prometheus + Grafana
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (or Azure OpenAI credentials)
- Docker and Docker Compose (for the containerized setup)
- A LangSmith API key (optional, for tracing/evaluation)

### Installation

```bash
git clone https://github.com/umeshgehlot/Financial-Statement-Analysis-System.git
cd Financial-Statement-Analysis-System

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env .env.local   # the repo's .env ships as a template — populate it with real values
```

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) | — |
| `OPENAI_MODEL` | Chat model used for the agent and RAG chain | `gpt-4o` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model for vector indexing | `text-embedding-3-large` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `true` |
| `LANGCHAIN_API_KEY` | LangSmith API key | — |
| `LANGCHAIN_PROJECT` | LangSmith project name | `financial-statement-analyzer` |
| `AZURE_STORAGE_ACCOUNT_NAME` / `AZURE_STORAGE_CONTAINER_NAME` | Blob storage for uploaded statements (production) | — |
| `AZURE_COSMOS_ENDPOINT` / `AZURE_COSMOS_KEY` / `AZURE_COSMOS_DATABASE` | Cosmos DB for persisted transaction data (production) | — |
| `AZURE_SEARCH_ENDPOINT` / `AZURE_SEARCH_API_KEY` | Azure AI Search as the vector store (production) | — |
| `ENVIRONMENT` | `development` \| `staging` \| `production` | `development` |
| `API_HOST` / `API_PORT` | Bind address for the FastAPI server | `0.0.0.0` / `8000` |
| `MAX_UPLOAD_SIZE_MB` | Maximum statement upload size | `50` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | RAG chunking parameters | `1000` / `200` |
| `TOP_K_RESULTS` | Number of chunks retrieved per query | `5` |

By default the app runs against a local Chroma store; setting `AZURE_SEARCH_ENDPOINT` switches the vector store to Azure AI Search.

## Running the Service

**Locally:**

```bash
python -m src.main
# or
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**With Docker Compose** (API + Chroma + Prometheus + Grafana):

```bash
docker compose up --build
```

This starts the API on `http://localhost:8000`, Chroma on `8100`, Prometheus on `9090`, and Grafana on `3000`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, vector store connectivity, indexed document count |
| `POST` | `/upload` | Upload a statement (PDF/CSV/XLSX); parses, normalizes, chunks, and indexes it |
| `POST` | `/query` | Ask a natural-language question; answered via the RAG chain with cited sources |
| `POST` | `/analyze` | Run the LangGraph agent (`comprehensive`, `spending`, `anomaly`, `forecast`, or `cashflow` analysis types) |
| `POST` | `/anomalies` | Run standalone anomaly detection over uploaded transactions |
| `POST` | `/forecast` | Generate spending/income/balance forecasts for N upcoming periods |
| `GET` | `/transactions` | List processed transactions with pagination and category filtering |
| `GET` | `/metrics` | Prometheus-formatted metrics |

**Example: upload a statement**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@statement.pdf"
```

**Example: ask a question**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How much did I spend on dining out last month?"}'
```

**Example: run an agentic analysis**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize my cash flow and flag anything unusual", "analysis_type": "comprehensive"}'
```

## Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite covers the LangGraph toolkit (`test_agents.py`), the document parsing pipeline (`test_parser.py`), and the chunking/normalization/RAG components (`test_rag.py`).

Lint and type checks:

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
```

## Observability

- **Metrics** — `/metrics` exposes Prometheus counters and histograms for document processing stages, RAG query latency, and agent execution.
- **Dashboards** — Grafana (via Docker Compose) for visualizing the Prometheus metrics.
- **Tracing & evaluation** — LangSmith tracing is enabled by default (`LANGCHAIN_TRACING_V2=true`); `src/monitoring/langsmith_eval.py` implements automated faithfulness, relevance, groundedness, and financial-accuracy evaluations.
- **Structured logs** — JSON logs in non-development environments via `structlog`, console-rendered logs in development.

## Deployment

**Infrastructure (Terraform, Azure):**

```bash
cd infra/terraform
terraform init
terraform plan -var="environment=staging"
terraform apply -var="environment=staging"
```

Provisions Azure AI Search, Blob Storage, Cosmos DB, and supporting resources, with remote state in an Azure Storage backend.

**Kubernetes:**

```bash
kubectl apply -f infra/k8s/deployment.yaml
kubectl apply -f infra/k8s/service.yaml
kubectl apply -f infra/k8s/ingress.yaml
```

The deployment runs 2 replicas with a rolling-update strategy, Prometheus scrape annotations, and resource requests/limits suited to AKS.

## CI/CD

GitHub Actions pipelines in `.github/workflows/`:

- **`ci.yml`** — lints with Ruff, type-checks with mypy, runs the pytest suite with coverage against a Chroma service container, and runs a Snyk security scan.
- **`cd.yml`** — builds and pushes a Docker image to Azure Container Registry, deploys to an AKS staging cluster on pushes to `main`, runs a smoke test against `/health`, and promotes to production on version tags (`v*`).

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes, ensuring `ruff check`, `ruff format --check`, and `pytest` all pass.
3. Open a pull request against `main` describing the change and any relevant test coverage.
