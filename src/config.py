# src/config.py
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class OpenAISettings(BaseSettings):
    api_key: str = Field(..., alias="OPENAI_API_KEY")
    model: str = Field("gpt-4o", alias="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    temperature: float = Field(0.1, alias="OPENAI_TEMPERATURE")
    max_tokens: int = Field(4096, alias="OPENAI_MAX_TOKENS")

    model_config = {"env_prefix": "", "extra": "ignore"}


class LangSmithSettings(BaseSettings):
    tracing_v2: bool = Field(True, alias="LANGCHAIN_TRACING_V2")
    api_key: str = Field(..., alias="LANGCHAIN_API_KEY")
    project: str = Field("financial-statement-analyzer", alias="LANGCHAIN_PROJECT")
    endpoint: str = Field(
        "https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )

    model_config = {"env_prefix": "", "extra": "ignore"}


class AzureSettings(BaseSettings):
    tenant_id: str = Field("", alias="AZURE_TENANT_ID")
    client_id: str = Field("", alias="AZURE_CLIENT_ID")
    client_secret: str = Field("", alias="AZURE_CLIENT_SECRET")
    storage_account_name: str = Field("", alias="AZURE_STORAGE_ACCOUNT_NAME")
    storage_container_name: str = Field(
        "bank-statements", alias="AZURE_STORAGE_CONTAINER_NAME"
    )
    cosmos_endpoint: str = Field("", alias="AZURE_COSMOS_ENDPOINT")
    cosmos_key: str = Field("", alias="AZURE_COSMOS_KEY")
    cosmos_database: str = Field("financial-db", alias="AZURE_COSMOS_DATABASE")
    search_endpoint: str = Field("", alias="AZURE_SEARCH_ENDPOINT")
    search_api_key: str = Field("", alias="AZURE_SEARCH_API_KEY")

    model_config = {"env_prefix": "", "extra": "ignore"}


class RAGSettings(BaseSettings):
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(5, alias="TOP_K_RESULTS")
    collection_name: str = "financial_documents"
    use_azure_search: bool = False

    model_config = {"env_prefix": "", "extra": "ignore"}


class AppSettings(BaseSettings):
    environment: Environment = Field(Environment.DEVELOPMENT, alias="ENVIRONMENT")
    log_level: str = Field("DEBUG", alias="LOG_LEVEL")
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    max_upload_size_mb: int = Field(50, alias="MAX_UPLOAD_SIZE_MB")

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)

    model_config = {"env_prefix": "", "extra": "ignore"}


@lru_cache
def get_settings() -> AppSettings:
    from dotenv import load_dotenv

    load_dotenv()
    return AppSettings()