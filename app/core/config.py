"""Settings, read from the environment (and a local .env file in development).

Nothing secret has a default. The app refuses to start in production without
the values it needs, instead of failing on the first request.
"""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AccessLevel(IntEnum):
    """Ordered clearance levels. A reader sees a document when their level is at least the document's."""

    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class Principal(BaseModel):
    """Who is calling, which departments they belong to, and their clearance."""

    model_config = ConfigDict(frozen=True)

    name: str
    departments: frozenset[str] = frozenset()
    clearance: AccessLevel = AccessLevel.PUBLIC
    can_ingest: bool = False

    @property
    def scope_key(self) -> str:
        """Stable key for anything that must never be shared across permission sets (the semantic cache)."""
        raw = f"{int(self.clearance)}|{','.join(sorted(self.departments))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_nested_delimiter="__")

    environment: Literal["development", "test", "production"] = "development"
    app_name: str = "enterprise-rag-engine"
    log_level: str = "INFO"

    # Databases
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    sql_tool_database_url: str | None = None  # read-only role for Text-to-SQL; falls back to database_url
    db_pool_size: int = 10
    db_max_overflow: int = 20
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None  # defaults to redis_url
    ingest_eager: bool = False  # run ingestion in-process (tests, single-node dev)
    upload_dir: str = "/tmp/rag-uploads"
    max_upload_mb: int = 25

    # LLM: any OpenAI-compatible endpoint (OpenAI, Groq, Ollama, vLLM...)
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"  # Groq free tier; Ollama: http://ollama:11434/v1
    llm_model: str = "openai/gpt-oss-120b"
    llm_grader_model: str | None = "openai/gpt-oss-20b"  # cheaper model for grading and routing
    llm_timeout_s: float = 60.0
    llm_temperature: float = 0.0
    # For reasoning models (gpt-oss, o-series): how much they think before answering.
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = "low"

    # Embeddings
    embedding_backend: Literal["sentence-transformers", "openai", "hashing"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    # Chunking
    parent_chunk_tokens: int = 1500
    child_chunk_tokens: int = 300
    child_chunk_overlap: int = 50

    # Retrieval
    dense_candidates: int = 30
    sparse_candidates: int = 30
    rrf_k: int = 60
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"  # or a local path to a fine-tuned model
    reranker_top_n: int = 8
    context_parents: int = 4
    min_rerank_score: float = 0.05

    # Agent
    max_rewrites: int = 2
    max_regenerations: int = 1
    history_turns: int = 6

    # Semantic cache
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_ttl_s: int = 86_400
    cache_max_entries_per_scope: int = 2_000

    # Security
    # JSON object: {"<key or sha256:digest>": {"name": "...", "departments": ["finance"],
    #                "clearance": "confidential", "can_ingest": true}}
    api_keys: SecretStr = SecretStr("{}")
    rate_limit_per_minute: int = 30

    # SQL tool
    sql_schema: str = "sales"
    sql_allowed_tables: list[str] = ["products", "customers", "orders", "order_items"]
    sql_max_rows: int = 200
    sql_timeout_ms: int = 3_000

    # Observability
    otel_exporter_otlp_endpoint: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    @field_validator("database_url", "sql_tool_database_url")
    @classmethod
    def _async_driver(cls, value: str | None) -> str | None:
        if value is None:
            return value
        for prefix in ("postgresql://", "postgres://", "postgresql+psycopg://"):
            if value.startswith(prefix):
                return "postgresql+asyncpg://" + value[len(prefix) :]
        return value

    @field_validator("cache_similarity_threshold")
    @classmethod
    def _threshold_range(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("cache_similarity_threshold must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def _production_requirements(self) -> Settings:
        if self.environment == "production":
            missing = []
            local = any(h in self.llm_base_url for h in ("localhost", "127.0.0.1", "ollama"))
            if self.llm_api_key is None and not local:
                missing.append("LLM_API_KEY")
            if not self.principals():
                missing.append("API_KEYS")
            if self.embedding_backend == "hashing":
                missing.append("a real EMBEDDING_BACKEND (hashing is for tests only)")
            if missing:
                raise ValueError("Missing production settings: " + ", ".join(missing))
        return self

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def grader_model(self) -> str:
        return self.llm_grader_model or self.llm_model

    def principals(self) -> dict[str, Principal]:
        """API keys indexed by their SHA-256 digest, so raw keys never sit in memory longer than parsing."""
        raw = json.loads(self.api_keys.get_secret_value() or "{}")
        table: dict[str, Principal] = {}
        for key, spec in raw.items():
            digest = key.removeprefix("sha256:") if key.startswith("sha256:") else hash_api_key(key)
            table[digest] = Principal(
                name=spec["name"],
                departments=frozenset(spec.get("departments", [])),
                clearance=AccessLevel[str(spec.get("clearance", "public")).upper()],
                can_ingest=bool(spec.get("can_ingest", False)),
            )
        return table


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@lru_cache
def get_settings() -> Settings:
    return Settings()
