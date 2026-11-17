"""
Central configuration.

Every value can be overridden with an environment variable or a `.env` file.
The defaults are chosen so that `pip install` + `uvicorn` works with SQLite and
local embeddings, without any API key.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Configurations shipped with PostgreSQL. Restricting the list keeps the value
# safe to interpolate into a to_tsvector(...) call.
_ALLOWED_FTS_LANGUAGES = {
    "simple",
    "english",
    "french",
    "german",
    "spanish",
    "italian",
    "portuguese",
    "dutch",
    "russian",
    "romanian",
    "arabic",
    "swedish",
    "danish",
    "finnish",
    "norwegian",
    "hungarian",
    "turkish",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Server ---
    APP_ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: str = "*"
    RAG_API_KEY: str = ""

    # --- Database ---
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/rag.db"
    AUTO_CREATE_TABLES: bool = True

    # --- Files ---
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_MB: int = 100

    # --- Generation LLM ---
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1200
    LLM_TIMEOUT_SECONDS: float = 90.0

    # --- Embeddings ---
    EMBEDDING_PROVIDER: str = "local"  # local | openai
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_CACHE_DIR: str = "./data/models"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DOCUMENT_PREFIX: str = ""
    EMBEDDING_QUERY_PREFIX: str = ""

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 2000

    # --- Retrieval ---
    TOP_K: int = 5
    HYBRID_SEARCH: bool = True
    CANDIDATES_PER_MODE: int = 20
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 10
    PARENT_CONTEXT_ENABLED: bool = True
    FTS_LANGUAGE: str = "simple"

    # ---------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------
    @property
    def embedding_provider(self) -> str:
        return self.EMBEDDING_PROVIDER.strip().lower()

    @property
    def fts_language(self) -> str:
        value = self.FTS_LANGUAGE.strip().lower()
        return value if value in _ALLOWED_FTS_LANGUAGES else "simple"

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.lower().startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.DATABASE_URL.lower().startswith("postgresql")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        # Remote OpenAI needs a key. Any other base URL is assumed to be a
        # local server or a gateway that may accept an empty key (Ollama,
        # LM Studio, vLLM, an internal proxy...). If it does require a key,
        # the provider returns a clear 401.
        return bool(self.LLM_MODEL.strip()) and (
            bool(self.LLM_API_KEY.strip())
            or self.LLM_BASE_URL.strip().rstrip("/") != "https://api.openai.com/v1"
        )

    def cors_origins(self) -> list[str]:
        raw = self.CORS_ORIGINS.strip()
        if not raw or raw == "*":
            return ["*"]
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def ensure_directories() -> None:
    """Create the data directories before the engine touches SQLite or uploads."""
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    if settings.embedding_provider == "local":
        Path(settings.EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    # SQLite creates the file on connect, but not its parent directory.
    if settings.is_sqlite:
        database = settings.DATABASE_URL.split(":///", 1)[-1]
        if database and database != ":memory:":
            Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
