from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 文件路径：相对于 backend/ 目录（config.py 的祖父目录）
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # LLM
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_API_BASE_URL: str = ""
    LLM_TEMPERATURE: float = Field(default=0.01, ge=0.0, le=2.0)

    # Embedding
    EMBEDDING_PROVIDER: str = "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_BASE_URL: str = ""
    HF_TOKEN: str = ""

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = "default"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Database
    DATABASE_URL: str = ""
    REDIS_URL: str = "redis://localhost:6379/0"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = Field(default=19530, ge=1, le=65535)
    MILVUS_COLLECTION: str = "ragmate_docs"
    MILVUS_TIMEOUT: float = Field(default=10.0, gt=0)

    # Ingestion
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=200, ge=0)

    # Hybrid Search + Reranking
    HYBRID_SEARCH_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES: int = Field(default=30, gt=0)
    FINAL_CONTEXT_K: int = Field(default=15, gt=0)
    RERANK_SCORE_THRESHOLD: float = Field(default=0.3, gt=0.0, lt=1.0)
    DROP_RATIO_SEARCH: float = Field(default=0.2, ge=0.0, lt=1.0)

    # Reranker 动态过滤调参
    DYNAMIC_THRESHOLD_RATIO: float = Field(default=0.5, gt=0.0, le=1.0)  # top_score 乘数
    HIGH_SCORE_RATIO: float = Field(default=0.6, gt=0.0, le=1.0)         # 高分 chunk 判定比例
    MAX_PER_SOURCE: int = Field(default=4, gt=0)                          # 单源最大 chunk 数
    MIN_PER_SOURCE: int = Field(default=2, gt=0)                          # 单源最小 chunk 数
    SCORE_GAP_THRESHOLD: float = Field(default=0.15, gt=0.0, lt=1.0)     # 分数断崖阈值

    # Agent
    AGENT_RECURSION_LIMIT: int = Field(default=30, gt=0)

    # Documents
    DOCUMENTS_DIR: str = str(_BACKEND_DIR / "documents")

    # CORS
    CORS_ORIGINS: str = "http://localhost:8000,http://127.0.0.1:8000"

    @model_validator(mode="after")
    def _check_required(self):
        missing = []
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if not self.LLM_API_KEY:
            missing.append("LLM_API_KEY")
        if missing:
            raise ValueError(f"Required env vars not set: {', '.join(missing)}. Check your .env file.")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})")
        if self.MIN_PER_SOURCE > self.MAX_PER_SOURCE:
            raise ValueError(f"MIN_PER_SOURCE ({self.MIN_PER_SOURCE}) must be <= MAX_PER_SOURCE ({self.MAX_PER_SOURCE})")
        return self


settings = Settings()
