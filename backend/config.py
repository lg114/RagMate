from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_API_BASE_URL: str = ""

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
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ragmate"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "ragmate_docs"
    MILVUS_TIMEOUT: float = 10.0

    # Ingestion
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Hybrid Search + Reranking
    HYBRID_SEARCH_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_CANDIDATES: int = 20       # rerank 候选池大小
    FINAL_CONTEXT_K: int = 4          # 最终给 LLM 的片段数
    RERANK_SCORE_THRESHOLD: float = 0.06  # 低于此分数的结果丢弃

    # Documents
    DOCUMENTS_DIR: str = "./documents"

    # CORS
    CORS_ORIGINS: str = "*"


settings = Settings()
