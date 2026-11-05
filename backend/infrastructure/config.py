import re
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 文件路径：相对于 backend/ 目录（config.py 的祖父目录）
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"

# huggingface / local / flagembedding：本进程 FlagEmbedding。openai / remote：HTTP，不加载 BGE。
LOCAL_EMBEDDING_PROVIDERS = ("huggingface", "local", "flagembedding")
REMOTE_EMBEDDING_PROVIDERS = ("openai", "remote")
# 表名加上 _dense_hnsw 后仍须落在 Postgres 63 字符标识符限制内。
_VECTOR_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")


def resolve_vector_table(vector_table: str) -> str:
    """VECTOR_TABLE 为空则默认 ragmate_docs。"""
    table = (vector_table or "").strip() or "ragmate_docs"
    if not _VECTOR_TABLE_RE.fullmatch(table):
        raise ValueError(
            "VECTOR_TABLE must be a SQL identifier of at most 40 characters, "
            f"got {table!r}"
        )
    return table


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # LLM
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_API_BASE_URL: str = ""
    LLM_TEMPERATURE: float = Field(default=0.01, ge=0.0, le=2.0)
    # 思考型模型（本地 Qwen3 等）默认把推理过程写进 reasoning 字段、让 content 为空，
    # 导致上层拿到空回复。设为 "none" 可关闭思考。留空则不传该参数，
    # 对不支持它的服务商（如 OpenAI 官方）没有任何影响。
    LLM_REASONING_EFFORT: str = ""

    # Embedding
    # 默认 remote：API 进程不加载 torch / FlagEmbedding。本进程 BGE 需显式设为 huggingface、local 或 flagembedding。
    EMBEDDING_PROVIDER: str = "remote"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"  # 仅本地 FlagEmbedding：cpu 或 cuda
    EMBEDDING_NORMALIZE: bool = True
    EMBEDDING_API_KEY: str = ""  # 远程嵌入密钥，可选；自建服务不校验时留空
    EMBEDDING_API_BASE_URL: str = ""  # remote/openai 时必填
    EMBEDDING_API_TIMEOUT: float = Field(default=60.0, gt=0)  # 远程 embedding 请求超时（秒）
    HF_TOKEN: str = ""

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = "default"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # Database
    DATABASE_URL: str = ""
    # 留空则不连接 Redis。入库和会话都在 PostgreSQL；限流退回本进程内存。
    REDIS_URL: str = ""

    # pgvector 表，建在 DATABASE_URL 的同一个 Postgres 里。
    # 留空则用 ragmate_docs。
    VECTOR_TABLE: str = ""

    # Ingestion
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=200, ge=0)

    # Adaptive chunking per file type
    CHUNK_SIZE_PDF: int = Field(default=600, gt=0)
    CHUNK_OVERLAP_PDF: int = Field(default=100, ge=0)
    CHUNK_SIZE_DOCX: int = Field(default=800, gt=0)
    CHUNK_OVERLAP_DOCX: int = Field(default=150, ge=0)
    CHUNK_SIZE_TXT: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP_TXT: int = Field(default=200, ge=0)

    # Parent-child chunking (small-to-big)
    CHUNK_SIZE_PARENT: int = Field(default=2500, gt=0)
    CHUNK_OVERLAP_PARENT: int = Field(default=300, ge=0)

    # Chunking strategy for markdown. Frozen at title_path (130q fused baseline).
    #   title_path   — 按 CHUNK_SIZE 切子块，并把「H1 > H2 > H3」写入子块正文
    #   parent_child — 同切块边界，但不把标题路径写入检索正文，只放进 parent_text
    # parent_child 只是环境变量，不是第二套策略。改它之前重跑确定性基线。
    CHUNKING_STRATEGY: str = "title_path"

    # Hybrid Search + Reranking. Core stays hybrid + RRF + rerank. Do not add fusion arms.
    RRF_K: int = Field(default=60, ge=1, le=200)  # 冻结。RRF 常数；越小越偏向「两边都靠前」的交集
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # 默认 api：POST {RERANKER_API_BASE_URL}/rerank，不在本进程加载 CrossEncoder。
    RERANKER_PROVIDER: str = "api"
    RERANKER_API_BASE_URL: str = ""   # RERANKER_PROVIDER=api 时必填，如 https://api.jina.ai/v1
    RERANKER_API_KEY: str = ""        # 远程 rerank 接口的密钥（可选，部分自建服务不需要）
    RERANKER_API_TIMEOUT: float = Field(default=30.0, gt=0)  # 远程 rerank 请求超时（秒）
    RERANK_CANDIDATES: int = Field(default=30, gt=0)
    FINAL_CONTEXT_K: int = Field(default=15, gt=0)
    RERANK_SCORE_THRESHOLD: float = Field(default=0.3, gt=0.0, lt=1.0)

    # 忠诚度校验（默认关闭；开启后只在检索置信度低时跑：证据扎实时跳过这次 LLM 调用；失败时放行，不阻断回答）
    FAITHFULNESS_CHECK: bool = False

    # Query processing
    QUERY_CONTEXTUALIZE: bool = False  # 追问改写：先用原问题检索，置信度低时才用 LLM 改写为自包含 query 再检一次；false 则从不改写
    # 冻结为关。多查询扩展不是默认；打开前先重跑 ragmate-eval retrieval 并更新门槛。
    QUERY_ROUTING_ENABLED: bool = False  # 简单查询跳过检索，直接回复寒暄


    # 超时
    AGENT_TIMEOUT: int = Field(default=120, gt=0)  # 检索/回答调用超时（秒）

    # Chat & Session
    MAX_HISTORY_TURNS: int = Field(default=10, gt=0)  # 上下文窗口保留最近 N 轮对话
    MAX_SESSION_MESSAGES: int = Field(default=200, gt=0)  # 单 session 最大消息数
    HEARTBEAT_INTERVAL: int = Field(default=15, gt=0)  # SSE 心跳间隔（秒）

    # Auth（可选）：为空时不鉴权（默认，单机/内网部署无感）；设置后见 app.py 的 _require_api_token
    API_AUTH_TOKEN: str = ""

    # Rate Limiting
    RATE_LIMIT_MAX: int = Field(default=30, gt=0)  # 每窗口最大请求数
    RATE_LIMIT_WINDOW: int = Field(default=60, gt=0)  # 限流窗口（秒）

    # Upload
    MAX_UPLOAD_SIZE: int = Field(default=50 * 1024 * 1024, gt=0)  # 上传文件大小上限（字节）

    # Ingest
    INGEST_LOCK_TTL: int = Field(default=600, gt=0)  # 入库锁租约（秒），由心跳续期；过期后其他 worker 可以抢走
    INGEST_HEARTBEAT_INTERVAL: int = Field(default=5, gt=0)  # 续期锁、刷新任务心跳、轮询取消（秒）
    INGEST_HEARTBEAT_STALE: int = Field(default=30, gt=0)  # 心跳超过该秒数视为持锁 worker 已退出

    # Documents
    DOCUMENTS_DIR: str = str(_BACKEND_DIR / "documents")

    # CORS
    CORS_ORIGINS: str = "http://localhost:8000,http://127.0.0.1:8000"

    @model_validator(mode="after")
    def _check_required(self):
        object.__setattr__(
            self, "VECTOR_TABLE", resolve_vector_table(self.VECTOR_TABLE)
        )
        if self.RERANKER_PROVIDER not in ("local", "api"):
            raise ValueError(f"RERANKER_PROVIDER must be 'local' or 'api', got '{self.RERANKER_PROVIDER}'")
        if self.EMBEDDING_PROVIDER not in LOCAL_EMBEDDING_PROVIDERS + REMOTE_EMBEDDING_PROVIDERS:
            raise ValueError(
                "EMBEDDING_PROVIDER must be huggingface, local, flagembedding, openai, or remote, "
                f"got '{self.EMBEDDING_PROVIDER}'"
            )

        missing = []
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if not self.LLM_API_KEY:
            missing.append("LLM_API_KEY")
        # 默认就是远程。URL 缺失时启动失败，避免进程悄悄去加载本地 torch / BGE。
        # 密钥保持可选：很多 OpenAI 兼容侧车不校验 Authorization。
        if self.EMBEDDING_PROVIDER in REMOTE_EMBEDDING_PROVIDERS and not self.EMBEDDING_API_BASE_URL.strip():
            missing.append("EMBEDDING_API_BASE_URL")
        if self.RERANKER_PROVIDER == "api" and not self.RERANKER_API_BASE_URL.strip():
            missing.append("RERANKER_API_BASE_URL")
        if missing:
            raise ValueError(
                f"Required env vars not set: {', '.join(missing)}. Check your .env file. "
                "EMBEDDING_API_BASE_URL is required when EMBEDDING_PROVIDER is openai or remote (the default). "
                "RERANKER_API_BASE_URL is required when RERANKER_PROVIDER=api (the default). "
                "Keys for those endpoints are optional. "
                "To load FlagEmbedding and the cross-encoder in this process instead, set "
                "EMBEDDING_PROVIDER=huggingface (local and flagembedding are aliases) and RERANKER_PROVIDER=local."
            )

        # 验证所有 chunk size 配置
        chunk_configs = [
            ("CHUNK_SIZE", "CHUNK_OVERLAP"),
            ("CHUNK_SIZE_PDF", "CHUNK_OVERLAP_PDF"),
            ("CHUNK_SIZE_DOCX", "CHUNK_OVERLAP_DOCX"),
            ("CHUNK_SIZE_TXT", "CHUNK_OVERLAP_TXT"),
            ("CHUNK_SIZE_PARENT", "CHUNK_OVERLAP_PARENT"),
        ]
        
        for size_name, overlap_name in chunk_configs:
            size = getattr(self, size_name)
            overlap = getattr(self, overlap_name)
            if overlap >= size:
                raise ValueError(
                    f"{overlap_name} ({overlap}) must be less than {size_name} ({size})"
                )
        
        if self.INGEST_HEARTBEAT_STALE <= self.INGEST_HEARTBEAT_INTERVAL:
            raise ValueError(
                f"INGEST_HEARTBEAT_STALE ({self.INGEST_HEARTBEAT_STALE}) must be greater than "
                f"INGEST_HEARTBEAT_INTERVAL ({self.INGEST_HEARTBEAT_INTERVAL})"
            )
        if self.INGEST_HEARTBEAT_STALE >= self.INGEST_LOCK_TTL:
            raise ValueError(
                f"INGEST_HEARTBEAT_STALE ({self.INGEST_HEARTBEAT_STALE}) must be less than "
                f"INGEST_LOCK_TTL ({self.INGEST_LOCK_TTL})"
            )
        return self


settings = Settings()
