from functools import lru_cache

from sqlalchemy import create_engine, inspect, insert, make_url, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.infrastructure.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    pool_pre_ping=True,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# 从 async URL 生成 sync URL：替换 driver 从 asyncpg 到 psycopg2
_sync_url = make_url(settings.DATABASE_URL).set(drivername="postgresql+psycopg2")


@lru_cache(maxsize=1)
def get_sync_engine():
    return create_engine(
        _sync_url,
        echo=False,
        pool_size=2,
        max_overflow=2,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 1},
    )


class SyncSession:
    """惰性 sync session：首次使用时才创建 engine。"""

    def __enter__(self):
        self._session = sessionmaker(bind=get_sync_engine())()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._session.rollback()
        self._session.close()


class Base(DeclarativeBase):
    pass


_MIGRATIONS = [
    # (table, column, type) — 每条是一次增量迁移
    ("documents", "file_mtime", "DOUBLE PRECISION"),
]


def _ensure_ingest_row(sync_conn):
    """保证 ingest_control 有且仅有 id=1 的控制行。终态留在这行上，不按 TTL 删除。"""
    from backend.domain.models import IngestControl

    found = sync_conn.execute(select(IngestControl.id).where(IngestControl.id == 1)).first()
    if found:
        return
    sync_conn.execute(
        insert(IngestControl).values(
            id=1,
            status="idle",
            detail={"status": "idle", "last_ingest": None},
            cancel_requested=False,
        )
    )


async def init_db():
    """初始化数据库：create_all 建表 + 增量迁移补列。"""
    import backend.domain.models  # noqa: F401 — 注册 ORM 表，含 ingest_control

    async with engine.begin() as conn:
        # 1. create_all：已存在的表自动跳过，只建新表
        await conn.run_sync(Base.metadata.create_all)

        # 2. 增量迁移：用 inspect 检查列是否存在，不存在则添加
        def _run_migrations(sync_conn):
            existing = {}
            insp = inspect(sync_conn)
            for table, _, _ in _MIGRATIONS:
                if table not in existing:
                    try:
                        existing[table] = {c["name"] for c in insp.get_columns(table)}
                    except Exception:
                        existing[table] = set()
            for table, column, col_type in _MIGRATIONS:
                if column not in existing.get(table, set()):
                    sync_conn.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                    ))

        await conn.run_sync(_run_migrations)
        await conn.run_sync(_ensure_ingest_row)

        def _ensure_vectors(sync_conn):
            from backend.infrastructure.vector_store import ensure_schema_conn

            ensure_schema_conn(sync_conn)

        await conn.run_sync(_ensure_vectors)

def ensure_ingest_schema_sync() -> None:
    """同步建表：ORM 表 + ingest 控制行 + 向量表/扩展。

    直接调 pipeline 入库时用（API 启动时 init_db 已做过，重复调用无害）。
    没有它，新库直接入库会在两处失败：可用性检查要求 vector 扩展已存在，
    而 _detect_new_files 要查的 documents 表也还不存在。
    """
    import backend.domain.models  # noqa: F401 — 注册 ORM 表

    from backend.infrastructure.vector_store import get_vector_store

    engine = get_sync_engine()
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        _ensure_ingest_row(conn)
    get_vector_store().ensure_schema()
