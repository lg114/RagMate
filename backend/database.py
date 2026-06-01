from functools import lru_cache

from sqlalchemy import create_engine, inspect, make_url, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

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
    )


class SyncSession:
    """惰性 sync session：首次使用时才创建 engine。"""

    def __enter__(self):
        self._session = sessionmaker(bind=get_sync_engine())()
        return self._session

    def __exit__(self, *args):
        self._session.close()


class Base(DeclarativeBase):
    pass


_MIGRATIONS = [
    # (table, column, type) — 每条是一次增量迁移
    ("documents", "file_mtime", "DOUBLE PRECISION"),
]


async def init_db():
    """初始化数据库：create_all 建表 + 增量迁移补列。"""
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
