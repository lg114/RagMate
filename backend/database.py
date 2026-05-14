from sqlalchemy import create_engine, make_url, text
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
_sync_engine = None


def get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            _sync_url,
            echo=False,
            pool_size=2,
            max_overflow=2,
            pool_pre_ping=True,
        )
    return _sync_engine


class SyncSession:
    """惰性 sync session：首次使用时才创建 engine。"""

    def __enter__(self):
        self._session = sessionmaker(bind=get_sync_engine())()
        return self._session

    def __exit__(self, *args):
        self._session.close()


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 增量迁移：补 file_mtime 列（旧数据库没有这列）
        await conn.execute(text(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_mtime DOUBLE PRECISION"
        ))
