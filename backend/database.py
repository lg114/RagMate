from functools import lru_cache

from sqlalchemy import create_engine, make_url
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


async def init_db():
    """初始化数据库：通过 Alembic 运行迁移（幂等，兼容全新和已有数据库）。"""
    import subprocess
    import sys
    from pathlib import Path

    alembic_ini = Path(__file__).parent / "alembic.ini"
    if alembic_ini.exists():
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(alembic_ini.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Alembic migration failed:\n{result.stderr}")
    else:
        # fallback：无 alembic.ini 时用 create_all（仅开发环境）
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
