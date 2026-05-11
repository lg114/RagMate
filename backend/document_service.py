import os
from datetime import datetime, timezone
from pathlib import Path

from pymilvus import MilvusClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from errors import ConflictError, NotFoundError, ValidationError
from models import Document
from redis_client import get_redis

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

_PATH_SEPARATORS = {os.sep}
if os.altsep:
    _PATH_SEPARATORS.add(os.altsep)


def validate_filename(filename: str) -> str:
    """校验文件名，拒绝路径穿越等非法输入。返回安全的纯文件名。"""
    if not filename:
        raise ValidationError("Filename is required")

    # 拒绝包含路径分隔符的输入（如 ../../../etc/passwd）
    if any(sep in filename for sep in _PATH_SEPARATORS):
        raise ValidationError("Invalid filename")

    # Path().name 比 os.path.basename 跨平台一致性更好
    name = Path(filename).name
    if name != filename:
        raise ValidationError("Invalid filename")

    from ingest import SUPPORTED_EXTENSIONS
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    return name


async def list_documents(docs_dir: str, session: AsyncSession) -> list[dict]:
    """从 PostgreSQL 查询文档列表，磁盘信息作为补充。"""
    result = await session.execute(select(Document).order_by(Document.uploaded_at.desc()))
    docs = result.scalars().all()

    documents = []
    for doc in docs:
        filepath = os.path.join(docs_dir, doc.filename)
        path_exists = os.path.exists(filepath)
        if path_exists:
            size_bytes = os.path.getsize(filepath)
            modified_at = datetime.fromtimestamp(os.path.getmtime(filepath), tz=timezone.utc)
        else:
            size_bytes = doc.size_bytes
            modified_at = doc.uploaded_at or datetime.now(timezone.utc)

        documents.append({
            "filename": doc.filename,
            "size_bytes": size_bytes,
            "status": doc.status,
            "chunk_count": doc.chunk_count,
            "uploaded_at": doc.uploaded_at or modified_at,
            "ingested_at": doc.ingested_at,
            "exists_on_disk": path_exists,
        })

    return documents


async def save_document(
    filename: str,
    content: bytes,
    docs_dir: str,
    session: AsyncSession,
) -> dict:
    """将上传的文件写入磁盘并记录到 PostgreSQL。PostgreSQL 为权威数据源。"""
    name = validate_filename(filename)

    # 1. 先查 DB，PostgreSQL 是 authoritative
    result = await session.execute(select(Document).where(Document.filename == name))
    existing = result.scalar_one_or_none()
    if existing:
        raise ConflictError(f"File '{name}' already exists")

    size_bytes = len(content)
    if size_bytes > MAX_FILE_SIZE:
        raise ValidationError("File exceeds 50MB limit")

    # 2. 写磁盘
    os.makedirs(docs_dir, exist_ok=True)
    filepath = os.path.join(docs_dir, name)

    with open(filepath, "wb") as f:
        f.write(content)

    # 3. 写 DB，失败则回滚并清理磁盘文件
    try:
        doc = Document(filename=name, size_bytes=size_bytes, status="uploaded")
        session.add(doc)
        await session.commit()
    except Exception:
        await session.rollback()
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

    return {
        "filename": name,
        "size_bytes": size_bytes,
        "status": "uploaded",
        "uploaded_at": datetime.now(timezone.utc),
    }


async def delete_document(
    filename: str,
    docs_dir: str,
    session: AsyncSession,
    milvus_client: MilvusClient | None = None,
) -> None:
    """删除文档：PostgreSQL 记录为主，磁盘文件和 Milvus 作为辅助清理。"""
    name = validate_filename(filename)

    # 1. 先查 DB
    result = await session.execute(select(Document).where(Document.filename == name))
    doc = result.scalar_one_or_none()
    if not doc:
        raise NotFoundError("Document", name)

    filepath = os.path.join(docs_dir, name)

    # 2. 删 DB 记录（先删 DB，失败了就不动磁盘）
    await session.execute(delete(Document).where(Document.filename == name))
    await session.commit()

    # 3. 清理磁盘文件和 Milvus 向量（DB 已删，这些失败不影响业务）
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

    if milvus_client and milvus_client.has_collection(settings.MILVUS_COLLECTION):
        try:
            escaped_name = name.replace('"', '\\"')
            milvus_client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=f'metadata["source"] == "{escaped_name}"',
            )
        except Exception:
            pass

    # 4. 清理 Redis 会话缓存（不影响主流程，失败忽略）
    try:
        r = await get_redis()
        await r.delete(f"ragmate:session:{name}")
    except Exception:
        pass
