import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pymilvus import MilvusClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.infrastructure.config import settings
from backend.domain.errors import NotFoundError, ValidationError
from backend.domain.models import Document
from backend.infrastructure.milvus import build_source_filter

logger = logging.getLogger("ragmate")

# Windows 保留文件名（不区分大小写）
_WIN_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB 文件大小上限

_PATH_SEPARATORS = {os.sep}
if os.altsep:
    _PATH_SEPARATORS.add(os.altsep)

# 文件头 magic bytes 映射
_MAGIC_BYTES = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",  # ZIP 格式
    ".doc": b"\xd0\xcf\x11\xe0",  # OLE2 格式
    ".xlsx": b"PK\x03\x04",  # ZIP 格式
    ".xls": b"\xd0\xcf\x11\xe0",  # OLE2 格式
}


def _validate_magic_bytes(filename: str, content: bytes):
    """校验文件头 magic bytes 是否与扩展名匹配。"""
    ext = os.path.splitext(filename)[1].lower()
    expected = _MAGIC_BYTES.get(ext)
    if expected and not content.startswith(expected):
        raise ValidationError(f"File content does not match extension {ext}")


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

    from backend.application.ingest.loaders import SUPPORTED_EXTENSIONS
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    # 白名单：字母数字、中文、连字符、下划线、点、空格、&、()、+、=、@
    if not re.match(r'^[\w\-. 一-鿿&()+=@]+\.\w+$', name):
        raise ValidationError("Filename contains invalid characters")

    # 拒绝 Windows 保留文件名
    stem = os.path.splitext(name)[0].upper()
    if stem in _WIN_RESERVED:
        raise ValidationError(f"Filename '{stem}' is a reserved system name")

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

    # Magic bytes 校验：验证文件内容与扩展名匹配
    _validate_magic_bytes(name, content)

    # 1. 先查 DB，PostgreSQL 是 authoritative
    result = await session.execute(select(Document).where(Document.filename == name))
    existing = result.scalar_one_or_none()
    if existing:
        raise ValidationError(f"File '{name}' already exists", status_code=409)

    size_bytes = len(content)
    if size_bytes > MAX_FILE_SIZE:
        raise ValidationError("File exceeds 50MB limit")

    # 2. 写临时文件（不直接写目标路径，避免崩溃后产生与 DB 不一致的孤立文件）
    os.makedirs(docs_dir, exist_ok=True)
    filepath = os.path.join(docs_dir, name)
    tmp_path = filepath + ".tmp"

    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        # 3. 写 DB
        doc = Document(filename=name, size_bytes=size_bytes, status="uploaded")
        session.add(doc)
        await session.commit()
    except Exception:
        await session.rollback()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    # 4. DB 提交成功后原子重命名（同文件系统上 rename 是原子操作）
    os.replace(tmp_path, filepath)

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
            logger.warning(f"Failed to delete file {filepath}", exc_info=True)

    if milvus_client and milvus_client.has_collection(settings.MILVUS_COLLECTION):
        try:
            milvus_client.delete(
                collection_name=settings.MILVUS_COLLECTION,
                filter=build_source_filter(name),
            )
        except Exception:
            logger.warning(f"Failed to delete Milvus vectors for {name}", exc_info=True)

