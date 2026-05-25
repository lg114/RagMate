"""PostgreSQL 文档状态同步。"""
import datetime
import logging
import os

from sqlalchemy import select

from database import SyncSession
from models import Document

logger = logging.getLogger("ragmate")


def sync_documents_table(directory: str, filenames: list[str], chunk_counts: dict[str, int]):
    """同步 PostgreSQL documents 表：将本次入库的文件标记为 ingested。已有记录则更新，没有则自动创建"""
    now = datetime.datetime.now(datetime.timezone.utc)
    with SyncSession() as session:
        result = session.execute(
            select(Document).where(Document.filename.in_(filenames))
        )
        existing = {doc.filename: doc for doc in result.scalars().all()}

        for filename in filenames:
            doc = existing.get(filename)
            filepath = os.path.join(directory, filename)
            try:
                st = os.stat(filepath)
                size_bytes, mtime = st.st_size, st.st_mtime
            except OSError:
                size_bytes, mtime = 0, None
            if doc:
                doc.status = "ingested"
                doc.chunk_count = chunk_counts.get(filename, 0)
                doc.size_bytes = size_bytes
                doc.file_mtime = mtime
                doc.ingested_at = now
            else:
                doc = Document(
                    filename=filename,
                    size_bytes=size_bytes,
                    file_mtime=mtime,
                    status="ingested",
                    chunk_count=chunk_counts.get(filename, 0),
                    uploaded_at=now,
                    ingested_at=now,
                )
                session.add(doc)
        session.commit()
