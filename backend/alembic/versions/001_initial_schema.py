"""初始 schema。同时兼容全新安装和已有数据库。

全新安装：创建所有表。
已有数据库：检查表/列是否存在，仅补缺失部分。

Revision ID: 001
Revises: None
Create Date: 2025-01-01
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = inspector.get_table_names()

    # ── documents 表 ──
    if "documents" not in existing_tables:
        op.create_table(
            "documents",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("filename", sa.String(255), unique=True, nullable=False),
            sa.Column("size_bytes", sa.Integer(), server_default="0"),
            sa.Column("file_mtime", sa.Float(), nullable=True),
            sa.Column("status", sa.String(20), server_default="uploaded"),
            sa.Column("chunk_count", sa.Integer(), server_default="0"),
            sa.Column("uploaded_at", sa.DateTime(timezone=True)),
            sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=True),
        )
    else:
        # 补 file_mtime 列（旧数据库可能缺少）
        existing_cols = {c["name"] for c in inspector.get_columns("documents")}
        if "file_mtime" not in existing_cols:
            op.add_column("documents", sa.Column("file_mtime", sa.Float(), nullable=True))

    # ── chat_history 表 ──
    if "chat_history" not in existing_tables:
        op.create_table(
            "chat_history",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("session_id", sa.String(255), nullable=False),
            sa.Column("role", sa.String(20), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )
        op.create_index("idx_chat_session", "chat_history", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_table("chat_history")
    op.drop_table("documents")
