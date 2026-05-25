"""ingest 包：文档加载、编码、向量入库。

重新导出公共 API，保持向后兼容。
"""
from .db_sync import sync_documents_table
from .encoding import encode_documents, encode_query, get_bge_m3
from .loaders import SUPPORTED_EXTENSIONS, load_document
from .milvus_ops import build_source_filter
from .pipeline import ingest_documents

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "build_source_filter",
    "encode_documents",
    "encode_query",
    "get_bge_m3",
    "ingest_documents",
    "load_document",
    "sync_documents_table",
]
