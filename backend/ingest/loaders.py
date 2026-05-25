"""文档加载：根据文件类型选择合适的 loader。"""
import logging
import os

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

logger = logging.getLogger("ragmate")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}


def load_document(filepath: str):
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(filepath).load()
    elif ext == ".docx":
        return Docx2txtLoader(filepath).load()
    elif ext in (".xlsx", ".xls"):
        return UnstructuredExcelLoader(filepath).load()
    elif ext in (".txt", ".md"):
        return TextLoader(filepath, encoding="utf-8").load()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []
