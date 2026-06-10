"""文档加载：根据文件类型选择合适的 loader。"""
import logging
import os

import pandas as pd
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger("ragmate")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md"}


def _df_to_markdown(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 markdown 表格（无需 tabulate 依赖）。"""
    headers = [str(h).replace("|", "\\|").replace("\n", " ") for h in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        cells = [str(v).replace("|", "\\|").replace("\n", " ") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _split_markdown_table(md_table: str, max_chars: int) -> list[str]:
    """将超长 markdown 表格按行拆分，每块都带表头行。"""
    lines = md_table.strip().split("\n")
    if len(lines) < 3:
        return [md_table]

    header = lines[0]
    separator = lines[1]
    data_lines = lines[2:]
    base_size = len(header) + len(separator) + 2

    chunks = []
    current_lines = []
    current_size = base_size

    for line in data_lines:
        line_size = len(line) + 1
        if current_lines and current_size + line_size > max_chars:
            chunks.append("\n".join([header, separator] + current_lines))
            current_lines = []
            current_size = base_size
        current_lines.append(line)
        current_size += line_size

    if current_lines:
        chunks.append("\n".join([header, separator] + current_lines))

    return chunks


def load_excel(filepath: str, max_chunk_chars: int = 1500) -> list[Document]:
    """读取 Excel，每个 sheet 转为 markdown 表格。大表按行拆分，保留表头。"""
    sheets = pd.read_excel(filepath, sheet_name=None, dtype=str)
    basename = os.path.basename(filepath)
    pages = []

    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df = df.fillna("")
        md_table = _df_to_markdown(df)

        if len(md_table) <= max_chunk_chars:
            pages.append(Document(
                page_content=md_table,
                metadata={"source": basename, "page": 1, "sheet": sheet_name},
            ))
        else:
            parts = _split_markdown_table(md_table, max_chunk_chars)
            for i, part in enumerate(parts):
                pages.append(Document(
                    page_content=part,
                    metadata={"source": basename, "page": 1, "sheet": sheet_name, "table_part": i + 1},
                ))

    return pages


def load_document(filepath: str, table_chunk_size: int = 1500) -> list[Document]:
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(filepath).load()
    elif ext == ".docx":
        return Docx2txtLoader(filepath).load()
    elif ext in (".xlsx", ".xls"):
        return load_excel(filepath, max_chunk_chars=table_chunk_size)
    elif ext in (".txt", ".md"):
        return TextLoader(filepath, encoding="utf-8").load()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []
