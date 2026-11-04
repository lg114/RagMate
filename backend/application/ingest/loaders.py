"""文档加载：根据文件类型直接解析，不经过 langchain-community。

PDF/DOCX/TXT 三处复刻原来三个 loader（PyPDFLoader / Docx2txtLoader /
TextLoader）的可观测行为：同样的 page_content 拆分与 metadata（source、
page），因为下游只消费这两项（其余 PDF 信息字典字段在入库时丢弃）。
这样主依赖不再需要 langchain-community（只为这三个薄 wrapper 而装）。
"""
import logging
import os

import docx2txt
import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

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


def _load_pdf(filepath: str) -> list[Document]:
    """逐页读 PDF：每页一个 Document，page_content 为当页提取文本 strip 后的结果。

    与 PyPDFLoader（page 模式、默认参数）一致：page 从 0 计，metadata 带
    source / page / total_pages / page_label。pypdf>=4 的 extract_text()
    默认即 plain 模式，与旧解析路径的调用等价。PDF 信息字典里的
    producer/creator 等字段不复刻——入库只存 source 与 page，无消费方。
    """
    reader = PdfReader(filepath)
    total_pages = len(reader.pages)
    labels = reader.page_labels
    docs = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(
            page_content=text.strip(),
            metadata={
                "source": filepath,
                "page": page_number,
                "total_pages": total_pages,
                "page_label": labels[page_number],
            },
        ))
    return docs


def _load_docx(filepath: str) -> list[Document]:
    """单个 Document：docx2txt 全文，metadata 只有 source（与 Docx2txtLoader 一致）。"""
    return [Document(page_content=docx2txt.process(filepath), metadata={"source": filepath})]


def _load_text(filepath: str) -> list[Document]:
    """单个 Document：utf-8 全文读入，metadata 只有 source（与 TextLoader 一致）。

    读取或解码失败时与 TextLoader 一样包成 RuntimeError，不透传原始异常类型。
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    return [Document(page_content=text, metadata={"source": filepath})]


def load_document(filepath: str, table_chunk_size: int = 1500) -> list[Document]:
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _load_pdf(filepath)
    elif ext == ".docx":
        return _load_docx(filepath)
    elif ext in (".xlsx", ".xls"):
        return load_excel(filepath, max_chunk_chars=table_chunk_size)
    elif ext in (".txt", ".md"):
        return _load_text(filepath)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []
