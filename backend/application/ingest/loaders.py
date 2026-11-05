"""文档加载：根据文件类型直接解析，不经过 langchain-community。

PDF/DOCX/TXT 三处复刻原来三个 loader（PyPDFLoader / Docx2txtLoader /
TextLoader）的可观测行为：同样的 page_content 拆分与 metadata（source、
page），因为下游只消费这两项（其余 PDF 信息字典字段在入库时丢弃）。
这样主依赖不再需要 langchain-community（只为这三个薄 wrapper 而装）。
"""
import logging
import os

import docx2txt
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger("ragmate")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


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


def load_document(filepath: str) -> list[Document]:
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _load_pdf(filepath)
    elif ext == ".docx":
        return _load_docx(filepath)
    elif ext in (".txt", ".md"):
        return _load_text(filepath)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []
