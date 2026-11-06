"""文档加载：根据文件类型直接解析，不经过 langchain-community。

PDF/DOCX/TXT 三处复刻原来三个 loader（PyPDFLoader / Docx2txtLoader /
TextLoader）的可观测行为：同样的 page_content 拆分与 metadata（source、
page），因为下游只消费这两项（其余 PDF 信息字典字段在入库时丢弃）。
这样主依赖不再需要 langchain-community（只为这三个薄 wrapper 而装）。
"""
import csv
import io
import json
import logging
import os
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger("ragmate")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".pptx", ".xlsx", ".csv", ".html", ".htm",
    ".rtf", ".json", ".xml", ".epub",
}


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
    import docx2txt
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


def _load_pptx(filepath: str) -> list[Document]:
    """逐页读 PPTX：每页幻灯片一个 Document，提取所有文本框内容。"""
    from pptx import Presentation
    prs = Presentation(filepath)
    docs = []
    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            docs.append(Document(
                page_content="\n".join(texts),
                metadata={"source": filepath, "slide": slide_num},
            ))
    return docs


def _load_xlsx(filepath: str) -> list[Document]:
    """读 XLSX：每个工作表一个 Document，将表格转为 CSV 格式文本。"""
    from openpyxl import load_workbook
    wb = load_workbook(filepath, read_only=True, data_only=True)
    docs = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(cell) if cell is not None else "" for cell in row]
            if any(cells):
                rows.append(",".join(cells))
        if rows:
            docs.append(Document(
                page_content="\n".join(rows),
                metadata={"source": filepath, "sheet": sheet_name},
            ))
    wb.close()
    return docs


def _load_csv(filepath: str) -> list[Document]:
    """读 CSV：整个文件一个 Document，保留表头。"""
    try:
        with open(filepath, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    if not rows:
        return []
    text = "\n".join(",".join(row) for row in rows)
    return [Document(page_content=text, metadata={"source": filepath})]


class _HTMLTextExtractor(HTMLParser):
    """从 HTML 提取纯文本。"""

    def __init__(self):
        super().__init__()
        self.result = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self.result.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.result.append(data)

    def get_text(self):
        return "".join(self.result).strip()


def _load_html(filepath: str) -> list[Document]:
    """读 HTML：提取纯文本，单个 Document。"""
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    extractor = _HTMLTextExtractor()
    extractor.feed(content)
    text = extractor.get_text()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": filepath})]


def _load_rtf(filepath: str) -> list[Document]:
    """读 RTF：提取纯文本。"""
    from striprtf.striprtf import rtf_to_text
    try:
        with open(filepath, encoding="utf-8") as f:
            rtf_content = f.read()
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    text = rtf_to_text(rtf_content)
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": filepath})]


def _load_json(filepath: str) -> list[Document]:
    """读 JSON：格式化为可读文本，单个 Document。"""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return [Document(page_content=text, metadata={"source": filepath})]


def _load_xml(filepath: str) -> list[Document]:
    """读 XML：提取所有文本内容，单个 Document。"""
    try:
        tree = ET.parse(filepath)
    except Exception as e:
        raise RuntimeError(f"Error loading {filepath}") from e
    root = tree.getroot()
    texts = []

    def extract_text(elem):
        if elem.text:
            t = elem.text.strip()
            if t:
                texts.append(t)
        for child in elem:
            extract_text(child)
        if elem.tail:
            t = elem.tail.strip()
            if t:
                texts.append(t)

    extract_text(root)
    if not texts:
        return []
    return [Document(page_content="\n".join(texts), metadata={"source": filepath})]


def _load_epub(filepath: str) -> list[Document]:
    """读 EPUB：逐章节提取文本。"""
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(filepath)
    docs = []
    for item in book.get_items_of_type(9):  # ITEM_DOCUMENT
        html_content = item.get_content().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        if text.strip():
            docs.append(Document(
                page_content=text.strip(),
                metadata={"source": filepath, "chapter": item.get_name()},
            ))
    return docs


def load_document(filepath: str) -> list[Document]:
    """根据文件扩展名选择合适的 loader 加载文档。"""
    ext = os.path.splitext(filepath)[1].lower()

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_text,
        ".pptx": _load_pptx,
        ".xlsx": _load_xlsx,
        ".csv": _load_csv,
        ".html": _load_html,
        ".htm": _load_html,
        ".rtf": _load_rtf,
        ".json": _load_json,
        ".xml": _load_xml,
        ".epub": _load_epub,
    }

    loader = loaders.get(ext)
    if loader:
        return loader(filepath)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return []
