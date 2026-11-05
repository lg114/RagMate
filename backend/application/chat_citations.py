"""引用整理与接地：把检索引用写进回答、按置信度拒答/保留。

纯文本函数，无 I/O。chat.py 会重新导出这里的名字（测试直接从
backend.application.chat 导入它们）。
"""

import re

_CITATION_RE = re.compile(r"【([^】]+\.(?:pdf|docx?|xlsx?|xls|txt|md))(?:[,，]\s*第\d+页)?】", re.IGNORECASE)


_SOURCE_LINE_RE = re.compile(r"^\s*数据来源[:：].*$", re.MULTILINE)


def normalize_citations(text: str) -> str:
    """Move repeated file citations into one source line for readability."""
    matches = _CITATION_RE.findall(text)
    if len(matches) < 2:
        return text
    sources = list(dict.fromkeys(matches))

    cleaned = _SOURCE_LINE_RE.sub("", text)
    for source in sources:
        # 移除所有匹配该文件名的引用（可能带页码）
        cleaned = re.sub(r"【" + re.escape(source) + r"(?:[,，]\s*第\d+页)?】", "", cleaned)

    cleaned = re.sub(r"[ \t]+([，。；：、,.!?！？])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    source_line = "数据来源：" + "、".join(f"【{source}】" for source in sources)
    return f"{cleaned}\n\n{source_line}" if cleaned else source_line


def citation_label(citation: dict) -> str:
    """把一条检索引用格式化成【文件名, 第N页】（页码从 1 开始）。"""
    source = str(citation.get("source") or "unknown").replace("\\", "/")
    name = source.rsplit("/", 1)[-1] or source
    page = citation.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        return f"【{name}, 第{page + 1}页】"
    return f"【{name}】"


def public_citations(citations: list[dict] | None) -> list[dict]:
    """整理对外引用：只保留文件名与从 1 开始的页码，不包含片段正文或分数。"""
    out: list[dict] = []
    seen: set[tuple] = set()
    for citation in citations or []:
        source = str(citation.get("source") or "unknown").replace("\\", "/")
        name = source.rsplit("/", 1)[-1] or source
        page = citation.get("page")
        page_out = page + 1 if isinstance(page, int) and not isinstance(page, bool) else None
        key = (name, page_out)
        if key in seen:
            continue
        seen.add(key)
        item = {"source": name}
        if page_out is not None:
            item["page"] = page_out
        out.append(item)
    return out


def attach_citations(text: str, citations: list[dict] | None) -> str:
    """用实际检索到来源写「数据来源」行，覆盖模型自己编造的来源行。"""
    labels: list[str] = []
    seen: set[str] = set()
    for citation in citations or []:
        label = citation_label(citation)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    if not labels:
        return text
    cleaned = _SOURCE_LINE_RE.sub("", text).strip()
    source_line = "数据来源：" + "、".join(labels)
    return f"{cleaned}\n\n{source_line}" if cleaned else source_line


_LOW_CONFIDENCE_HEDGE = "检索置信度较低，以下内容可能不完整或不准确，请对照引用来源核对后再采信。"


_NO_EVIDENCE_REFUSAL = "没有找到足够相关的资料，无法可靠回答这个问题。请换个问法，或确认相关文档已经入库。"


_UNSUPPORTED_NOTE = "注意：部分说法未能在检索资料中找到依据，请以上方引用来源为准。"


def apply_grounding(text: str, confidence: dict | None, citations: list[dict] | None) -> tuple[str, str | None]:
    """按检索置信度收紧回答。

    没有检索信号时原样返回。检索过但一个片段都没留下时拒答，
    不看置信度档位。低置信但仍有片段时在答案前加明确保留，并始终附上真实引用。

    Returns:
        (回答文本, grounding)。grounding 为 "refused"、"hedged"、"grounded" 或 None。
    """
    if not confidence:
        return text, None
    level = confidence.get("level")
    chunks = int(confidence.get("chunks") or 0)
    # 检索发生过但一个片段都没留下：不论置信度档位，一律拒答。
    if chunks <= 0:
        return _NO_EVIDENCE_REFUSAL, "refused"
    grounded = attach_citations(text, citations)
    if level == "low":
        if not grounded.startswith(_LOW_CONFIDENCE_HEDGE):
            grounded = f"{_LOW_CONFIDENCE_HEDGE}\n\n{grounded}" if grounded else _LOW_CONFIDENCE_HEDGE
        return grounded, "hedged"
    return grounded, "grounded"


def note_unsupported_claims(text: str, unsupported: list[dict] | None) -> str:
    """有未通过忠实度校验的声明时，在答案末尾加一句可见的保留说明。"""
    if not unsupported or _UNSUPPORTED_NOTE in text:
        return text
    return f"{text}\n\n{_UNSUPPORTED_NOTE}" if text else _UNSUPPORTED_NOTE
