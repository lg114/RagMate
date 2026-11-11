"""引用整理与接地：把检索引用写进回答、按置信度拒答/保留。

纯文本函数，无 I/O。chat.py 会重新导出这里的名字（测试直接从
backend.application.chat 导入它们）。
"""

import re

# 模型写进回答里的引用。中文【文件名, 第N页】与英文 [filename, page N]
# 共用这一条，页码都从 1 开始。文件名必须带已知后缀，避免误伤普通括号。
_CITE_RE = re.compile(
    # 页码可以写在括号里（【文件, 第N页】），也可以写在括号外（上下文标签的【文件】 第N页）。
    r"【([^】]+\.(?:pdf|docx?|txt|md))(?:[,，]\s*第(\d+)页)?】(?:\s*第(\d+)页)?"
    r"|\[([^\[\]]+\.(?:pdf|docx?|txt|md))(?:,\s*page\s+(\d+))?\]",
    re.IGNORECASE,
)

_SOURCE_LINE_RE = re.compile(
    r"^\s*(?:数据来源[:：]|Sources:)\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# 汉字不少于「汉字 + 拉丁字母」的一半时视为中文问题。
_CJK_LANG_RATIO = 0.5

_FALLBACK_CITATIONS = 3


def question_language(text: str | None) -> str:
    """用汉字占比判断问题语言，返回 "zh" 或 "en"。

    空文本和纯标点沿用中文，避免把原来的中文引用格式翻成英文。
    """
    if not text:
        return "zh"
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk + latin == 0:
        return "zh"
    return "zh" if cjk / (cjk + latin) >= _CJK_LANG_RATIO else "en"


def _basename(source: str | None) -> str:
    text = str(source or "unknown").replace("\\", "/")
    return text.rsplit("/", 1)[-1] or text


def _public_page(page) -> int | None:
    if isinstance(page, int) and not isinstance(page, bool):
        return page + 1
    return None


def _score(citation: dict) -> float:
    score = citation.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return 0.0
    return float(score)


def _marker_parts(match: re.Match) -> tuple[str, str | None]:
    if match.group(1) is not None:
        return match.group(1).strip(), match.group(2) or match.group(3)
    return (match.group(4) or "").strip(), match.group(5)


def extract_model_citations(text: str) -> list[tuple[str, int | None]]:
    """按出现顺序抽出模型写的引用：(文件名, 从 1 开始的页码或 None)。"""
    found: list[tuple[str, int | None]] = []
    for match in _CITE_RE.finditer(text or ""):
        name, page = _marker_parts(match)
        if not name:
            continue
        found.append((_basename(name), int(page) if page else None))
    return found


def _label(name: str, page: str | int | None, lang: str) -> str:
    """page 已经是展示用的页码（从 1 开始），不要再加一。"""
    base = _basename(name)
    if lang == "en":
        return f"[{base}, page {page}]" if page else f"[{base}]"
    return f"【{base}, 第{page}页】" if page else f"【{base}】"


def _source_line(labels: list[str], lang: str) -> str:
    if lang == "en":
        return "Sources: " + "; ".join(labels)
    return "数据来源：" + "、".join(labels)


def _clean_prose(text: str) -> str:
    """去掉引用标记和已有的来源行。只有真的删过标记时才收紧空白。"""
    had_marker = _CITE_RE.search(text) is not None
    cleaned = _SOURCE_LINE_RE.sub("", text)
    if had_marker:
        cleaned = _CITE_RE.sub("", cleaned)
        cleaned = re.sub(r"[ \t]+([，。；：、,.!?！？])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_citation_markup(text: str) -> str:
    """从回答正文里去掉引用标记和来源行。"""
    if not text:
        return text
    return _clean_prose(text)


def normalize_citations(text: str) -> str:
    """把正文里的引用收成一行来源。哪怕只有一条也收走，中英文格式都认。"""
    matches = list(_CITE_RE.finditer(text or ""))
    if not matches:
        return text
    zh_markers = sum(1 for match in matches if match.group(1) is not None)
    lang = "zh" if zh_markers * 2 >= len(matches) else "en"
    labels: list[str] = []
    for match in matches:
        name, page = _marker_parts(match)
        label = _label(name, page, lang)
        if label not in labels:
            labels.append(label)
    cleaned = _clean_prose(text)
    source_line = _source_line(labels, lang)
    return f"{cleaned}\n\n{source_line}" if cleaned else source_line


def citation_label(citation: dict, lang: str = "zh") -> str:
    """格式化一条检索引用。页码从 1 开始。

    中文：【文件名, 第N页】。英文：[filename, page N]。
    """
    return _label(_basename(citation.get("source")), _public_page(citation.get("page")), lang)


def public_citations(citations: list[dict] | None) -> list[dict]:
    """整理对外引用：只保留文件名与从 1 开始的页码，不包含片段正文或分数。"""
    out: list[dict] = []
    seen: set[tuple] = set()
    for citation in citations or []:
        name = _basename(citation.get("source"))
        page_out = _public_page(citation.get("page"))
        key = (name, page_out)
        if key in seen:
            continue
        seen.add(key)
        item = {"source": name}
        if page_out is not None:
            item["page"] = page_out
        out.append(item)
    return out


def _top_citations(citations: list[dict], limit: int) -> list[dict]:
    ranked = sorted(enumerate(citations), key=lambda item: (-_score(item[1]), item[0]))
    out: list[dict] = []
    seen: set[tuple] = set()
    for _, citation in ranked:
        key = (_basename(citation.get("source")).casefold(), _public_page(citation.get("page")))
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
        if len(out) >= limit:
            break
    return out


def citations_for_answer(answer: str, citations: list[dict] | None) -> list[dict]:
    """模型实际写过、且能对上检索片段（文件名 + 页码）的引用，按首次出现排序。

    一条都解析不到时，改回取得分最高的 3 条（分数相同则保持原顺序）。
    解析到了但没有一条对上检索结果时返回空列表，不拿检索结果充数。
    """
    pool = list(citations or [])
    parsed = extract_model_citations(answer)
    if not parsed:
        return _top_citations(pool, _FALLBACK_CITATIONS)

    selected: list[dict] = []
    seen: set[tuple] = set()
    for name, page in parsed:
        base = name.casefold()
        hits = [
            citation for citation in pool
            if _basename(citation.get("source")).casefold() == base
            and (page is None or _public_page(citation.get("page")) == page)
        ]
        if not hits:
            continue
        hit = max(enumerate(hits), key=lambda item: (_score(item[1]), -item[0]))[1]
        key = (_basename(hit.get("source")).casefold(), _public_page(hit.get("page")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)
    return selected


def attach_citations(text: str, citations: list[dict] | None, lang: str = "zh") -> str:
    """用来源行覆盖模型自己写的来源。lang 决定中文或英文格式。"""
    labels: list[str] = []
    seen: set[str] = set()
    for citation in citations or []:
        label = citation_label(citation, lang)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    if not labels:
        return text
    cleaned = _clean_prose(text)
    source_line = _source_line(labels, lang)
    return f"{cleaned}\n\n{source_line}" if cleaned else source_line


_LOW_CONFIDENCE_HEDGE = "检索置信度较低，以下内容可能不完整或不准确，请对照引用来源核对后再采信。"


_NO_EVIDENCE_REFUSAL = "没有找到足够相关的资料，无法可靠回答这个问题。请换个问法，或确认相关文档已经入库。"


_UNSUPPORTED_NOTE = "注意：部分说法未能在检索资料中找到依据，请以上方引用来源为准。"


def apply_grounding(
    text: str,
    confidence: dict | None,
    citations: list[dict] | None,
    lang: str = "zh",
) -> tuple[str, str | None]:
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
    grounded = attach_citations(text, citations, lang)
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
