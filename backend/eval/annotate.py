"""新增题目的最小标注流程：人工定 gold，工具只校验和落盘。

gold 标注是人工判断，工具不生成题目、不猜 gold。它只做三件事：

1. ``chunks``：用生产切分复现 chunk 清单，标注者读 chunk 文本定 gold。
2. ``annotate``：把一道题追加进 testset JSON，写盘前校验 schema 与 gold
   存在性，任何含糊输入（重复题、不存在的 gold、类型与 gold 不一致）
   都拒绝写盘。
3. 写盘后提醒后续三步：``check-gold``、重建线上索引、重跑
   ``retrieval`` 生成新基线——题集一改，门禁在基线更新前一定失败，
   这是门禁按题干对齐的设计，不是 bug。

chunk_index 按文件独立编号（见 pipeline._load_and_split），新增文档不改
已有 gold。改切块配置或改动已有文档会改 chunk 清单，届时 check-gold 会
把失效的 gold 指出来，先修 gold 再谈别的。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .retrieval_eval import (
    QUESTION_TYPES,
    build_chunk_inventory,
    load_golden_testset,
    validate_gold_schema,
    verify_gold_in_inventory,
)

PREVIEW_CHARS = 120


def parse_gold_spec(spec: str) -> dict:
    """解析 ``source:chunk_index``，如 ``01-docker-compose.md:2``。

    gold 三元组的 page 恒为 None（清单校验只认 source + chunk_index，
    与已入库的 98 题一致）。source 只认文件名，不认路径。
    """
    text = spec.strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"gold 格式应为 source:chunk_index，got {spec!r}")
    source, index = parts[0].strip(), parts[1].strip()
    if not source or "/" in source or "\\" in source:
        raise ValueError(f"gold source 应为语料文件名，got {spec!r}")
    if not index.isdigit():
        raise ValueError(f"gold chunk_index 应为非负整数，got {spec!r}")
    return {"source": source, "page": None, "chunk_index": int(index)}


def build_case(
    *,
    question: str,
    reference: str,
    question_type: str,
    gold: list[dict],
    claims: list[str],
    source_doc: str | None = None,
) -> dict:
    """组装一道题。source_doc 省略时：答不出题记 \"\"，单源 gold 取该源。"""
    question = question.strip()
    reference = reference.strip()
    claims = [c.strip() for c in claims if c.strip()]
    if not question:
        raise ValueError("question 不能为空")
    if not reference:
        raise ValueError("reference 不能为空（答不出题也要写明语料里没有相关内容）")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type 应为 {QUESTION_TYPES} 之一，got {question_type!r}")
    if not claims:
        raise ValueError("expected_claims 至少一条")
    if source_doc is None:
        if question_type == "unanswerable":
            source_doc = ""
        elif gold and len({g["source"] for g in gold}) == 1:
            source_doc = gold[0]["source"]
        else:
            raise ValueError("多源 gold 时必须显式给 --source-doc")
    return {
        "user_input": question,
        "reference": reference,
        "gold_chunks": [dict(g) for g in gold],
        "expected_claims": claims,
        "question_type": question_type,
        "source_doc": source_doc,
    }


def validate_new_case(case: dict, existing: list[dict], inventory: dict) -> list[str]:
    """新题自身的全部问题：重复题、schema、gold 存在性。空即通过。"""
    errors = []
    if any(c.get("user_input") == case.get("user_input") for c in existing):
        errors.append(f"duplicate question: {case.get('user_input', '')[:40]}")
    errors.extend(validate_gold_schema([case]))
    errors.extend(verify_gold_in_inventory([case], inventory))
    return errors


def append_case(testset_path, case: dict, corpus_dir, dry_run: bool = False) -> list[dict]:
    """校验后把一道题追加进 testset。写盘是原子的：先写临时文件再替换。

    testset 文件本身 schema 不通过时拒绝追加——不在坏文件上继续加题。
    返回追加后的全部题（dry_run 时返回未写盘的追加后列表）。
    """
    testset_path = Path(testset_path)
    cases = load_golden_testset(testset_path)
    inventory = build_chunk_inventory(corpus_dir)
    errors = validate_new_case(case, cases, inventory)
    if errors:
        raise ValueError("标注未通过，拒绝写盘：\n" + "\n".join(f"  - {e}" for e in errors))
    updated = cases + [case]
    if dry_run:
        return updated
    fd, tmp_name = tempfile.mkstemp(dir=testset_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, testset_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return updated


def list_chunk_rows(inventory: dict, source: str | None = None) -> list[dict]:
    """清单转行：按 (source, chunk_index) 排序，preview 截断到 120 字。"""
    rows = []
    for (src, chunk_index), text in inventory.items():
        if source is not None and src != source:
            continue
        collapsed = " ".join(text.split())
        rows.append({
            "source": src,
            "chunk_index": chunk_index,
            "chars": len(text),
            "preview": collapsed[:PREVIEW_CHARS],
        })
    rows.sort(key=lambda r: (r["source"], r["chunk_index"]))
    return rows
