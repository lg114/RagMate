"""CI 用的确定性词法检索。

不加载 BGE-M3、不连 Milvus、不用 GPU。样本语料只有几十个 chunk，
BM25 在固定分词上对同一问题的排序是确定的，适合每次 PR 重放。

分词：拉丁字母/数字串（小写）+ 连续汉字的重叠二字组。中文技术文档里
命令、端口、标识符和二字词已经能把 gold chunk 排进前列；它不是线上
稠密检索的替代品，只是同一套 gold chunk ID 上可复现的排序。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[A-Za-z0-9]+(?:[./:+_-][A-Za-z0-9]+)*")

BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """把一段文本切成 BM25 词项。空文本得到空列表。"""
    tokens: list[str] = []
    for match in _LATIN.finditer(text):
        tokens.append("L:" + match.group(0).lower())
    for match in _CJK.finditer(text):
        chars = match.group(0)
        if len(chars) == 1:
            tokens.append("C:" + chars)
        else:
            tokens.extend("C:" + chars[i:i + 2] for i in range(len(chars) - 1))
    return tokens


class LexicalIndex:
    """对一批 chunk 建 BM25 索引。`rank` 的并列名次按 chunk key 稳定打破。"""

    def __init__(self, chunks: list[dict], *, k1: float = BM25_K1, b: float = BM25_B):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(chunk["text"]) for chunk in chunks]
        self._tf = [Counter(tokens) for tokens in self._docs]
        self._df: Counter = Counter()
        for tokens in self._docs:
            self._df.update(set(tokens))
        self._n = len(chunks)
        total = sum(len(tokens) for tokens in self._docs)
        self._avgdl = total / self._n if self._n else 0.0

    def rank(self, query: str) -> list[tuple]:
        """按相关度从高到低返回 chunk key `(source, page, chunk_index)`。"""
        if not self.chunks:
            return []
        query_tf = Counter(tokenize(query))
        scored = [
            (self._score(i, query_tf), self.chunks[i]["key"])
            for i in range(self._n)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [key for _, key in scored]

    def _score(self, index: int, query_tf: Counter) -> float:
        tf = self._tf[index]
        dl = len(self._docs[index])
        if not query_tf or self._avgdl <= 0:
            return 0.0
        score = 0.0
        for term, qf in query_tf.items():
            freq = tf.get(term)
            if not freq:
                continue
            df = self._df[term]
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (freq * (self.k1 + 1)) / denom * (1 + math.log(qf))
        return score
