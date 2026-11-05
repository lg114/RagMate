"""BM25 词法检索：生产 sparse 臂与 CI 词法重放共用的实现。

sparse 臂已与 embedding 模型解耦：它不再读 `sparsevec` 列（BGE-M3 自带
稀疏权重，Jina 不给，远程下只能退化成词哈希），而是对表里全部 chunk 文本
算 BM25。同一套分词、同一组参数（k1=1.2，b=0.75），门禁的词法重放与
线上 sparse 臂排出来的顺序一致。

分词：拉丁字母/数字串（小写）+ 连续汉字的重叠二字组。中文技术文档里
命令、端口、标识符和二字词已经能把 gold chunk 排进前列；它不是线上
稠密检索的替代品，只是与稠密臂互补的词法臂。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[A-Za-z0-9]+(?:[./:+_-][A-Za-z0-9]+)*")

BM25_K1 = 1.2
BM25_B = 0.75

# 记入评测 manifest 的 sparse 实现标记：换实现时改这里，manifest diff 可见。
SPARSE_METHOD = "bm25"


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
        return [key for key, _ in self.rank_with_scores(query)]

    def rank_with_scores(self, query: str) -> list[tuple[tuple, float]]:
        """按相关度从高到低返回 `(chunk key, BM25 分数)`。"""
        if not self.chunks:
            return []
        query_tf = Counter(tokenize(query))
        scored = [
            (self.chunks[i]["key"], self._score(i, query_tf))
            for i in range(self._n)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

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
