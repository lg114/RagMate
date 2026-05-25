"""BGE-M3 向量编码：dense + sparse 双向量提取。"""
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

from config import settings

ENCODE_BATCH_SIZE = 64


@lru_cache(maxsize=1)
def get_bge_m3():
    """加载 bge-m3 模型（单例），用于提取 dense + sparse 向量。"""
    return BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=False)


def encode_documents(texts: list[str]) -> tuple[list, list]:
    """用 bge-m3 提取 dense 和 sparse 向量（分批处理避免内存溢出）。返回 (dense_vecs, sparse_vecs)。"""
    model = get_bge_m3()
    all_dense = []
    all_sparse = []

    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch = texts[i:i + ENCODE_BATCH_SIZE]
        output = model.encode(batch, return_dense=True, return_sparse=True)
        all_dense.extend(output["dense_vecs"].tolist())
        for lexical_weight in output["lexical_weights"]:
            all_sparse.append({int(k): float(v) for k, v in lexical_weight.items()})

    return all_dense, all_sparse


def encode_query(query: str) -> tuple[list, dict]:
    """用 bge-m3 提取 query 的 dense 和 sparse 向量。"""
    model = get_bge_m3()
    output = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = output["dense_vecs"][0].tolist()
    sparse_vec = {int(k): float(v) for k, v in output["lexical_weights"][0].items()}
    return dense_vec, sparse_vec
