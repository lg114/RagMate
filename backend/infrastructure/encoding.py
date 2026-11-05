"""向量编码：默认走远程 OpenAI 兼容 HTTP；显式 opt-in 后才在进程内用 FlagEmbedding。"""
import json
import logging
import math
import time
import urllib.error
import urllib.request
import zlib
from functools import lru_cache

from backend.infrastructure.config import REMOTE_EMBEDDING_PROVIDERS, settings

logger = logging.getLogger("ragmate")

ENCODE_BATCH_SIZE = 64
_SPARSE_BUCKETS = 262144
# 429 重试：与 reranker（PR #40）同一口径——优先用 Retry-After，
# 否则按 5/10/20/40 秒退避，最多 5 次；其它错误直接抛出。
_EMBED_MAX_ATTEMPTS = 5
_EMBED_BACKOFF_SECONDS = (5.0, 10.0, 20.0, 40.0)


class EncodingCancelled(Exception):
    """入库线程在编码过程中观察到取消标志。"""


def uses_remote_embeddings() -> bool:
    """嵌入是否走 HTTP，而不是在本进程加载 BGE-M3。默认是。"""
    return settings.EMBEDDING_PROVIDER in REMOTE_EMBEDDING_PROVIDERS


@lru_cache(maxsize=1)
def get_bge_m3():
    """加载 bge-m3 模型（单例），用于提取 dense + sparse 向量。

    仅 EMBEDDING_PROVIDER 为 huggingface / local / flagembedding 时调用。
    FlagEmbedding 在函数内导入，默认远程路径不会加载它。
    未安装本地模型 extra 时，报错会指明安装方式。
    """
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise ImportError(
            "Local embedding models need the 'local' extra: "
            'pip install -e ".[local]"'
        ) from exc

    device = settings.EMBEDDING_DEVICE
    use_fp16 = device == "cuda"
    return BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=use_fp16, devices=device)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            tokens.append("".join(buf).lower())
            buf.clear()

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            flush()
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            flush()
    flush()
    return [tok for tok in tokens if len(tok) >= 2 or "\u4e00" <= tok <= "\u9fff"]


def _hash_sparse(text: str) -> dict[int, float]:
    """远程服务没有返回 sparse 时，用词哈希造一个稳定的稀疏向量。

    这不是 BGE-M3 的 lexical weights。索引和查询必须用同一种嵌入方式，切换后要重新入库。
    """
    weights: dict[int, float] = {}
    for tok in _tokens(text):
        idx = zlib.crc32(tok.encode("utf-8")) % _SPARSE_BUCKETS
        weights[idx] = weights.get(idx, 0.0) + 1.0
    if not weights:
        return {0: 1.0}
    return {idx: 1.0 + math.log(weight) for idx, weight in weights.items()}


def _coerce_sparse(raw) -> dict[int, float] | None:
    try:
        if isinstance(raw, dict) and raw:
            return {int(k): float(v) for k, v in raw.items()}
        if isinstance(raw, list) and raw:
            out: dict[int, float] = {}
            for item in raw:
                if isinstance(item, dict) and "index" in item:
                    value = item.get("value", item.get("weight", 0.0))
                    out[int(item["index"])] = float(value)
            return out or None
    except (TypeError, ValueError):
        return None
    return None


def _post_embeddings(texts: list[str]) -> list[dict]:
    """调用 OpenAI 兼容的 /embeddings。返回按 index 排好的 data 项。

    429 时重试：优先用响应里的 Retry-After，否则按 5/10/20/40 秒指数退避，
    最多 5 次；其它错误直接抛出。入库一次调几百个 batch，没有重试的话
    一次限流就让整个入库白跑（编码全在写入前，失败不落盘）。
    """
    url = settings.EMBEDDING_API_BASE_URL.rstrip("/") + "/embeddings"
    payload = json.dumps({
        "model": settings.EMBEDDING_MODEL,
        "input": texts,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if settings.EMBEDDING_API_KEY:
        req.add_header("Authorization", f"Bearer {settings.EMBEDDING_API_KEY}")
    data = None
    for attempt in range(_EMBED_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=settings.EMBEDDING_API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == _EMBED_MAX_ATTEMPTS - 1:
                raise
            retry_after = e.headers.get("Retry-After") if e.headers else None
            try:
                wait = float(retry_after) if retry_after else _EMBED_BACKOFF_SECONDS[attempt]
            except (TypeError, ValueError):
                wait = _EMBED_BACKOFF_SECONDS[attempt]
            logger.warning("embedding 429，第 %d 次重试前等待 %.1f 秒", attempt + 1, wait)
            time.sleep(wait)
    rows = list(data.get("data") or [])
    rows.sort(key=lambda row: row.get("index", 0))
    if len(rows) != len(texts):
        raise RuntimeError(
            f"embedding service returned {len(rows)} vectors for {len(texts)} inputs"
        )
    return rows


def _encode_remote(texts: list[str], cancel_event=None) -> tuple[list, list]:
    """远程 dense（以及服务若提供的 sparse）。缺 sparse 时用本地词哈希补上。"""
    if cancel_event is not None and cancel_event.is_set():
        raise EncodingCancelled()
    all_dense: list = []
    all_sparse: list = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        if cancel_event is not None and cancel_event.is_set():
            raise EncodingCancelled()
        batch = texts[i:i + ENCODE_BATCH_SIZE]
        for text, row in zip(batch, _post_embeddings(batch)):
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("embedding service returned an empty vector")
            all_dense.append([float(x) for x in embedding])
            sparse = _coerce_sparse(row.get("sparse") if row.get("sparse") is not None else row.get("lexical_weights"))
            all_sparse.append(sparse if sparse is not None else _hash_sparse(text))
    return all_dense, all_sparse


def _encode_local(texts: list[str], cancel_event=None) -> tuple[list, list]:
    if cancel_event is not None and cancel_event.is_set():
        raise EncodingCancelled()
    model = get_bge_m3()
    all_dense = []
    all_sparse = []
    for i in range(0, len(texts), ENCODE_BATCH_SIZE):
        if cancel_event is not None and cancel_event.is_set():
            raise EncodingCancelled()
        batch = texts[i:i + ENCODE_BATCH_SIZE]
        output = model.encode(batch, return_dense=True, return_sparse=True)
        all_dense.extend(output["dense_vecs"].tolist())
        for lexical_weight in output["lexical_weights"]:
            all_sparse.append({int(k): float(v) for k, v in lexical_weight.items()})
    return all_dense, all_sparse


def encode_documents(texts: list[str], cancel_event=None) -> tuple[list, list]:
    """提取 dense 和 sparse 向量（分批处理避免内存溢出）。

    默认（EMBEDDING_PROVIDER 为 remote 或 openai）请求 OpenAI 兼容的 /embeddings，
    本进程不加载模型。huggingface / local / flagembedding 才用本进程里的 FlagEmbedding。
    cancel_event 已置位时抛出 EncodingCancelled，且不会为了取消去加载模型。
    返回 (dense_vecs, sparse_vecs)。
    """
    if not texts:
        return [], []
    if uses_remote_embeddings():
        return _encode_remote(texts, cancel_event)
    return _encode_local(texts, cancel_event)


def encode_query(query: str) -> tuple[list, dict]:
    """提取 query 的 dense 和 sparse 向量。远程模式下不会加载本地 BGE-M3。"""
    dense, sparse = encode_documents([query])
    return dense[0], sparse[0]
