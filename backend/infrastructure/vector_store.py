"""pgvector 向量库：同一套 Postgres 上的稠密 + 稀疏混合检索。

稠密字段是 vector(1024)，HNSW、内积。稀疏字段是 sparsevec，精确内积
（pgvector 0.8 没有稀疏 ANN 索引）。查询侧仍按 DROP_RATIO_SEARCH 丢掉
最小的一批稀疏权重，再和稠密名次做 RRF（k=60）。

表不存在时检索返回空；结构对不上时拒绝启动，不在原地改向量列。
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading

from sqlalchemy import text

from backend.core.fusion import rrf_contribution
from backend.domain.errors import ServiceUnavailableError, ValidationError
from backend.infrastructure.config import settings
from backend.infrastructure.database import get_sync_engine

logger = logging.getLogger("ragmate")

DENSE_DIM = 1024
# 1-based pgvector 下标 = 词 id + 1。BGE-M3 词表约 25 万，哈希桶 262144，100 万留余量。
SPARSE_DIM = 1_000_000
_MIN_PGVECTOR = (0, 7, 0)
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_REQUIRED_COLUMNS = (
    "id",
    "dense",
    "sparse",
    "text",
    "parent_text",
    "source",
    "page",
    "chunk_index",
    "content_hash",
    "ingest_batch_id",
)

_ready_tables: set[str] = set()
_ready_lock = threading.Lock()

_QUERY_PAGE = 1000
_QUERY_HARD_CAP = 100_000
_DELETE_BATCH = 500


def quote_ident(name: str) -> str:
    """只接受短标识符，再加双引号。表名来自配置，不能直接拼进 SQL。"""
    if not isinstance(name, str) or not _TABLE_RE.fullmatch(name):
        raise ValidationError(f"Invalid vector table name: {name!r}")
    return '"' + name + '"'


def _parse_version(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in str(value).split("."):
        digits = ""
        for ch in piece:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_at_least(value: str, minimum: tuple[int, ...]) -> bool:
    got = _parse_version(value)
    width = max(len(got), len(minimum))
    padded = got + (0,) * (width - len(got))
    target = minimum + (0,) * (width - len(minimum))
    return padded >= target


def as_sparse_dict(raw) -> dict[int, float]:
    """把编码结果收成 {index: weight}。scipy 稀疏矩阵走 indices/data。"""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[int, float] = {}
        for key, value in raw.items():
            try:
                idx = int(key)
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(weight) and weight != 0.0:
                out[idx] = weight
        return out
    indices = getattr(raw, "indices", None)
    data = getattr(raw, "data", None)
    if indices is not None and data is not None:
        out = {}
        for idx, weight in zip(indices, data):
            weight = float(weight)
            if math.isfinite(weight) and weight != 0.0:
                out[int(idx)] = weight
        return out
    raise ValidationError("sparse vector must be a mapping of index to weight")


def apply_drop_ratio(weights: dict[int, float], drop_ratio: float) -> dict[int, float]:
    """丢掉绝对值最小的一批权重。比例为 0 时原样返回。至少留一个非零项。"""
    if not weights or drop_ratio <= 0:
        return dict(weights)
    items = sorted(weights.items(), key=lambda kv: (abs(float(kv[1])), int(kv[0])))
    n_drop = int(len(items) * float(drop_ratio))
    kept = items[n_drop:] if n_drop > 0 else items
    if not kept:
        kept = items[-1:]
    return {int(k): float(v) for k, v in kept}


def format_dense(vec) -> str:
    if not isinstance(vec, (list, tuple)):
        raise ValidationError("dense vector must be a list of floats")
    if len(vec) != DENSE_DIM:
        raise ValidationError(
            f"dense vector length {len(vec)} != {DENSE_DIM}. "
            "Re-ingest after changing the embedding model."
        )
    return "[" + ",".join(format(float(x), ".8g") for x in vec) + "]"


def format_sparse(raw) -> str:
    """pgvector sparsevec 文本：下标从 1 开始，`{1:0.5,3:0.2}/dim`。"""
    weights = as_sparse_dict(raw)
    items: list[tuple[int, float]] = []
    for idx, weight in weights.items():
        pg_index = idx + 1
        if pg_index < 1 or pg_index > SPARSE_DIM:
            raise ValidationError(
                f"sparse index {idx} is outside 0..{SPARSE_DIM - 1}"
            )
        items.append((pg_index, weight))
    if not items:
        items = [(1, 1.0)]
    items.sort()
    body = ",".join(f"{idx}:{format(weight, '.8g')}" for idx, weight in items)
    return "{" + body + "}/" + str(SPARSE_DIM)


def fuse_hits(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    *,
    limit: int,
    rrf_k: int,
) -> list[dict]:
    """按 id 融合两路命中：RRF 用名次累加。"""
    by_id: dict[int, dict] = {}
    for hit in list(dense_hits) + list(sparse_hits):
        by_id.setdefault(int(hit["id"]), hit)
    scores: dict[int, float] = {}
    for rank, hit in enumerate(dense_hits):
        scores[int(hit["id"])] = scores.get(int(hit["id"]), 0.0) + rrf_contribution(rank, int(rrf_k))
    for rank, hit in enumerate(sparse_hits):
        scores[int(hit["id"])] = scores.get(int(hit["id"]), 0.0) + rrf_contribution(rank, int(rrf_k))
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    fused = []
    for item_id in ordered[:limit]:
        hit = dict(by_id[item_id])
        hit["id"] = item_id
        hit["distance"] = scores[item_id]
        fused.append(hit)
    return fused


def dense_search_sql(table: str) -> str:
    ident = quote_ident(table)
    return (
        "SELECT id, text, parent_text, source, page, chunk_index, content_hash, ingest_batch_id, "
        f"-(dense <#> CAST(:q AS vector)) AS score FROM {ident} "
        "ORDER BY dense <#> CAST(:q AS vector), id LIMIT :lim"
    )


def sparse_search_sql(table: str) -> str:
    ident = quote_ident(table)
    return (
        "SELECT id, text, parent_text, source, page, chunk_index, content_hash, ingest_batch_id, "
        f"-(sparse <#> CAST(:q AS sparsevec)) AS score FROM {ident} "
        "ORDER BY sparse <#> CAST(:q AS sparsevec), id LIMIT :lim"
    )


def _row_to_hit(row) -> dict:
    text_value = row["text"] or ""
    parent = row["parent_text"] or text_value
    page = row["page"]
    if isinstance(page, bool) or not isinstance(page, int):
        page = None
    return {
        "id": int(row["id"]),
        "distance": float(row["score"]) if row["score"] is not None else 0.0,
        "entity": {
            "text": text_value,
            "parent_text": parent,
            "metadata": {
                "source": row["source"] or "",
                "page": page,
                "chunk_index": row["chunk_index"],
                "content_hash": row["content_hash"] or "",
                "ingest_batch_id": row["ingest_batch_id"] or "",
            },
        },
    }


def _as_int_id(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _row_batch_id(row: dict) -> str:
    batch = row.get("ingest_batch_id")
    if isinstance(batch, str) and batch.strip():
        return batch.strip()
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        nested = meta.get("ingest_batch_id")
        if isinstance(nested, str):
            return nested.strip()
    return ""


def _stale_vector_ids(rows: list[dict], ingest_batch_id: str | None, keep_ids: set[int]) -> list[int]:
    """挑出要删除的主键。

    行上有 ingest_batch_id 时，只保留最新批次。没有该字段（或为空）时，
    退回「不在本批插入 id 里」的规则，用来清掉更早写入的向量。
    """
    stale: list[int] = []
    for row in rows:
        row_id = _as_int_id(row.get("id"))
        batch = _row_batch_id(row)
        if ingest_batch_id and batch:
            if batch == ingest_batch_id:
                continue
            if row_id is None:
                raise ServiceUnavailableError("旧向量缺少主键，无法确认删除")
            stale.append(row_id)
            continue
        if row_id is not None and row_id in keep_ids:
            continue
        if row_id is None:
            raise ServiceUnavailableError("旧向量缺少主键，无法确认删除")
        stale.append(row_id)
    return stale


class PgVectorStore:
    """对一张 pgvector 表的同步访问。连接来自现有的 SQLAlchemy engine。"""

    def __init__(self, table: str | None = None):
        self.table = table or settings.VECTOR_TABLE

    def _connect(self):
        return get_sync_engine().begin()

    def extension_version(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )).first()
        if not row:
            return None
        return str(row[0])

    def table_exists(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :name"
            ), {"name": self.table}).first()
        return row is not None

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            ensure_schema_conn(conn, self.table)

    def drop_table(self) -> None:
        ident = quote_ident(self.table)
        with self._connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {ident}"))
        with _ready_lock:
            _ready_tables.discard(self.table)

    def validate_schema(self) -> None:
        with self._connect() as conn:
            _validate_schema(conn, self.table)

    def search_dense(self, dense_vec, limit: int) -> list[dict]:
        limit = int(limit)
        if limit <= 0:
            return []
        ef = min(1000, max(limit * 4, 128))
        return self._ann_query(
            dense_search_sql(self.table),
            {"q": format_dense(dense_vec), "lim": limit},
            ef_search=ef,
        )

    def search_sparse(self, sparse_vec, limit: int, drop_ratio: float | None = None) -> list[dict]:
        limit = int(limit)
        if limit <= 0:
            return []
        ratio = settings.DROP_RATIO_SEARCH if drop_ratio is None else float(drop_ratio)
        weights = apply_drop_ratio(as_sparse_dict(sparse_vec), ratio)
        return self._ann_query(
            sparse_search_sql(self.table),
            {"q": format_sparse(weights), "lim": limit},
            ef_search=None,
        )

    def search_hybrid(self, dense_vec, sparse_vec, limit: int) -> list[dict]:
        """稠密、稀疏各取 limit 条，再用 RRF 融合到 limit 条。"""
        dense_hits = self.search_dense(dense_vec, limit)
        sparse_hits = self.search_sparse(sparse_vec, limit)
        return fuse_hits(
            dense_hits,
            sparse_hits,
            limit=int(limit),
            rrf_k=settings.RRF_K,
        )

    def _ann_query(self, sql: str, params: dict, *, ef_search: int | None) -> list[dict]:
        try:
            with self._connect() as conn:
                if ef_search is not None:
                    try:
                        with conn.begin_nested():
                            conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
                    except Exception:
                        logger.debug("hnsw.ef_search was not applied", exc_info=True)
                rows = conn.execute(text(sql), params).mappings().all()
        except (ValidationError, ServiceUnavailableError):
            raise
        except Exception as e:
            logger.error("pgvector search failed", exc_info=True)
            raise ServiceUnavailableError("检索服务异常，请稍后重试") from e
        return [_row_to_hit(row) for row in rows]

    def delete_source(self, filename: str) -> None:
        ident = quote_ident(self.table)
        with self._connect() as conn:
            conn.execute(
                text(f"DELETE FROM {ident} WHERE source = :source"),
                {"source": filename},
            )

    def delete_ids(self, ids: list[int]) -> None:
        if not ids:
            return
        ident = quote_ident(self.table)
        with self._connect() as conn:
            for start in range(0, len(ids), _DELETE_BATCH):
                batch = ids[start:start + _DELETE_BATCH]
                placeholders = ", ".join(f":id{i}" for i in range(len(batch)))
                params = {f"id{i}": int(item_id) for i, item_id in enumerate(batch)}
                conn.execute(text(f"DELETE FROM {ident} WHERE id IN ({placeholders})"), params)

    def source_rows(self, filename: str, *, limit: int | None = None) -> list[dict]:
        ident = quote_ident(self.table)
        if limit is not None:
            with self._connect() as conn:
                rows = conn.execute(text(
                    f"SELECT id, ingest_batch_id FROM {ident} WHERE source = :source LIMIT :lim"
                ), {"source": filename, "lim": int(limit)}).mappings().all()
            return [dict(row) for row in rows]

        rows_out: list[dict] = []
        offset = 0
        while True:
            with self._connect() as conn:
                batch = conn.execute(text(
                    f"SELECT id, ingest_batch_id FROM {ident} "
                    "WHERE source = :source ORDER BY id LIMIT :lim OFFSET :off"
                ), {"source": filename, "lim": _QUERY_PAGE, "off": offset}).mappings().all()
            rows_out.extend(dict(row) for row in batch)
            if len(batch) < _QUERY_PAGE:
                return rows_out
            offset += len(batch)
            if offset > _QUERY_HARD_CAP:
                raise ServiceUnavailableError("向量数量异常，无法确认旧向量已删除")

    def hashes_present(self, hashes: list[str]) -> set[str]:
        if not hashes:
            return set()
        ident = quote_ident(self.table)
        found: set[str] = set()
        for start in range(0, len(hashes), 100):
            batch = hashes[start:start + 100]
            placeholders = ", ".join(f":h{i}" for i in range(len(batch)))
            params = {f"h{i}": value for i, value in enumerate(batch)}
            with self._connect() as conn:
                rows = conn.execute(text(
                    f"SELECT content_hash FROM {ident} WHERE content_hash IN ({placeholders})"
                ), params).all()
            for row in rows:
                if row[0]:
                    found.add(row[0])
        return found

    def existing_identity_keys(self, hashes: list[str]) -> set[tuple]:
        """(source, chunk_index, content_hash) 已在表里的键，给迁移去重用。"""
        if not hashes:
            return set()
        ident = quote_ident(self.table)
        found: set[tuple] = set()
        for start in range(0, len(hashes), 100):
            batch = hashes[start:start + 100]
            placeholders = ", ".join(f":h{i}" for i in range(len(batch)))
            params = {f"h{i}": value for i, value in enumerate(batch)}
            with self._connect() as conn:
                rows = conn.execute(text(
                    f"SELECT source, chunk_index, content_hash FROM {ident} "
                    f"WHERE content_hash IN ({placeholders})"
                ), params).all()
            for source, chunk_index, content_hash in rows:
                found.add((source or "", chunk_index, content_hash or ""))
        return found

    def insert_rows(self, rows: list[dict]) -> list[int]:
        if not rows:
            return []
        ident = quote_ident(self.table)
        sql = (
            f"INSERT INTO {ident} "
            "(dense, sparse, text, parent_text, source, page, chunk_index, content_hash, ingest_batch_id) "
            "VALUES (CAST(:dense AS vector), CAST(:sparse AS sparsevec), :text, :parent_text, "
            ":source, :page, :chunk_index, :content_hash, :ingest_batch_id) RETURNING id"
        )
        ids: list[int] = []
        try:
            with self._connect() as conn:
                for row in rows:
                    page = row.get("page")
                    if isinstance(page, bool) or not isinstance(page, int):
                        page = None
                    new_id = conn.execute(text(sql), {
                        "dense": format_dense(row["dense"]),
                        "sparse": format_sparse(row["sparse"]),
                        "text": row.get("text") or "",
                        "parent_text": row.get("parent_text") or row.get("text") or "",
                        "source": row.get("source") or "unknown",
                        "page": page,
                        "chunk_index": int(row.get("chunk_index") or 0),
                        "content_hash": row.get("content_hash") or "",
                        "ingest_batch_id": row.get("ingest_batch_id") or "",
                    }).scalar_one()
                    ids.append(int(new_id))
        except (ValidationError, ServiceUnavailableError):
            raise
        except Exception as e:
            logger.error("pgvector insert failed", exc_info=True)
            raise ServiceUnavailableError("向量写入失败") from e
        if len(ids) != len(rows):
            raise ServiceUnavailableError("向量写入结果无法确认")
        return ids

    def count_rows(self) -> int:
        ident = quote_ident(self.table)
        with self._connect() as conn:
            value = conn.execute(text(f"SELECT COUNT(*) FROM {ident}")).scalar_one()
        return int(value)

    def fetch_chunks(self, limit: int = 16384) -> list[dict]:
        ident = quote_ident(self.table)
        with self._connect() as conn:
            rows = conn.execute(text(
                f"SELECT text, parent_text, source, page, chunk_index FROM {ident} "
                "ORDER BY source, chunk_index LIMIT :lim"
            ), {"lim": int(limit)}).mappings().all()
        return [dict(row) for row in rows]

    def chunk_exists(self, source: str, chunk_index: int) -> bool:
        ident = quote_ident(self.table)
        with self._connect() as conn:
            row = conn.execute(text(
                f"SELECT 1 FROM {ident} WHERE source = :source AND chunk_index = :chunk_index LIMIT 1"
            ), {"source": source, "chunk_index": int(chunk_index)}).first()
        return row is not None


def _require_extension(conn) -> None:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    row = conn.execute(text(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )).first()
    if not row or not _version_at_least(str(row[0]), _MIN_PGVECTOR):
        found = row[0] if row else "missing"
        raise ValidationError(
            f"pgvector {found} is too old; sparse vectors need >= 0.7. "
            "Use the pgvector/pgvector:0.8.7-pg15 image (or any Postgres with pgvector >= 0.7)."
        )


def _validate_schema(conn, table: str) -> None:
    rows = conn.execute(text(
        "SELECT column_name, udt_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :name"
    ), {"name": table}).all()
    columns = {name: udt for name, udt in rows}
    missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValidationError(
            f"向量表 '{table}' 结构不匹配（缺少: {', '.join(missing)}）。"
            f"请确认后删除该表并重新入库：DROP TABLE {quote_ident(table)};"
        )
    expected = {"dense": "vector", "sparse": "sparsevec"}
    wrong = [
        f"{name} is {columns[name]}"
        for name, want in expected.items()
        if columns.get(name) != want
    ]
    if wrong:
        raise ValidationError(
            f"向量表 '{table}' 类型不匹配（{', '.join(wrong)}）。"
            f"请删除该表后重新入库：DROP TABLE {quote_ident(table)};"
        )


def _create_indexes(conn, table: str) -> None:
    ident = quote_ident(table)
    conn.execute(text(
        f"CREATE INDEX IF NOT EXISTS {quote_ident(table + '_source_idx')} ON {ident} (source)"
    ))
    conn.execute(text(
        f"CREATE INDEX IF NOT EXISTS {quote_ident(table + '_hash_idx')} ON {ident} (content_hash)"
    ))
    conn.execute(text(
        f"CREATE INDEX IF NOT EXISTS {quote_ident(table + '_dense_hnsw')} "
        f"ON {ident} USING hnsw (dense vector_ip_ops)"
    ))


def ensure_schema_conn(conn, table: str | None = None) -> None:
    """在已有连接上创建扩展、表和索引。给 init_db 和入库共用。"""
    table = table or settings.VECTOR_TABLE
    quote_ident(table)
    _require_extension(conn)
    exists = conn.execute(text(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = :name"
    ), {"name": table}).first()
    if not exists:
        ident = quote_ident(table)
        conn.execute(text(
            f"CREATE TABLE {ident} ("
            "id BIGSERIAL PRIMARY KEY, "
            f"dense vector({DENSE_DIM}) NOT NULL, "
            f"sparse sparsevec({SPARSE_DIM}) NOT NULL, "
            "text TEXT NOT NULL, "
            "parent_text TEXT NOT NULL, "
            "source TEXT NOT NULL, "
            "page INTEGER, "
            "chunk_index INTEGER NOT NULL DEFAULT 0, "
            "content_hash TEXT NOT NULL DEFAULT '', "
            "ingest_batch_id TEXT NOT NULL DEFAULT ''"
            ")"
        ))
    _validate_schema(conn, table)
    _create_indexes(conn, table)
    with _ready_lock:
        _ready_tables.add(table)


def get_vector_store() -> PgVectorStore:
    return PgVectorStore()


def reset_vector_store_cache() -> None:
    """切换 VECTOR_TABLE 后调用，让下次 init 重新检查那张表。"""
    with _ready_lock:
        _ready_tables.clear()


def init_vector_store() -> PgVectorStore | None:
    """表已存在且结构匹配时返回 store。表不存在返回 None。库不可达则抛 503。"""
    store = get_vector_store()
    with _ready_lock:
        if store.table in _ready_tables:
            return store
    try:
        if not store.table_exists():
            return None
        store.validate_schema()
    except ValidationError:
        raise
    except Exception as e:
        raise ServiceUnavailableError("检索服务异常，请稍后重试") from e
    with _ready_lock:
        _ready_tables.add(store.table)
    return store


def check_vector_store_available() -> bool:
    """Postgres 上的 pgvector 扩展是否可用（>= 0.7，含 sparsevec）。"""
    try:
        version = get_vector_store().extension_version()
    except Exception:
        logger.debug("pgvector availability check failed", exc_info=True)
        return False
    return bool(version) and _version_at_least(version, _MIN_PGVECTOR)


def deduplicate_chunks(store, chunks, chunk_counter, force_include_sources: set[str] | None = None):
    """查已存在的 content_hash，跳过重复 chunk。返回 (filtered_chunks, skipped_files)。"""
    from collections import Counter

    skipped_files = {}
    force_include_sources = force_include_sources or set()
    try:
        collection_exists = store.table_exists()
    except Exception as e:
        logger.warning(f"Content dedup query failed, proceeding without dedup: {e}")
        return chunks, skipped_files

    if not collection_exists or not chunks:
        return chunks, skipped_files

    new_hashes = list({c.metadata["content_hash"] for c in chunks})
    try:
        existing_hashes = store.hashes_present(new_hashes)
        if existing_hashes:
            original_count = len(chunks)
            file_skip_count = Counter()
            filtered_chunks = []
            for chunk in chunks:
                src = os.path.basename(chunk.metadata.get("source", ""))
                if src not in force_include_sources and chunk.metadata["content_hash"] in existing_hashes:
                    file_skip_count[src] += 1
                else:
                    filtered_chunks.append(chunk)

            for src, skip_cnt in file_skip_count.items():
                total_in_file = chunk_counter.get(src, 0)
                if skip_cnt >= total_in_file:
                    skipped_files[src] = "内容与其他文档完全重复"
                    logger.info(f"Skip {src}: all {skip_cnt} chunks already exist")
                else:
                    logger.info(f"Dedup {src}: {skip_cnt}/{total_in_file} chunks already exist")

            chunks = filtered_chunks
            logger.info(
                f"Content dedup: {original_count} -> {len(chunks)} chunks "
                f"({original_count - len(chunks)} skipped)"
            )
    except Exception as e:
        logger.warning(f"Content dedup query failed, proceeding without dedup: {e}")

    return chunks, skipped_files


def delete_source_vectors(store, filename: str) -> None:
    """删除某个文件的全部向量，并确认已经不存在。

    失败抛 ServiceUnavailableError。调用方必须在此成功之后才能提交数据库删除。
    """
    if store is None:
        raise ServiceUnavailableError("向量库不可用，已保留文档记录")
    try:
        exists = store.table_exists()
    except Exception as e:
        logger.error("Vector store unavailable during document delete for %s", filename, exc_info=True)
        raise ServiceUnavailableError("向量库不可用，已保留文档记录") from e
    if not exists:
        return
    try:
        store.delete_source(filename)
        leftover = store.source_rows(filename, limit=1)
    except ServiceUnavailableError:
        raise
    except Exception as e:
        logger.error("Vector delete failed for %s", filename, exc_info=True)
        raise ServiceUnavailableError("向量删除失败，已保留文档记录") from e
    if leftover:
        logger.error("Vector delete left rows for %s", filename)
        raise ServiceUnavailableError("向量删除未完成，已保留文档记录")


def delete_old_chunks(store, keep_ids_by_source: dict[str, list[int]], *, ingest_batch_id: str | None = None) -> None:
    """删除指定文件的旧向量。

    优先按 ingest_batch_id 去掉非最新批次。行上没有该字段时，保留本批插入的 id。
    删除没有确认完成就抛 ServiceUnavailableError，调用方不得把入库记为成功。
    """
    if not keep_ids_by_source:
        return
    batch_id = None
    if ingest_batch_id is not None:
        if not isinstance(ingest_batch_id, str) or not ingest_batch_id.strip():
            raise ServiceUnavailableError("入库批次号无效，无法删除旧向量")
        batch_id = ingest_batch_id.strip()

    for filename, keep_ids in keep_ids_by_source.items():
        keep = {item_id for item_id in (keep_ids or []) if isinstance(item_id, int)}
        try:
            rows = store.source_rows(filename)
            stale = _stale_vector_ids(rows, batch_id, keep)
            if not stale:
                continue
            store.delete_ids(stale)
            remaining = store.source_rows(filename)
            left = _stale_vector_ids(remaining, batch_id, keep)
            if left:
                raise ServiceUnavailableError(f"旧向量删除未完成: {filename}")
        except ServiceUnavailableError:
            logger.error("Failed to delete old chunks for %s", filename, exc_info=True)
            raise
        except Exception as e:
            logger.error("Failed to delete old chunks for %s", filename, exc_info=True)
            raise ServiceUnavailableError(f"删除旧向量失败: {filename}") from e


def insert_chunks(store, chunks, dense_vecs, sparse_vecs) -> dict[str, list[int]]:
    """把编码后的 chunk 写入 pgvector，并返回按来源文件分组的新 id。"""
    rows = []
    sources = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        parent = chunk.metadata.pop("parent_text", chunk.page_content)
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        page = chunk.metadata.get("page")
        if isinstance(page, bool) or not isinstance(page, int):
            page = None
        rows.append({
            "dense": dense,
            "sparse": sparse,
            "text": chunk.page_content,
            "parent_text": parent,
            "source": source,
            "page": page,
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "content_hash": chunk.metadata.get("content_hash", ""),
            "ingest_batch_id": chunk.metadata.get("ingest_batch_id", ""),
        })
        sources.append(source)
    ids = store.insert_rows(rows)
    ids_by_source: dict[str, list[int]] = {}
    for item_id, source in zip(ids, sources):
        ids_by_source.setdefault(source, []).append(item_id)
    return ids_by_source
