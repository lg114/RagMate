"""删除和重新入库必须真正清掉旧向量，失败时不能留下成功状态。"""
import asyncio
from collections import Counter

import pytest

from backend.application import document_service
from backend.application.ingest import pipeline
from backend.domain.errors import NotFoundError, ServiceUnavailableError
from backend.infrastructure.vector_store import delete_old_chunks, delete_source_vectors


class _Doc:
    filename = "notes.txt"


class _Result:
    def __init__(self, doc):
        self._doc = doc

    def scalar_one_or_none(self):
        return self._doc


class _Session:
    def __init__(self, log):
        self.log = log

    async def commit(self):
        self.log.append("commit")

    async def rollback(self):
        self.log.append("rollback")


class _Store:
    def __init__(self, log, *, exists=True, fail=False, leftover=False):
        self.log = log
        self.exists = exists
        self.fail = fail
        self.leftover = leftover

    def table_exists(self):
        self.log.append("table_exists")
        return self.exists

    def delete_source(self, filename):
        self.log.append("vector_delete")
        if self.fail:
            raise RuntimeError("vector store down")

    def source_rows(self, filename, limit=None):
        self.log.append("vector_query")
        if self.leftover:
            return [{"id": 7, "ingest_batch_id": ""}]
        return []


def _session(log, doc=_Doc()):
    session = _Session(log)

    async def execute(stmt):
        name = stmt.__class__.__name__
        log.append("db_delete" if name == "Delete" else "db_select")
        return _Result(doc)

    session.execute = execute
    return session


def test_delete_document_commits_only_after_vectors_are_gone(tmp_path):
    log = []
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    session = _session(log)
    client = _Store(log)

    asyncio.run(document_service.delete_document("notes.txt", str(tmp_path), session, client))

    assert log == ["db_select", "table_exists", "vector_delete", "vector_query", "db_delete", "commit"]
    assert not path.exists()


def test_delete_document_keeps_row_when_vector_delete_fails(tmp_path):
    log = []
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    session = _session(log)

    with pytest.raises(ServiceUnavailableError):
        asyncio.run(document_service.delete_document(
            "notes.txt", str(tmp_path), session, _Store(log, fail=True),
        ))

    assert "commit" not in log
    assert "db_delete" not in log
    assert "rollback" in log
    assert path.exists()


def test_delete_document_keeps_row_when_vectors_remain(tmp_path):
    log = []
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    session = _session(log)

    with pytest.raises(ServiceUnavailableError) as exc:
        asyncio.run(document_service.delete_document(
            "notes.txt", str(tmp_path), session, _Store(log, leftover=True),
        ))

    assert exc.value.status_code == 503
    assert "commit" not in log
    assert "db_delete" not in log
    assert path.exists()


def test_delete_document_keeps_row_without_a_client(tmp_path):
    log = []
    session = _session(log)
    with pytest.raises(ServiceUnavailableError):
        asyncio.run(document_service.delete_document("notes.txt", str(tmp_path), session, None))
    assert "commit" not in log
    assert "db_delete" not in log


def test_delete_document_missing_is_not_found():
    log = []
    session = _session(log, doc=None)
    with pytest.raises(NotFoundError):
        asyncio.run(document_service.delete_document("notes.txt", "/tmp", session, _Store(log)))
    assert "vector_delete" not in log
    assert "commit" not in log


def test_delete_source_vectors_skips_missing_table():
    client = _Store([], exists=False)
    delete_source_vectors(client, "notes.txt")
    assert "vector_delete" not in client.log


def test_delete_old_chunks_prefers_batch_id_over_keep_list():
    first = [
        {"id": 1, "ingest_batch_id": "new"},
        {"id": 2, "ingest_batch_id": "old"},
        {"id": 3, "ingest_batch_id": ""},
        {"id": 4, "ingest_batch_id": ""},
    ]
    second = [
        {"id": 1, "ingest_batch_id": "new"},
        {"id": 4, "ingest_batch_id": ""},
    ]
    deleted = {}

    class Client:
        def source_rows(self, filename, limit=None):
            return first if "seen" not in deleted else second

        def delete_ids(self, ids):
            deleted["seen"] = True
            deleted["ids"] = ids

    delete_old_chunks(Client(), {"a.txt": [4]}, ingest_batch_id="new")
    assert deleted["ids"] == [2, 3]


def test_delete_old_chunks_raises_when_rows_remain():
    rows = [{"id": 2, "metadata": {"ingest_batch_id": "old"}}]

    class Client:
        def source_rows(self, filename, limit=None):
            return rows

        def delete_ids(self, ids):
            return None

    with pytest.raises(ServiceUnavailableError):
        delete_old_chunks(Client(), {"a.txt": [1]}, ingest_batch_id="new")


def test_delete_old_chunks_raises_when_delete_throws():
    class Client:
        def source_rows(self, filename, limit=None):
            return [{"id": 2, "ingest_batch_id": "old"}]

        def delete_ids(self, ids):
            raise RuntimeError("vector store down")

    with pytest.raises(ServiceUnavailableError):
        delete_old_chunks(Client(), {"a.txt": [1]}, ingest_batch_id="new")


class _Chunk:
    def __init__(self, source):
        self.page_content = "hello world"
        self.metadata = {"source": source}


def _patch_ingest(monkeypatch, tmp_path, delete_impl):
    (tmp_path / "a.txt").write_text("hello world")
    chunk = _Chunk(str(tmp_path / "a.txt"))
    statuses = []
    synced = {"called": False}

    class _IngestStore:
        def ensure_schema(self):
            return True

    monkeypatch.setattr(pipeline, "ensure_ingest_schema_sync", lambda: None)
    monkeypatch.setattr(pipeline, "check_vector_store_available", lambda: True)
    monkeypatch.setattr(pipeline, "ingest_cancel_requested_sync", lambda: False)
    monkeypatch.setattr(pipeline, "_detect_new_files", lambda docs_dir, files: (["a.txt"], {"a.txt": 1.0}))
    monkeypatch.setattr(pipeline, "_load_and_split", lambda *args, **kwargs: ([chunk], Counter({"a.txt": 1}), []))
    monkeypatch.setattr(pipeline, "get_vector_store", lambda: _IngestStore())
    monkeypatch.setattr(pipeline, "deduplicate_chunks", lambda *args, **kwargs: ([chunk], {}))
    monkeypatch.setattr(pipeline, "encode_documents", lambda texts, cancel_event=None: [[0.1]])
    monkeypatch.setattr(pipeline, "insert_chunks", lambda *args, **kwargs: {"a.txt": [11]})
    monkeypatch.setattr(pipeline, "delete_old_chunks", delete_impl)
    monkeypatch.setattr(pipeline, "set_ingest_status_sync", lambda data: statuses.append(dict(data)))

    def sync(*args, **kwargs):
        synced["called"] = True

    monkeypatch.setattr(pipeline, "sync_documents_table", sync)
    return statuses, synced


def test_reingest_delete_failure_is_not_success(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise ServiceUnavailableError("milvus delete failed")

    statuses, synced = _patch_ingest(monkeypatch, tmp_path, boom)
    result = pipeline.ingest_documents(str(tmp_path), ["a.txt"])

    assert result["status"] == "failed"
    assert "未标记为成功" in result["error"]
    assert synced["called"] is False
    assert statuses
    assert all(item.get("status") != "success" for item in statuses)
    assert statuses[-1]["status"] == "failed"


def test_reingest_delete_success_still_syncs(monkeypatch, tmp_path):
    seen = {}

    def delete_ok(client, keep_ids_by_source, ingest_batch_id=None):
        seen["batch"] = ingest_batch_id
        seen["keep"] = keep_ids_by_source

    statuses, synced = _patch_ingest(monkeypatch, tmp_path, delete_ok)
    result = pipeline.ingest_documents(str(tmp_path), ["a.txt"])

    assert result["status"] == "success"
    assert synced["called"] is True
    assert seen["batch"]
    assert seen["keep"]["a.txt"] == [11]
    assert statuses[-1]["status"] == "success"
