"""删除和重新入库必须真正清掉旧向量，失败时不能留下成功状态。"""
import asyncio
from collections import Counter

import pytest

from backend.application import document_service
from backend.application.ingest import pipeline
from backend.domain.errors import NotFoundError, ServiceUnavailableError
from backend.infrastructure.milvus import delete_old_chunks, delete_source_vectors


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


class _Milvus:
    def __init__(self, log, *, exists=True, fail=False, leftover=False):
        self.log = log
        self.exists = exists
        self.fail = fail
        self.leftover = leftover

    def has_collection(self, name):
        self.log.append("has_collection")
        return self.exists

    def delete(self, **kwargs):
        self.log.append("milvus_delete")
        if self.fail:
            raise RuntimeError("milvus down")
        return {"delete_count": 1}

    def query(self, **kwargs):
        self.log.append("milvus_query")
        if self.leftover:
            return [{"id": 7}]
        return []


def _session(log, doc=_Doc()):
    session = _Session(log)

    async def execute(stmt):
        name = stmt.__class__.__name__
        log.append("db_delete" if name == "Delete" else "db_select")
        return _Result(doc)

    session.execute = execute
    return session


def test_delete_document_commits_only_after_milvus_confirms(tmp_path):
    log = []
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    session = _session(log)
    client = _Milvus(log)

    asyncio.run(document_service.delete_document("notes.txt", str(tmp_path), session, client))

    assert log == ["db_select", "has_collection", "milvus_delete", "milvus_query", "db_delete", "commit"]
    assert not path.exists()


def test_delete_document_keeps_row_when_milvus_delete_fails(tmp_path):
    log = []
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    session = _session(log)

    with pytest.raises(ServiceUnavailableError):
        asyncio.run(document_service.delete_document(
            "notes.txt", str(tmp_path), session, _Milvus(log, fail=True),
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
            "notes.txt", str(tmp_path), session, _Milvus(log, leftover=True),
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
        asyncio.run(document_service.delete_document("notes.txt", "/tmp", session, _Milvus(log)))
    assert "milvus_delete" not in log
    assert "commit" not in log


def test_delete_source_vectors_skips_missing_collection():
    client = _Milvus([], exists=False)
    delete_source_vectors(client, "notes.txt")
    assert "milvus_delete" not in client.log


def test_delete_old_chunks_prefers_batch_id_over_keep_list():
    first = [
        {"id": 1, "metadata": {"ingest_batch_id": "new"}},
        {"id": 2, "metadata": {"ingest_batch_id": "old"}},
        {"id": 3, "metadata": {}},
        {"id": 4, "metadata": {"ingest_batch_id": ""}},
    ]
    second = [
        {"id": 1, "metadata": {"ingest_batch_id": "new"}},
        {"id": 4, "metadata": {"ingest_batch_id": ""}},
    ]
    deleted = {}

    class Client:
        def query(self, **kwargs):
            return first if "seen" not in deleted else second

        def delete(self, **kwargs):
            deleted["seen"] = True
            deleted["ids"] = kwargs.get("ids")

    delete_old_chunks(Client(), {"a.txt": [4]}, ingest_batch_id="new")
    assert deleted["ids"] == [2, 3]


def test_delete_old_chunks_raises_when_rows_remain():
    rows = [{"id": 2, "metadata": {"ingest_batch_id": "old"}}]

    class Client:
        def query(self, **kwargs):
            return rows

        def delete(self, **kwargs):
            return {"delete_count": 0}

    with pytest.raises(ServiceUnavailableError):
        delete_old_chunks(Client(), {"a.txt": [1]}, ingest_batch_id="new")


def test_delete_old_chunks_raises_when_delete_throws():
    class Client:
        def query(self, **kwargs):
            return [{"id": 2, "metadata": {"ingest_batch_id": "old"}}]

        def delete(self, **kwargs):
            raise RuntimeError("milvus down")

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

    monkeypatch.setattr(pipeline, "check_milvus_available", lambda: True)
    monkeypatch.setattr(pipeline, "ingest_cancel_requested_sync", lambda: False)
    monkeypatch.setattr(pipeline, "_detect_new_files", lambda docs_dir, files: (["a.txt"], {"a.txt": 1.0}))
    monkeypatch.setattr(pipeline, "_load_and_split", lambda *args, **kwargs: ([chunk], Counter({"a.txt": 1}), []))
    monkeypatch.setattr(pipeline, "get_milvus_client", lambda: object())
    monkeypatch.setattr(pipeline, "deduplicate_chunks", lambda *args, **kwargs: ([chunk], {}))
    monkeypatch.setattr(pipeline, "ensure_collection", lambda client: True)
    monkeypatch.setattr(pipeline, "encode_documents", lambda texts, cancel_event=None: ([[0.1]], [{}]))
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
