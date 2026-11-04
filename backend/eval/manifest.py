"""实验 manifest：一次评测运行的完整配置快照。

A/B 对比只在“除一个变量外全部相同”时成立。manifest 把题库版本（文件
SHA-256）、全部检索相关配置、harness 参数写进每次运行的报告旁，两个运
行的差异用 `diff_manifests` 一眼看清。索引侧信息（集合行数）由调用方在
向量表行数由调用方在 Postgres 可达时另行采集——manifest 本身不依赖任何外部服务。
"""
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

# 改动其中任何一项都可能改变检索结果，因此全部记入 manifest。
RETRIEVAL_SETTINGS_KEYS = [
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "HYBRID_SEARCH_ENABLED",
    "FUSION_MODE",
    "RRF_K",
    "FUSION_DENSE_WEIGHT",
    "DROP_RATIO_SEARCH",
    "RERANKER_PROVIDER",
    "RERANKER_MODEL",
    "RERANK_CANDIDATES",
    "FINAL_CONTEXT_K",
    "RERANK_SCORE_THRESHOLD",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE_PARENT",
    "CHUNK_OVERLAP_PARENT",
    "CHUNKING_STRATEGY",
    "MULTI_QUERY_ENABLED",
    "MULTI_QUERY_COUNT",
    "CONTEXTUAL_COMPRESSION",
    "COMPRESSION_SCORE_THRESHOLD",
    "COMPRESSION_MIN_CHARS",
    "DYNAMIC_THRESHOLD_RATIO",
    "HIGH_SCORE_RATIO",
    "MAX_PER_SOURCE",
    "MIN_PER_SOURCE",
    "SCORE_GAP_THRESHOLD",
    "SOURCE_DOMINANCE_THRESHOLD",
    "SOURCE_DOMINANCE_BOOST",
    "VECTOR_TABLE",
]


def sha256_file(path) -> str:
    """文件内容的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def get_git_commit(repo_root=None) -> str | None:
    """尽力而为地取当前 git commit；不在 git 仓库里时返回 None。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def snapshot_retrieval_settings() -> dict:
    """当前生效的检索相关配置快照。"""
    from backend.infrastructure.config import settings

    return {key: getattr(settings, key) for key in RETRIEVAL_SETTINGS_KEYS}


def build_manifest(*, testset_path, stages, over_fetch, ks, repeat,
                   n_questions, git_commit=None) -> dict:
    """组装一次运行的 manifest。testset_path 参与 SHA-256 计算。"""
    testset_path = Path(testset_path)
    return {
        "generated_at": datetime.now().isoformat(),
        "git_commit": git_commit if git_commit is not None else get_git_commit(),
        "testset": {
            "path": str(testset_path),
            "sha256": sha256_file(testset_path),
            "n_questions": n_questions,
        },
        "retrieval_config": snapshot_retrieval_settings(),
        "harness": {
            "stages": list(stages),
            "over_fetch": over_fetch,
            "ks": list(ks),
            "repeat": repeat,
        },
    }


def diff_manifests(old: dict, new: dict) -> dict:
    """对比两个 manifest，返回变化的配置项 {key: {"old": .., "new": ..}}。

    只比 retrieval_config 与 harness 两节：题库 SHA 变化单列为
    testset_changed，提醒对比双方题库已不同。
    """
    changes = {}
    old_config = old.get("retrieval_config", {})
    new_config = new.get("retrieval_config", {})
    for key in sorted(set(old_config) | set(new_config)):
        if old_config.get(key) != new_config.get(key):
            changes[f"retrieval_config.{key}"] = {"old": old_config.get(key), "new": new_config.get(key)}
    old_harness = old.get("harness", {})
    new_harness = new.get("harness", {})
    for key in sorted(set(old_harness) | set(new_harness)):
        if old_harness.get(key) != new_harness.get(key):
            changes[f"harness.{key}"] = {"old": old_harness.get(key), "new": new_harness.get(key)}
    old_sha = old.get("testset", {}).get("sha256")
    new_sha = new.get("testset", {}).get("sha256")
    if old_sha != new_sha:
        changes["testset_changed"] = {"old": old_sha, "new": new_sha}
    return changes
