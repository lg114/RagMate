"""Settings management API.
Provides endpoints to read and update runtime model connection configuration.
Enforces security, field whitelisting, atomic writes, and restart awareness.
"""

from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
import time
import httpx
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..infrastructure.vector_store import (
    get_indexed_chunks_count,
    get_vector_index_metadata,
)
from ..infrastructure.config import (
    LOCAL_EMBEDDING_PROVIDERS,
    REMOTE_EMBEDDING_PROVIDERS,
    Settings,
    _ENV_FILE,
    settings as current_settings,
)

router = APIRouter()

# Fields allowed to be edited via the management UI/API.
# Operational parameters (CORS, timeouts, rate limits, chunking, database connection)
# are managed by deployment environment / .env and cannot be changed here.
EDITABLE_KEYS = {
    # Chat Model
    "LLM_MODEL",
    "LLM_API_BASE_URL",
    "LLM_API_KEY",
    "LLM_TEMPERATURE",
    "LLM_REASONING_EFFORT",
    "LLM_IS_LOCAL",
    # Embedding Model
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DEVICE",
    "EMBEDDING_NORMALIZE",
    # Reranker Model
    "RERANKER_PROVIDER",
    "RERANKER_MODEL",
    "RERANKER_API_BASE_URL",
    "RERANKER_API_KEY",
    # LangSmith Observability
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
}

SECRET_KEYS = {
    "LLM_API_KEY",
    "EMBEDDING_API_KEY",
    "HF_TOKEN",
    "API_AUTH_TOKEN",
    "RERANKER_API_KEY",
    "LANGSMITH_API_KEY",
}

# Modifying any of these keys requires a process restart to take effect
RESTART_KEYS = {
    "LLM_MODEL",
    "LLM_API_BASE_URL",
    "LLM_IS_LOCAL",
    "LLM_REASONING_EFFORT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_NORMALIZE",
    "EMBEDDING_API_BASE_URL",
    "RERANKER_PROVIDER",
    "RERANKER_MODEL",
    "RERANKER_API_BASE_URL",
    "DATABASE_URL",
    "VECTOR_TABLE",
    "DOCUMENTS_DIR",
    "CORS_ORIGINS",
    "API_AUTH_TOKEN",
    "LANGSMITH_TRACING",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "RATE_LIMIT_MAX",
    "RATE_LIMIT_WINDOW",
    "MAX_UPLOAD_SIZE",
    "AGENT_TIMEOUT",
}


def _read_env_lines() -> list[str]:
    env_path = Path(_ENV_FILE)
    if env_path.exists():
        return env_path.read_text(encoding="utf-8").splitlines()
    return []


def _parse_env_dict() -> Dict[str, str]:
    result = {}
    for line in _read_env_lines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            v_stripped = v.strip()
            if v_stripped.startswith("#"):
                clean_v = ""
            else:
                clean_v = re.split(r"\s+#", v_stripped)[0].strip()
            result[k.strip()] = clean_v
    return result


def _calculate_env_version() -> str:
    """Calculate a hash of current .env content for optimistic concurrency control."""
    env_path = Path(_ENV_FILE)
    if not env_path.exists():
        return "empty"
    content = env_path.read_bytes()
    return sha256(content).hexdigest()[:16]


def _validate_url(key: str, url_str: str) -> None:
    url_str = url_str.strip()
    if not url_str:
        return
    parsed = urlparse(url_str)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail=f"{key} must be a valid HTTP or HTTPS URL (got: {url_str!r})",
        )


def _validate_target_config(target: Dict[str, str], raw_updates: Dict[str, Any]) -> None:
    """Validate full target configuration before writing to disk."""
    # 1. Reject non-editable or unknown fields
    for k in raw_updates:
        if k not in EDITABLE_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{k}' is not editable via the settings API. Please configure it in your deployment environment.",
            )

    # 2. Temperature validation
    if "LLM_TEMPERATURE" in target:
        try:
            temp = float(target["LLM_TEMPERATURE"])
            if temp < 0.0 or temp > 2.0:
                raise ValueError()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="LLM_TEMPERATURE must be a number between 0.0 and 2.0",
            )

    # 3. LLM Reasoning effort validation
    if "LLM_REASONING_EFFORT" in target:
        effort = target["LLM_REASONING_EFFORT"].strip()
        if effort and effort not in ("none", "low", "medium", "high"):
            raise HTTPException(
                status_code=400,
                detail=f"LLM_REASONING_EFFORT must be 'none' or empty (got: {effort!r})",
            )

    # 4. Embedding provider validation
    valid_emb = LOCAL_EMBEDDING_PROVIDERS + REMOTE_EMBEDDING_PROVIDERS
    emb_provider = target.get("EMBEDDING_PROVIDER", "remote").strip().lower()
    if emb_provider not in valid_emb:
        raise HTTPException(
            status_code=400,
            detail=f"EMBEDDING_PROVIDER must be one of {valid_emb} (got: {emb_provider!r})",
        )

    # 5. Reranker provider validation
    reranker_provider = target.get("RERANKER_PROVIDER", "api").strip().lower()
    if reranker_provider not in ("api", "local"):
        raise HTTPException(
            status_code=400,
            detail=f"RERANKER_PROVIDER must be 'api' or 'local' (got: {reranker_provider!r})",
        )

    # 6. URL format validation
    for url_key in ("LLM_API_BASE_URL", "EMBEDDING_API_BASE_URL", "RERANKER_API_BASE_URL"):
        if url_key in target and target[url_key]:
            _validate_url(url_key, target[url_key])

    # 7. Cross-field requirements (only enforce for components being updated)
    if "EMBEDDING_PROVIDER" in raw_updates or "EMBEDDING_API_BASE_URL" in raw_updates:
        if emb_provider in REMOTE_EMBEDDING_PROVIDERS and not target.get("EMBEDDING_API_BASE_URL", "").strip():
            raise HTTPException(
                status_code=400,
                detail="EMBEDDING_API_BASE_URL is required when EMBEDDING_PROVIDER is 'remote' or 'openai'.",
            )

    if "RERANKER_PROVIDER" in raw_updates or "RERANKER_API_BASE_URL" in raw_updates:
        if reranker_provider == "api" and not target.get("RERANKER_API_BASE_URL", "").strip():
            raise HTTPException(
                status_code=400,
                detail="RERANKER_API_BASE_URL is required when RERANKER_PROVIDER is 'api'.",
            )


def _atomic_write_env(updates: Dict[str, str]) -> None:
    """Atomically write updates to .env file, preserving unedited keys and comments."""
    env_path = Path(_ENV_FILE)
    lines = _read_env_lines()
    remaining = dict(updates)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in remaining:
            new_val = remaining.pop(k)
            lines[i] = f"{k}={new_val}"

    # Append any newly added keys
    for k, v in remaining.items():
        lines.append(f"{k}={v}")

    content = "\n".join(lines) + "\n"

    # Write to a temporary file in the same directory, then atomic rename
    env_dir = env_path.parent
    env_dir.mkdir(parents=True, exist_ok=True)
    tf = tempfile.NamedTemporaryFile("w", dir=env_dir, delete=False, encoding="utf-8")
    try:
        tf.write(content)
        tf.flush()
        tf.close()
        os.replace(tf.name, str(env_path))
    except Exception:
        if os.path.exists(tf.name):
            try:
                os.remove(tf.name)
            except Exception:
                pass
        raise






def _parse_bool(val: Any) -> Optional[bool]:
    """Parse boolean from bool or string ('true', 'false', '1', '0', etc.)."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return None

def _get_field_sources(env_dict: Dict[str, str]) -> Dict[str, str]:
    """Detect whether each setting is managed by environment, .env file, or code default."""
    deploy_managed = set(filter(None, [k.strip() for k in os.environ.get("DEPLOY_MANAGED_KEYS", "").split(",")]))
    sources = {}
    for k in Settings.model_fields:
        if k not in EDITABLE_KEYS or k in deploy_managed:
            sources[k] = "environment"
        elif k in env_dict:
            sources[k] = "file"
        elif k in os.environ:
            sources[k] = "env"
        else:
            sources[k] = "default"
    return sources

def _calculate_pending_restart(env_dict: Dict[str, str]) -> list[str]:
    """Identify keys whose saved values on disk differ from the active runtime configuration."""
    pending = []
    for k in EDITABLE_KEYS:
        disk_val = env_dict.get(k)
        if disk_val is None:
            continue
        running_val = getattr(current_settings, k, None)
        if k in SECRET_KEYS:
            if disk_val and disk_val != running_val:
                pending.append(k)
        elif isinstance(running_val, bool) or k in ("LLM_IS_LOCAL", "EMBEDDING_NORMALIZE", "FAITHFULNESS_CHECK", "LANGSMITH_TRACING"):
            disk_bool = _parse_bool(disk_val)
            if disk_bool is not None and running_val is not None:
                if disk_bool != running_val:
                    pending.append(k)
            elif (disk_bool is not None) != (running_val is not None):
                pending.append(k)
        elif isinstance(running_val, (int, float)):
            try:
                if float(disk_val) != float(running_val):
                    pending.append(k)
            except ValueError:
                pending.append(k)
        else:
            if disk_val.strip() != (str(running_val) if running_val is not None else "").strip():
                pending.append(k)
    return sorted(pending)

@router.get("/settings")
def get_settings():
    """Return runtime configuration. Mask secrets completely and exclude DATABASE_URL."""
    try:
        data = current_settings.model_dump()
    except Exception:
        data = {}

    # Overlay with raw values from .env if any
    env_dict = _parse_env_dict()
    for k, v in env_dict.items():
        if k in Settings.model_fields:
            if k in ("LLM_IS_LOCAL", "EMBEDDING_NORMALIZE", "FAITHFULNESS_CHECK", "LANGSMITH_TRACING"):
                b = _parse_bool(v)
                data[k] = b if b is not None else False
            else:
                data[k] = v

    try:
        from backend.application.ingest.loaders import SUPPORTED_EXTENSIONS
        supported_exts = sorted(list(SUPPORTED_EXTENSIONS))
    except Exception:
        supported_exts = [".csv", ".docx", ".epub", ".htm", ".html", ".json", ".md", ".pdf", ".pptx", ".rtf", ".txt", ".xml"]

    # STRICT SECURITY: Never expose DATABASE_URL or database credentials
    data.pop("DATABASE_URL", None)

    # Mask secrets: return empty string and boolean configured flag
    for k in SECRET_KEYS:
        val = data.get(k, "")
        is_configured = bool(val and str(val).strip() and str(val).strip() != "*****")
        data[f"{k}_configured"] = is_configured
        data[k] = ""  # Never expose plaintext secret or mask placeholder to UI

    # Source detection
    field_sources = _get_field_sources(env_dict)

    # Vector index metadata and locking status
    try:
        chunk_count = get_indexed_chunks_count(current_settings.VECTOR_TABLE)
        meta = get_vector_index_metadata(current_settings.VECTOR_TABLE)
        has_indexed = chunk_count > 0
    except Exception:
        chunk_count = 0
        meta = None
        has_indexed = False

    index_status = {
        "has_indexed_docs": has_indexed,
        "chunk_count": chunk_count,
        "embedding_provider": meta["embedding_provider"] if meta else (current_settings.EMBEDDING_PROVIDER if not has_indexed else None),
        "embedding_model": meta["embedding_model"] if meta else (current_settings.EMBEDDING_MODEL if not has_indexed else None),
        "embedding_base_url": meta["embedding_base_url"] if meta else (current_settings.EMBEDDING_API_BASE_URL if not has_indexed else None),
        "dimension": meta["dimension"] if meta else 1024,
        "compatibility_unconfirmed": has_indexed and (meta is None),
        "locked_keys": sorted(["EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_BASE_URL"]) if has_indexed else [],
    }

    pending_restart = _calculate_pending_restart(env_dict)

    return {
        "settings": data,
        "editable_keys": sorted(list(EDITABLE_KEYS)),
        "secret_keys": sorted(list(SECRET_KEYS)),
        "restart_keys": sorted(list(RESTART_KEYS)),
        "field_sources": field_sources,
        "pending_restart": pending_restart,
        "index_status": index_status,
        "version": _calculate_env_version(),
    }


class BulkSettingsUpdate(BaseModel):
    settings: Dict[str, Any]
    version: Optional[str] = None


@router.put("/settings")
@router.post("/settings")
def update_settings_bulk(payload: BulkSettingsUpdate):
    """Update multiple model connection settings at once with atomic write and optimistic concurrency."""
    current_version = _calculate_env_version()
    if payload.version and payload.version != current_version:
        raise HTTPException(
            status_code=409,
            detail="Configuration was modified in another session. Please refresh settings and try again.",
        )

    env_dict = _parse_env_dict()
    target_env = dict(env_dict)
    # 1. Reject non-editable or unknown fields first
    for key in payload.settings:
        if key not in EDITABLE_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{key}' is not editable via the settings API.",
            )

    # 2. Environment source protection: do not allow modifying settings managed by deployment environment
    sources = _get_field_sources(env_dict)
    for key in payload.settings:
        if sources.get(key) == "environment":
            raise HTTPException(
                status_code=400,
                detail=f"Field '{key}' is managed by the deployment environment and cannot be updated via the web interface.",
            )

    # 3. Vector index protection: prevent changing embedding model/provider/URL when indexed documents exist
    LOCKED_INDEX_KEYS = {"EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_API_BASE_URL"}
    locked_attempted = [k for k in payload.settings if k in LOCKED_INDEX_KEYS]
    if locked_attempted:
        try:
            chunk_count = get_indexed_chunks_count(current_settings.VECTOR_TABLE)
        except Exception:
            chunk_count = 0
        if chunk_count > 0:
            meta = get_vector_index_metadata(current_settings.VECTOR_TABLE)
            for k in locked_attempted:
                new_val = str(payload.settings[k]).strip()
                current_val = str(getattr(current_settings, k, "")).strip()
                expected_val = (str(meta.get(k.lower()) or "")).strip() if meta else current_val
                if new_val != expected_val and new_val != current_val:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Cannot modify '{k}' while indexed documents ({chunk_count} chunks) exist in the vector store. "
                            "Changing the embedding model or endpoint requires re-indexing all documents."
                        ),
                    )

    updates_to_write: Dict[str, str] = {}
    restart_required = False

    # Process each provided setting
    for key, value in payload.settings.items():
        if key not in EDITABLE_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{key}' is not editable via the settings API.",
            )

        # Handle secret keys: None, "", or mask means preserve existing
        if key in SECRET_KEYS:
            if value is None or value == "" or str(value).startswith("••"):
                continue
            if value == "__CLEAR__":
                updates_to_write[key] = ""
                target_env[key] = ""
            else:
                str_val = str(value).strip()
                updates_to_write[key] = str_val
                target_env[key] = str_val
        else:
            b_val = _parse_bool(value)
            if b_val is not None and (isinstance(value, bool) or key in ("LLM_IS_LOCAL", "EMBEDDING_NORMALIZE", "FAITHFULNESS_CHECK", "LANGSMITH_TRACING")):
                str_val = "true" if b_val else "false"
            elif value is None:
                str_val = ""
            else:
                str_val = str(value).strip()
            updates_to_write[key] = str_val
            target_env[key] = str_val

        if key in RESTART_KEYS:
            restart_required = True

    # Validate the full target configuration before touching .env
    _validate_target_config(target_env, payload.settings)

    if updates_to_write:
        _atomic_write_env(updates_to_write)

    new_env = _parse_env_dict()
    new_pending = _calculate_pending_restart(new_env)

    return {
        "success": True,
        "updated": list(updates_to_write.keys()),
        "restart_required": restart_required or bool(new_pending),
        "pending_restart": new_pending,
        "version": _calculate_env_version(),
    }


class SingleSettingUpdate(BaseModel):
    value: Any
    version: Optional[str] = None


@router.put("/settings/{key}")
def update_single_setting(key: str, payload: SingleSettingUpdate):
    """Update a single configuration setting."""
    if key not in EDITABLE_KEYS:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{key}' is not editable via the settings API.",
        )

    res = update_settings_bulk(
        BulkSettingsUpdate(settings={key: payload.value}, version=payload.version)
    )
    return {
        "key": key,
        "updated": key in res["updated"],
        "restart_required": key in RESTART_KEYS,
        "version": res["version"],
    }


def _sanitize_error(exc: Exception) -> str:
    """Sanitize error messages so secrets or stack traces never leak to client."""
    msg = str(exc)
    msg = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [MASKED]", msg)
    msg = re.sub(r"key=[A-Za-z0-9_\-\.]+", "key=[MASKED]", msg)
    if len(msg) > 160:
        msg = msg[:160] + "..."
    return msg


def _format_http_error(resp: httpx.Response) -> str:
    """Extract readable message from JSON or text error responses."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and "message" in err:
                return _sanitize_error(Exception(str(err["message"])))
            if isinstance(err, str):
                return _sanitize_error(Exception(err))
            if "message" in data:
                return _sanitize_error(Exception(str(data["message"])))
            if "detail" in data:
                return _sanitize_error(Exception(str(data["detail"])))
    except Exception:
        pass
    text = (resp.text or "").strip()
    if text:
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = " ".join(clean.split())
        if clean:
            return _sanitize_error(Exception(clean))
    return f"Endpoint not found ({resp.status_code}). Check Base URL path and Model Name."


def _normalize_endpoint(base_url: str, default_endpoint: str, suffix: str) -> str:
    if not base_url or not base_url.strip():
        return default_endpoint
    clean = base_url.strip().rstrip("/")
    if clean.endswith(suffix):
        return clean
    return f"{clean}{suffix}"


class ConnectionTestPayload(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None


@router.post("/settings/test/llm")
async def test_llm_connection(payload: Optional[ConnectionTestPayload] = None):
    """Test Chat Model connection with a short prompt. Timeout: 5 seconds."""
    env_dict = _parse_env_dict()
    model = (payload.model if payload and payload.model else None) or env_dict.get("LLM_MODEL") or current_settings.LLM_MODEL
    base_url = (payload.base_url if payload and payload.base_url else None) or env_dict.get("LLM_API_BASE_URL") or current_settings.LLM_API_BASE_URL
    api_key = (payload.api_key if payload and payload.api_key else None) or env_dict.get("LLM_API_KEY") or current_settings.LLM_API_KEY

    if not model:
        return {"ok": False, "detail": "LLM_MODEL is not configured."}

    start = time.perf_counter()
    endpoint = _normalize_endpoint(base_url, "https://api.openai.com/v1/chat/completions", "/chat/completions")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req_body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(endpoint, json=req_body, headers=headers)
        latency = int((time.perf_counter() - start) * 1000)
        if resp.status_code == 200:
            return {"ok": True, "latency_ms": latency, "model": model, "detail": f"Success ({latency}ms)"}
        return {
            "ok": False,
            "latency_ms": latency,
            "model": model,
            "detail": f"HTTP {resp.status_code}: {_format_http_error(resp)}",
        }
    except httpx.TimeoutException:
        latency = int((time.perf_counter() - start) * 1000)
        return {"ok": False, "latency_ms": latency, "model": model, "detail": "Request timed out (>8s)"}
    except Exception as e:
        latency = int((time.perf_counter() - start) * 1000)
        detail = _sanitize_error(e) or type(e).__name__
        return {"ok": False, "latency_ms": latency, "model": model, "detail": detail}


@router.post("/settings/test/embedding")
async def test_embedding_connection(payload: Optional[ConnectionTestPayload] = None):
    """Test Embedding Model connection and verify 1024-dimension compatibility. Timeout: 5s."""
    env_dict = _parse_env_dict()
    provider = (payload.provider if payload and payload.provider else None) or env_dict.get("EMBEDDING_PROVIDER") or current_settings.EMBEDDING_PROVIDER
    model = (payload.model if payload and payload.model else None) or env_dict.get("EMBEDDING_MODEL") or current_settings.EMBEDDING_MODEL
    base_url = (payload.base_url if payload and payload.base_url else None) or env_dict.get("EMBEDDING_API_BASE_URL") or current_settings.EMBEDDING_API_BASE_URL
    api_key = (payload.api_key if payload and payload.api_key else None) or env_dict.get("EMBEDDING_API_KEY") or current_settings.EMBEDDING_API_KEY

    start = time.perf_counter()
    if provider in LOCAL_EMBEDDING_PROVIDERS:
        try:
            from FlagEmbedding import BGEM3FlagModel
            latency = int((time.perf_counter() - start) * 1000)
            return {
                "ok": True,
                "latency_ms": latency,
                "model": model,
                "dimension": 1024,
                "detail": f"In-process FlagEmbedding available ({latency}ms)",
            }
        except ImportError:
            return {"ok": False, "detail": "FlagEmbedding package is not installed. Run: pip install -e \".[local]\""}
        except Exception as e:
            return {"ok": False, "detail": _sanitize_error(e)}

    if not base_url:
        return {"ok": False, "detail": "EMBEDDING_API_BASE_URL is required for remote embedding."}

    endpoint = _normalize_endpoint(base_url, "https://api.openai.com/v1/embeddings", "/embeddings")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    emb_payload = {"model": model, "input": ["healthcheck"]}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(endpoint, json=emb_payload, headers=headers)
        latency = int((time.perf_counter() - start) * 1000)
        if resp.status_code != 200:
            return {
                "ok": False,
                "latency_ms": latency,
                "model": model,
                "detail": f"HTTP {resp.status_code}: {_format_http_error(resp)}",
            }
        data = resp.json()
        items = data.get("data") or []
        if not items or "embedding" not in items[0]:
            return {"ok": False, "latency_ms": latency, "detail": "Response missing embedding vector."}
        dim = len(items[0]["embedding"])
        if dim != 1024:
            return {
                "ok": False,
                "latency_ms": latency,
                "dimension": dim,
                "detail": f"Incompatible dimension: returned {dim} dims, but RagMate vector store requires exactly 1024 dimensions.",
            }
        return {"ok": True, "latency_ms": latency, "dimension": dim, "model": model, "detail": f"Success ({dim} dims, {latency}ms)"}
    except httpx.TimeoutException:
        latency = int((time.perf_counter() - start) * 1000)
        return {"ok": False, "latency_ms": latency, "model": model, "detail": "Request timed out (>8s)"}
    except Exception as e:
        latency = int((time.perf_counter() - start) * 1000)
        detail = _sanitize_error(e) or type(e).__name__
        return {"ok": False, "latency_ms": latency, "model": model, "detail": detail}


@router.post("/settings/test/reranker")
async def test_reranker_connection(payload: Optional[ConnectionTestPayload] = None):
    """Test Reranker Model connection with a short query. Timeout: 5 seconds."""
    env_dict = _parse_env_dict()
    provider = (payload.provider if payload and payload.provider else None) or env_dict.get("RERANKER_PROVIDER") or current_settings.RERANKER_PROVIDER
    model = (payload.model if payload and payload.model else None) or env_dict.get("RERANKER_MODEL") or current_settings.RERANKER_MODEL
    base_url = (payload.base_url if payload and payload.base_url else None) or env_dict.get("RERANKER_API_BASE_URL") or current_settings.RERANKER_API_BASE_URL
    api_key = (payload.api_key if payload and payload.api_key else None) or env_dict.get("RERANKER_API_KEY") or current_settings.RERANKER_API_KEY

    start = time.perf_counter()
    if provider == "local":
        try:
            from sentence_transformers import CrossEncoder
            latency = int((time.perf_counter() - start) * 1000)
            return {"ok": True, "latency_ms": latency, "model": model, "detail": f"Local CrossEncoder available ({latency}ms)"}
        except ImportError:
            return {"ok": False, "detail": "Sentence-transformers package is not installed. Run: pip install -e \".[local]\""}
        except Exception as e:
            return {"ok": False, "detail": _sanitize_error(e)}

    if not base_url:
        return {"ok": False, "detail": "RERANKER_API_BASE_URL is required for remote reranker."}

    endpoint = _normalize_endpoint(base_url, "", "/rerank")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    rerank_payload = {
        "model": model,
        "query": "test query",
        "documents": ["candidate doc 1", "candidate doc 2"],
        "top_n": 2,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(endpoint, json=rerank_payload, headers=headers)
        latency = int((time.perf_counter() - start) * 1000)
        if resp.status_code != 200:
            return {
                "ok": False,
                "latency_ms": latency,
                "model": model,
                "detail": f"HTTP {resp.status_code}: {_format_http_error(resp)}",
            }
        data = resp.json()
        results = data.get("results") or []
        if not results:
            return {"ok": False, "latency_ms": latency, "detail": "Response missing rerank results."}
        return {"ok": True, "latency_ms": latency, "model": model, "detail": f"Success ({latency}ms)"}
    except httpx.TimeoutException:
        latency = int((time.perf_counter() - start) * 1000)
        return {"ok": False, "latency_ms": latency, "model": model, "detail": "Request timed out (>8s)"}
    except Exception as e:
        latency = int((time.perf_counter() - start) * 1000)
        detail = _sanitize_error(e) or type(e).__name__
        return {"ok": False, "latency_ms": latency, "model": model, "detail": detail}

