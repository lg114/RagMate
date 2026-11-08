"""Settings management API.
Provides endpoints to read and update runtime model connection configuration.
Enforces security, field whitelisting, atomic writes, and restart awareness.
"""

from hashlib import sha256
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
            data[k] = v

    # STRICT SECURITY: Never expose DATABASE_URL or database credentials
    data.pop("DATABASE_URL", None)

    # Mask secrets: return empty string and boolean configured flag
    for k in SECRET_KEYS:
        val = data.get(k, "")
        is_configured = bool(val and str(val).strip() and str(val).strip() != "*****")
        data[f"{k}_configured"] = is_configured
        data[k] = ""  # Never expose plaintext secret or mask placeholder to UI

    return {
        "settings": data,
        "editable_keys": sorted(list(EDITABLE_KEYS)),
        "secret_keys": sorted(list(SECRET_KEYS)),
        "restart_keys": sorted(list(RESTART_KEYS)),
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
    updates_to_write: Dict[str, str] = {}
    restart_required = False

    # Process each provided setting
    for key, value in payload.settings.items():
        if key not in EDITABLE_KEYS:
            raise HTTPException(
                status_code=400,
                detail=f"Field '{key}' is not editable via the settings API.",
            )

        # Handle secret keys: None or "" means preserve existing
        if key in SECRET_KEYS:
            if value is None or value == "":
                continue
            if value == "__CLEAR__":
                updates_to_write[key] = ""
                target_env[key] = ""
            else:
                str_val = str(value).strip()
                updates_to_write[key] = str_val
                target_env[key] = str_val
        else:
            if isinstance(value, bool):
                str_val = "true" if value else "false"
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

    return {
        "success": True,
        "updated": list(updates_to_write.keys()),
        "restart_required": restart_required,
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
