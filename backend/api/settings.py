"""Settings management API.
Provides endpoints to read, update, and reset runtime configuration.
Secret values are masked on read. Updates write to backend/.env and
signal whether a service restart is required.
"""

from pathlib import Path
import re
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from ..infrastructure.config import Settings, _ENV_FILE, settings as current_settings

router = APIRouter()

SECRET_KEYS = {
    "LLM_API_KEY",
    "EMBEDDING_API_KEY",
    "HF_TOKEN",
    "API_AUTH_TOKEN",
    "RERANKER_API_KEY",
    "LANGSMITH_API_KEY",
}

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
            result[k.strip()] = v.strip().split(" #")[0].strip()
    return result


def _write_env_updates(updates: Dict[str, Any]) -> None:
    env_path = Path(_ENV_FILE)
    lines = _read_env_lines()
    remaining = dict(updates)

    # First, replace existing keys
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in remaining:
            new_val = remaining.pop(k)
            lines[i] = f"{k}={new_val}"

    # Then append any new keys
    for k, v in remaining.items():
        lines.append(f"{k}={v}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@router.get("/settings")
def get_settings():
    """Return all configurable settings, masking secrets."""
    # Start with runtime settings dump
    try:
        data = current_settings.model_dump()
    except Exception:
        data = {}

    # Overlay with raw values from .env if any
    env_dict = _parse_env_dict()
    for k, v in env_dict.items():
        if k in Settings.model_fields:
            data[k] = v

    # Mask secrets
    for k in SECRET_KEYS:
        val = data.get(k, "")
        is_configured = bool(val and str(val).strip() and str(val).strip() != "*****")
        data[f"{k}_configured"] = is_configured
        data[k] = ""  # never echo secret

    return {
        "settings": data,
        "secret_keys": sorted(list(SECRET_KEYS)),
        "restart_keys": sorted(list(RESTART_KEYS)),
    }


class BulkSettingsUpdate(BaseModel):
    settings: Dict[str, Any]


@router.put("/settings")
@router.post("/settings")
def update_settings_bulk(payload: BulkSettingsUpdate):
    """Update multiple settings at once."""
    updates_to_write = {}
    restart_required = False
    env_dict = _parse_env_dict()

    for key, value in payload.settings.items():
        if key not in Settings.model_fields:
            continue

        # Handle secret keys
        if key in SECRET_KEYS:
            if value is None or value == "":
                # Skip updating secret if empty string or None provided (preserve existing)
                continue
            if value == "__CLEAR__":
                updates_to_write[key] = ""
            else:
                updates_to_write[key] = str(value).strip()
        else:
            # Format booleans cleanly
            if isinstance(value, bool):
                str_val = "true" if value else "false"
            elif value is None:
                str_val = ""
            else:
                str_val = str(value)
            updates_to_write[key] = str_val

        if key in RESTART_KEYS:
            restart_required = True

    if updates_to_write:
        _write_env_updates(updates_to_write)

    return {
        "success": True,
        "updated": list(updates_to_write.keys()),
        "restart_required": restart_required,
    }


class SingleSettingUpdate(BaseModel):
    value: Any


@router.put("/settings/{key}")
def update_single_setting(key: str, payload: SingleSettingUpdate):
    """Update a single configuration setting."""
    if key not in Settings.model_fields:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")

    res = update_settings_bulk(BulkSettingsUpdate(settings={key: payload.value}))
    return {
        "key": key,
        "updated": key in res["updated"],
        "restart_required": key in RESTART_KEYS,
    }


@router.post("/settings/reset")
def reset_settings():
    """Reset settings to default using .env.example, preserving DATABASE_URL."""
    backend_dir = Path(_ENV_FILE).parent
    example_path = backend_dir / ".env.example"
    current_env = _parse_env_dict()
    db_url = current_env.get("DATABASE_URL", "")

    if example_path.exists():
        content = example_path.read_text(encoding="utf-8")
        if db_url:
            # Keep database url intact
            content = re.sub(
                r"^DATABASE_URL=.*$",
                f"DATABASE_URL={db_url}",
                content,
                flags=re.MULTILINE,
            )
        Path(_ENV_FILE).write_text(content, encoding="utf-8")
    else:
        raise HTTPException(status_code=500, detail=".env.example not found")

    return {
        "status": "reset",
        "message": "Settings reset to defaults from .env.example (DATABASE_URL preserved)",
        "restart_required": True,
    }
