# Settings Page Redesign Proposal for RagMate

## 1. Overview
The **Settings** page is the only place where operators can view and modify runtime configuration that currently lives in a `.env` file. To make the system **discoverable, safe, and operable**, we propose a **single‑page UI** built with plain HTML/CSS/JS (no extra framework) that mirrors the logical groups in `backend/infrastructure/config.py`.

---

## 2. UI Layout (Desktop / Responsive)
```
+-----------------------------------------------------------+
| Header (logo, App name, Settings button)                  |
+-----------------------------------------------------------+
| Sidebar (collapsible sections)                           |
|   • Authentication   • Rate limiting   • Model & Embed   |
|   • Retrieval         • Ingestion       • Chat & Session |
|   • System            • Miscellaneous                 |
+-----------------------------------------------------------+
| Main panel – active section content (form)                |
|   • Section title                                         |
|   • Brief description / tooltip                           |
|   • Settings grid (label | input | help)                 |
|   • Save / Reset buttons (per‑section)                    |
+-----------------------------------------------------------+
| Footer (status bar, last saved timestamp)                |
+-----------------------------------------------------------+
```

*The sidebar stays visible on wide screens and collapses into a hamburger menu on mobile.*

### 2.1 Section Example – **Authentication**
| Setting | Input Type | Description |
|---|---|---|
| `API_AUTH_TOKEN` | password (masked) with eye‑toggle | Token required for API calls. Empty = no auth. |
| `LLM_API_KEY` | password (masked) | Key for the LLM provider. |
| `EMBEDDING_API_KEY` | password (masked) | Optional key for remote embedding services. |
| `HF_TOKEN` | password (masked) | HuggingFace token (used only when provider = `huggingface`). |
| **Save** | button | Persists changes via `PUT /settings/{key}`. |
| **Reset to defaults** | button | Calls `POST /settings/reset` (new endpoint). |

### 2.2 Input Types Mapping
| Pydantic type | UI widget |
|---|---|
| `str` (secret) | `<input type="password">` with show/hide toggle |
| `str` (regular) | `<input type="text">` |
| `bool` | `<input type="checkbox">` |
| `int` / `float` | `<input type="number">` (step/ min‑max derived from `Field` constraints) |
| `enum`‑like (e.g. `EMBEDDING_PROVIDER`) | `<select>` with allowed values |

---

## 3. Backend API Changes
Create **`backend/api/settings.py`** with a router that:
1. **GET `/settings`** – Returns the full configuration as a JSON dict. Secret fields are masked (`"*****"`).
2. **PUT `/settings/{key}`** – Validates the new value against the `Settings` model (re‑use Pydantic validators). Writes the key/value pair to `.env` (append or replace the line) and returns the updated dict.  
   *Note*: Runtime settings are **not hot‑reloaded**; the response includes a flag `restart_required: true` for keys that need a process restart.
3. **POST `/settings/reset`** – Overwrites `.env` with default values taken from `Settings.__fields__` defaults and returns the refreshed dict.
4. **Security** – Protect all routes with the existing token middleware (`_PROTECTED_PREFIXES`).
5. **Include router** in `backend/app.py` (`app.include_router(settings_router, prefix="/api")`).

### 3.1 Minimal Implementation Sketch
```python
# backend/api/settings.py
from fastapi import APIRouter, HTTPException
from ..infrastructure.config import Settings, _ENV_FILE
from pathlib import Path

router = APIRouter()

def _load_settings():
    return Settings()

@router.get("/settings")
def get_settings():
    s = _load_settings()
    data = s.model_dump()
    # mask secrets
    for k in ["LLM_API_KEY", "EMBEDDING_API_KEY", "HF_TOKEN", "API_AUTH_TOKEN", "RERANKER_API_KEY"]:
        if k in data:
            data[k] = "*****"
    return data

@router.put("/settings/{key}")
def update_setting(key: str, value: str):
    if not hasattr(Settings, key):
        raise HTTPException(status_code=404, detail="Setting not found")
    env_path = Path(_ENV_FILE)
    lines = env_path.read_text(encoding="utf-8").splitlines()
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"key": key, "value": value, "restart_required": True}
```
*The real implementation will reuse the existing validators and return a proper Pydantic model.*

---

## 4. Front‑end Integration (static JS)
1. **Add a Settings button** to `frontend/index.html` navigation bar linking to `#settings` or opening a modal.
2. **Create `settings.html`** (or a hidden `<div id="settingsModal">`) that contains:
   * a sidebar generated from a static JSON describing sections & fields.
   * a form that populates values via `fetchJson('/api/settings')`.
   * on **Save**, call `fetch('/api/settings/<key>', {method: 'PUT', body: JSON.stringify({value})})`.
3. **Utility functions** – re‑use `fetchJson` from `js/shell.js`.
4. **Client‑side validation** – respect the `gt`, `lt`, `ge`, `le` constraints defined in `config.py`.  The backend will also enforce them, but early feedback improves UX.
5. **Accessibility** – proper `<label for>` bindings, ARIA descriptions for secret fields, keyboard‑navigable sidebar.
6. **Feedback UI** – toast messages for success/failure, and a banner indicating *“Restart required for changes to take effect”* when applicable.

---

## 5. User Experience Flow
1. User clicks **Settings** → sidebar expands, default section **Authentication** loads.
2. Existing values appear; secret fields show masked bullets.
3. User edits a value, clicks **Save** → UI disables the row, shows a spinner, then a success toast.
4. If the changed key needs a restart, a non‑intrusive banner appears with a *Restart now* button (calls a new endpoint `POST /restart` that sends a signal to the running process, or simply instructs the operator to restart the service).
5. *Reset to defaults* clears custom values and writes the defaults back to `.env`.

---

## 6. Documentation Updates
* Add a **Settings UI** section to `README.md` with screenshots, a quick‑start guide, and instructions on how to apply changes (service restart).
* Document the new API endpoints in `backend/api/openapi.yaml` (FastAPI does this automatically).  
* Mention that secrets are never logged nor returned in clear text.

---

## 7. Timeline & Risk Assessment
| Milestone | Estimate | Risk |
|---|---|---|
| Backend router + tests | 1 day | Low – straightforward file I/O.
| Front‑end modal & sidebar | 1 day | Medium – need to ensure responsive CSS without a framework.
| Validation & UI polish (tooltips, masks) | 0.5 day | Low.
| Documentation & screenshots | 0.5 day | Low.
| End‑to‑end manual testing (restart flow) | 0.5 day | Medium – ensure `.env` changes are respected after restart.

---

## 8. Next Steps
1. **Confirm the design** with the stakeholder (you).  If any visual style or additional sections are required, let me know.
2. Once approved, I will generate the concrete code changes for:
   * `backend/api/settings.py`
   * inclusion in `backend/app.py`
   * new/updated HTML + JS files.
3. Run the test suite (if any) and perform a manual UI verification.

---

*This proposal stays within the existing zero‑build static‑site architecture, adds only a single FastAPI router, and respects security by masking secrets.*
