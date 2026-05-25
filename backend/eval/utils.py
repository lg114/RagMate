"""评估工具函数。"""
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TESTSETS_DIR = PROJECT_ROOT / "eval" / "testsets"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


def resolve_path(p: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_score(value) -> float | None:
    """Convert NaN/Inf to None for JSON serialization."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return float(value)
