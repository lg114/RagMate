import re


def canonical_source(source: str) -> str:
    """Normalize near-duplicate source filenames for retrieval and evaluation."""
    return re.sub(r"_\d+\.", ".", source or "").lower()
