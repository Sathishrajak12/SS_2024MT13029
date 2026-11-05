"""Small reusable helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing the meaning of a memory."""
    return " ".join(text.strip().split())


def fingerprint_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)


def read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def important_terms(text: str) -> set[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "for",
        "with", "on", "by", "be", "can", "may", "from", "this", "that", "when",
        "after", "before", "into", "as", "at", "it", "only", "all", "should",
    }
    words = re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
    return {word for word in words if word not in stop_words}


def extract_entities(text: str) -> list[str]:
    """Extract a small set of explainable entities; no cloud NLP is required."""
    entities = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
    entities.update(re.findall(r"(?:/[\w.-]+){2,}", text))
    vocabulary = {
        "diagnostic report", "audit log", "support contact", "administrator",
        "approval", "external destination", "customer record", "configuration",
        "maintenance window", "incident ticket", "access request",
    }
    lower = text.lower()
    entities.update(term.title() for term in vocabulary if term in lower)
    return sorted(entities)


def make_memory_id(existing_count: int) -> str:
    return f"MEM-{existing_count + 1:04d}"

