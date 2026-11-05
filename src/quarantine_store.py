"""Separate JSON audit stores for quarantine and pending review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import read_json_list, write_json


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def add(self, record: dict[str, Any]) -> None:
        records = read_json_list(self.path)
        records.append(record)
        write_json(self.path, records)

    def all(self) -> list[dict[str, Any]]:
        return read_json_list(self.path)

    def count(self) -> int:
        return len(self.all())

    def remove(self, memory_id: str) -> bool:
        """Remove one audit record by its stable memory/event ID."""
        records = self.all()
        remaining = [
            record for record in records
            if record.get("memory_id") != memory_id and record.get("event_id") != memory_id
        ]
        if len(remaining) == len(records):
            return False
        write_json(self.path, remaining)
        return True

    def clear(self) -> None:
        write_json(self.path, [])

