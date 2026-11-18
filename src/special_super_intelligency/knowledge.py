"""Explicit, provenance-aware in-memory knowledge store prototype."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeRecord:
    record_id: str
    content: str
    source: str


class KnowledgeStore:
    """Store only records explicitly added by the caller; no automatic capture."""

    def __init__(self) -> None:
        self._records: dict[str, KnowledgeRecord] = {}

    def add(self, record: KnowledgeRecord) -> None:
        if not all(value.strip() for value in (
            record.record_id, record.content, record.source
        )):
            raise ValueError("record_id, content, and source must be non-empty.")
        if record.record_id in self._records:
            raise ValueError(f"Record {record.record_id!r} already exists.")
        self._records[record.record_id] = record

    def search(self, query: str) -> list[KnowledgeRecord]:
        if not query.strip():
            raise ValueError("query must be non-empty.")
        needle = query.casefold()
        return [
            record for record in self._records.values()
            if needle in record.content.casefold()
        ]

    def delete(self, record_id: str) -> bool:
        """Delete a record by exact ID; return whether a record was present."""
        return self._records.pop(record_id, None) is not None

    def __len__(self) -> int:
        return len(self._records)
