"""Dataset metadata audit; it does not make legal determinations."""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import TextExample


@dataclass(frozen=True)
class DatasetAudit:
    total_records: int
    duplicate_text_records: int
    undeclared_or_unapproved_licenses: tuple[str, ...]
    sources: tuple[str, ...]
    approved_records: int


def audit_dataset(
    records: list[TextExample], *, approved_licenses: set[str]
) -> DatasetAudit:
    """Summarize provenance metadata and flag licenses not explicitly approved."""
    seen: set[str] = set()
    duplicates = 0
    unapproved: set[str] = set()
    sources: set[str] = set()
    approved = 0
    for record in records:
        if not record.text.strip() or not record.source.strip() or not record.license.strip():
            raise ValueError("All records must contain text, source, and license metadata.")
        if record.text in seen:
            duplicates += 1
        seen.add(record.text)
        sources.add(record.source)
        if record.license in approved_licenses:
            approved += 1
        else:
            unapproved.add(record.license)
    return DatasetAudit(
        total_records=len(records),
        duplicate_text_records=duplicates,
        undeclared_or_unapproved_licenses=tuple(sorted(unapproved)),
        sources=tuple(sorted(sources)),
        approved_records=approved,
    )
