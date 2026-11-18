import pytest

from special_super_intelligency.audit import audit_dataset
from special_super_intelligency.dataset import TextExample


def test_audit_counts_records_duplicates_and_licenses():
    records = [
        TextExample("same", "source-a", "CC0"),
        TextExample("same", "source-b", "Unknown"),
    ]
    result = audit_dataset(records, approved_licenses={"CC0"})
    assert result.total_records == 2
    assert result.duplicate_text_records == 1
    assert result.approved_records == 1
    assert result.undeclared_or_unapproved_licenses == ("Unknown",)
    assert result.sources == ("source-a", "source-b")


def test_audit_rejects_missing_metadata():
    with pytest.raises(ValueError):
        audit_dataset([TextExample("text", "", "CC0")], approved_licenses={"CC0"})


def test_audit_empty_dataset_is_well_defined():
    result = audit_dataset([], approved_licenses={"CC0"})
    assert result.total_records == 0
    assert result.approved_records == 0
