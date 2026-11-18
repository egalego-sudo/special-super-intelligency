import pytest

from special_super_intelligency.knowledge import KnowledgeRecord, KnowledgeStore


def test_add_and_search_is_provenance_aware():
    store = KnowledgeStore()
    store.add(KnowledgeRecord("r1", "Quantum computing notes", "user-provided"))
    results = store.search("QUANTUM")
    assert len(results) == 1
    assert results[0].source == "user-provided"


def test_delete_removes_exact_record():
    store = KnowledgeStore()
    store.add(KnowledgeRecord("r1", "note", "manual"))
    assert store.delete("r1") is True
    assert store.delete("r1") is False
    assert len(store) == 0


def test_duplicate_ids_are_rejected():
    store = KnowledgeStore()
    record = KnowledgeRecord("r1", "one", "source")
    store.add(record)
    with pytest.raises(ValueError):
        store.add(record)


def test_empty_search_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeStore().search(" ")
