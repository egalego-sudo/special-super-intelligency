import pytest

from special_super_intelligency.dataset import TextExample, encode_examples, load_jsonl
from special_super_intelligency.tokenizer import ByteTokenizer


def test_loads_jsonl_with_provenance(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text":"नमस्ते","source":"author","license":"CC0"}\n',
        encoding="utf-8",
    )
    assert load_jsonl(str(path)) == [TextExample("नमस्ते", "author", "CC0")]


def test_rejects_missing_license(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"hello","source":"unknown"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        load_jsonl(str(path))


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(str(path))


def test_encodes_examples():
    examples = [TextExample("AI", "test", "CC0")]
    assert encode_examples(examples, ByteTokenizer()) == [[65, 73]]
