import pytest

from special_super_intelligency.tokenizer import ByteTokenizer


def test_round_trip_ascii():
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode("Quantum AI")) == "Quantum AI"


def test_round_trip_multilingual_unicode():
    tokenizer = ByteTokenizer()
    text = "नमस्ते 🌌 — QUANTUM AI"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_tokenizer_is_deterministic():
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("same input") == tokenizer.encode("same input")


def test_rejects_out_of_range_token():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([256])


def test_rejects_invalid_utf8_sequence():
    with pytest.raises(ValueError):
        ByteTokenizer().decode([0xFF])


def test_rejects_non_string_input():
    with pytest.raises(TypeError):
        ByteTokenizer().encode(123)  # type: ignore[arg-type]
