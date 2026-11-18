from special_super_intelligency.attention import causal_self_attention  # noqa: I001


IDENTITY = [[1.0, 0.0], [0.0, 1.0]]


def assert_raises(exception, callback):
    try:
        callback()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}.")


def test_attention_returns_same_sequence_shape():
    sequence = [[1.0, 0.0], [0.0, 1.0]]
    output = causal_self_attention(sequence, IDENTITY, IDENTITY, IDENTITY)
    assert len(output) == 2
    assert all(len(token) == 2 for token in output)


def test_attention_is_causal():
    first = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    changed_future = [[1.0, 0.0], [0.0, 1.0], [-50.0, 70.0]]
    out_a = causal_self_attention(first, IDENTITY, IDENTITY, IDENTITY)
    out_b = causal_self_attention(changed_future, IDENTITY, IDENTITY, IDENTITY)
    assert out_a[0] == out_b[0]
    assert out_a[1] == out_b[1]


def test_attention_rejects_empty_sequence():
    assert_raises(ValueError, lambda: causal_self_attention([], IDENTITY, IDENTITY, IDENTITY))


def test_attention_rejects_inconsistent_width():
    assert_raises(
        ValueError,
        lambda: causal_self_attention([[1.0, 0.0], [1.0]], IDENTITY, IDENTITY, IDENTITY),
    )


def test_attention_rejects_bad_projection_shape():
    assert_raises(
        ValueError,
        lambda: causal_self_attention([[1.0, 0.0]], [[1.0]], IDENTITY, IDENTITY),
    )
