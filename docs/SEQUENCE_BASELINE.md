# Tiny Byte-Level Sequence Training Baseline

**Creator by Bittu Pandit**

The sequence module converts byte-token sequences into sliding-window supervised examples and trains the repository's small MLP to regress the next byte ID.

This provides an executable, independent, trainable sequence-prediction experiment. It is intentionally a tiny numeric baseline, not a competitive language model.

## Limitations

- A scalar regression target is not a proper categorical language-model head.
- No cross-entropy objective, sampling strategy, context positional encoding, attention training, or large-scale corpus pipeline exists yet.
- The simple MLP has limited sequence capacity and is only a stepping stone for model research.
