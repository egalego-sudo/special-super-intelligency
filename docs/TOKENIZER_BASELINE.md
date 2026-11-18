# Tokenizer Baseline

**Creator by Bittu Pandit**

## Current implementation

`ByteTokenizer` maps UTF-8 bytes directly to integer IDs 0 through 255. It is deterministic and lossless for valid Unicode strings.

## Why this baseline exists

- Establishes a fully inspectable tokenizer with no external model dependency.
- Provides a stable interface for early data-pipeline and model experiments.
- Handles Hindi, English, emoji, and other Unicode scripts through UTF-8.

## Known limitations

- It is not a learned tokenizer (BPE, unigram, or WordPiece).
- It may use many more tokens than a trained subword tokenizer.
- It has no special tokens, vocabulary training, normalization policy, or streaming decode.
- It does not imply that a language model exists or has been trained.

## Next research steps

1. Define normalization and special-token policy with reversible tests.
2. Build a corpus audit tool with licensing and provenance metadata.
3. Compare byte-level and subword tokenization on a legally usable multilingual corpus.
4. Publish compression, unknown-text, and round-trip measurements before selecting a production tokenizer.
