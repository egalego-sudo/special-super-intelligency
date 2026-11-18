# Causal Attention Research Block

**Creator by Bittu Pandit**

The first attention primitive implements single-head scaled dot-product causal self-attention with explicit query, key, and value projection matrices. At position *t*, its softmax includes only positions up to and including *t*.

## Verification target

The tests check output shape, invalid input rejection, and causality: changing a future token must not alter any earlier output.

## Limitations

This is a research primitive, not a complete Transformer. It does not yet include learned embeddings, multi-head attention, positional encoding, residual connections, normalization, feed-forward blocks, batching, efficient kernels, or a training/autograd system.
