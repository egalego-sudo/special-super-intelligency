# Independent Neural Core — First Baseline

**Creator by Bittu Pandit**

The repository now includes `TinyMLP`, a small, dependency-free, one-hidden-layer neural network. It has deterministic initialization, explicit trainable weights, forward inference, and batch-gradient training with mean squared error.

## What this proves

- The project can initialize and update its own model parameters.
- Inference is performed by repository-owned code.
- A small supervised learning task can be used as an executable training check.

## What it does not prove

- It is not a Transformer, language model, multimodal model, or GPT-class system.
- It has no tokenizer-to-model sequence pipeline, attention, large-scale data, or distributed training.
- Passing a toy learning test is not evidence of broad reasoning or general intelligence.

## Next steps

1. Add numerical gradient checks against finite differences.
2. Add checkpoint serialization with strict shape and version validation.
3. Add a tokenizer-to-sequence dataset interface.
4. Research and implement a small causal self-attention block.
5. Add reproducible benchmark tasks before making quality claims.
