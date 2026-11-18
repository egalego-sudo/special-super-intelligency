# Reproducible Experiments and Evaluation

**Creator by Bittu Pandit**

Experiments use an explicit immutable configuration containing context length, hidden width, epoch count, learning rate, and random seed. The current runner trains the byte-level next-token regression baseline.

The evaluation helper reports sample count, mean squared error, and count within a declared tolerance. It does not alter model weights.

These are engineering baselines only. They are not benchmark results for general reasoning, coding, factuality, or GPT-class performance.
