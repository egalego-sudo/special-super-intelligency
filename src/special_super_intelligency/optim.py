"""Small, validated AdamW optimizer for autodiff scalar parameters."""
from __future__ import annotations

import math
from collections.abc import Iterable

from .autodiff import Value


class AdamW:
    """Adam with decoupled weight decay and global gradient-norm clipping."""

    def __init__(self, *, learning_rate=1e-3, betas=(0.9, 0.999),
                 epsilon=1e-8, weight_decay=0.01, max_grad_norm=1.0):
        if (type(learning_rate) not in (int, float)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("learning_rate must be positive and finite.")
        if (not isinstance(betas, tuple) or len(betas) != 2
                or any(type(beta) not in (int, float) or not 0 <= beta < 1
                       for beta in betas)):
            raise ValueError("betas must contain two numbers in [0, 1).")
        for name, value in (("epsilon", epsilon), ("max_grad_norm", max_grad_norm)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if (type(weight_decay) not in (int, float)
                or not math.isfinite(weight_decay) or weight_decay < 0):
            raise ValueError("weight_decay must be finite and non-negative.")
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2 = map(float, betas)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.steps = 0
        self._first_moment = []
        self._second_moment = []

    def step(self, parameters: Iterable[Value]):
        """Update parameters in place; return (unclipped_norm, clip_scale)."""
        parameters = list(parameters)
        if not parameters or any(not isinstance(item, Value) for item in parameters):
            raise ValueError("parameters must be a non-empty iterable of Value objects.")
        gradients = [item.grad for item in parameters]
        if any(not math.isfinite(gradient) for gradient in gradients):
            raise ArithmeticError("Non-finite gradient detected.")
        norm = math.sqrt(sum(gradient * gradient for gradient in gradients))
        scale = min(1.0, self.max_grad_norm / max(norm, 1e-12))
        if not self._first_moment:
            self._first_moment = [0.0] * len(parameters)
            self._second_moment = [0.0] * len(parameters)
        if len(parameters) != len(self._first_moment):
            raise ValueError("Parameter count changed after optimizer initialization.")

        self.steps += 1
        correction1 = 1.0 - self.beta1 ** self.steps
        correction2 = 1.0 - self.beta2 ** self.steps
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            gradient *= scale
            self._first_moment[index] = (
                self.beta1 * self._first_moment[index] + (1 - self.beta1) * gradient
            )
            self._second_moment[index] = (
                self.beta2 * self._second_moment[index] + (1 - self.beta2) * gradient**2
            )
            first = self._first_moment[index] / correction1
            second = self._second_moment[index] / correction2
            updated = parameter.data * (1 - self.learning_rate * self.weight_decay)
            updated -= self.learning_rate * first / (math.sqrt(second) + self.epsilon)
            if not math.isfinite(updated):
                raise ArithmeticError("Non-finite AdamW parameter update detected.")
            parameter.data = updated
        return norm, scale
