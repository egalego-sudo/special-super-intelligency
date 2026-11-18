"""Minimal reverse-mode scalar automatic differentiation engine.

This research primitive is intentionally small and dependency-free. It provides
correct scalar gradients for arithmetic and common smooth activations; it is
not yet wired into DecoderLanguageModel parameters.
"""
from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """A scalar in a computation graph with reverse-mode differentiation."""

    def __init__(self, data: float, _children=(), _op: str = ""):
        if type(data) not in (int, float) or not math.isfinite(data):
            raise ValueError("Value data must be a finite number.")
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (self._coerce(other) ** -1)

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def __pow__(self, power):
        if type(power) not in (int, float) or not math.isfinite(power):
            raise ValueError("power must be a finite scalar.")
        out = Value(self.data ** power, (self,), f"**{power}")
        def backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = backward
        return out

    def exp(self):
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")
        def backward():
            self.grad += result * out.grad
        out._backward = backward
        return out

    def tanh(self):
        result = math.tanh(self.data)
        out = Value(result, (self,), "tanh")
        def backward():
            self.grad += (1.0 - result * result) * out.grad
        out._backward = backward
        return out

    def log(self):
        if self.data <= 0:
            raise ValueError("log requires a positive value.")
        out = Value(math.log(self.data), (self,), "log")
        def backward():
            self.grad += out.grad / self.data
        out._backward = backward
        return out

    def __lt__(self, other):
        other = self._coerce(other)
        return self.data < other.data

    def backward(self):
        """Accumulate gradients from this scalar through its full graph."""
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)
        # Intermediate gradients belong to this traversal; clear them before
        # recomputation while preserving accumulated gradients on leaf values.
        for node in topo:
            if node._prev:
                node.grad = 0.0
        self.grad += 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset this value and all reachable graph gradients."""
        visited = set()
        def clear(node):
            if node not in visited:
                visited.add(node)
                node.grad = 0.0
                for parent in node._prev:
                    clear(parent)
        clear(self)

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
