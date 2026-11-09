"""The sole source of pseudo-randomness used by the core."""

from __future__ import annotations

from typing import Any

import numpy as np


class SimRNG:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    @property
    def generator(self) -> np.random.Generator:
        return self._rng

    @property
    def state(self) -> dict[str, Any]:
        return self._rng.bit_generator.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._rng.bit_generator.state = value

    def random(self) -> float:
        return float(self._rng.random())

    def uniform(
        self, low: float, high: float, size: int | tuple[int, ...] | None = None
    ) -> Any:
        return self._rng.uniform(low, high, size)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> Any:
        return self._rng.integers(low, high, size)
