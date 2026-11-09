"""Reference template for a learned Policy plugin.

This is a numpy-only demonstration of the Policy extension contract:

- it declares ``needs_observations = True`` (it reads obs and drive_state),
- it keeps internal recurrent state (a small GRU-like cell with fixed random
  weights) that a policy with state must step every step,
- it supports ``reset(n_agents, rng)``, ``get_state()`` and ``set_state()``
  so snapshots can capture and restore its internal state (scrub-back replay),
- it never imports torch and only depends on numpy.

Alignment note: ``act`` is called once per live agent per step, in stable
iteration order, without an agent id. For simplicity the hidden state is
indexed by call order modulo the agent count, which is deterministic for a
fixed live set (the purpose of this template). A real learned policy should
index hidden state by a stable per-agent identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from simlab.core.registry import register
from simlab.core.types import Action

_HIDDEN = 16
_DRIVE_IN = 8
_THREAT_FULL = 20.0


def _clamp(x: float) -> float:
    return float(max(-1.0, min(1.0, x)))


class ExampleLearnedPolicy:
    needs_observations = True

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        rng = np.random.default_rng(0)
        self._in_dim = 2 + 1 + _DRIVE_IN
        self._wi = rng.normal(0.0, 0.2, (self._in_dim, _HIDDEN)).astype(np.float32)
        self._wh = rng.normal(0.0, 0.2, (_HIDDEN, _HIDDEN)).astype(np.float32)
        self._bi = rng.normal(0.0, 0.1, (_HIDDEN,)).astype(np.float32)
        self._wo = rng.normal(0.0, 0.2, (_HIDDEN, 3)).astype(np.float32)
        self._bo = rng.normal(0.0, 0.1, (3,)).astype(np.float32)
        self.n_agents = 0
        self.hidden = np.zeros((0, _HIDDEN), dtype=np.float32)
        self._call_index = 0

    def reset(self, n_agents: int, rng: Any) -> None:
        self.n_agents = int(n_agents)
        self.hidden = np.zeros((self.n_agents, _HIDDEN), dtype=np.float32)
        self._call_index = 0

    def _features(self, obs: Any, drive_state: Any) -> np.ndarray:
        rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
        ray_types = getattr(obs, "ray_types", ())
        food_dists = [
            float(d)
            for d, k in zip(rays, ray_types)
            if k == 1
        ]
        hazard_dists = [
            float(d)
            for d, k in zip(rays, ray_types)
            if k == 2
        ]
        best_food = min(food_dists) if food_dists else _THREAT_FULL
        threat = min(hazard_dists) if hazard_dists else _THREAT_FULL
        mean_ray = float(np.mean(rays)) if rays.size else 0.0
        drive = np.asarray(drive_state, dtype=np.float32)
        if drive.size < _DRIVE_IN:
            drive = np.concatenate(
                [drive, np.zeros(_DRIVE_IN - drive.size, dtype=np.float32)]
            )
        return np.concatenate(
            [
                np.asarray(
                    [
                        mean_ray,
                        1.0 / (1.0 + best_food),
                        1.0 / (1.0 + threat),
                    ],
                    dtype=np.float32,
                ),
                drive[:_DRIVE_IN],
            ]
        )

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if self.n_agents == 0:
            self.reset(1, np.random.default_rng(0))
        idx = self._call_index % self.n_agents
        x = self._features(obs, drive_state)
        h_prev = self.hidden[idx]
        h_new = np.tanh(
            x @ self._wi + h_prev @ self._wh + self._bi
        )
        logits = h_new @ self._wo + self._bo
        turn = _clamp(float(logits[0]))
        thrust = _clamp(float(logits[1]) * 0.5 + 0.5)
        eat = bool(logits[2] > 0.0)
        self.hidden[idx] = h_new
        self._call_index += 1
        return Action(turn, thrust, eat, "example_learned")

    def get_state(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden.copy(),
            "n_agents": self.n_agents,
            "call_index": self._call_index,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.hidden = np.asarray(state["hidden"], dtype=np.float32)
        self.n_agents = int(state["n_agents"])
        self._call_index = int(state["call_index"])


register("policy", "example_learned")(ExampleLearnedPolicy)
