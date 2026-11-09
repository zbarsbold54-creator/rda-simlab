"""Drive-core and retention interfaces."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class RetentionRule(Protocol):
    def step(
        self, previous: np.ndarray, drives: np.ndarray, context: dict[str, Any]
    ) -> np.ndarray:
        """Retain or update a batch of drive values."""


class DriveCore(Protocol):
    names: tuple[str, ...]
    needs_observations: bool

    def reset(self, n_agents: int, rng: Any) -> None: ...

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray: ...


class FixedDecay:
    """Default retention rule: E = lambda * previous + (1-lambda) * drives."""

    def __init__(self, lambdas: float | list[float] = 0.9) -> None:
        self.lambdas = np.asarray(lambdas, dtype=np.float32)

    def step(
        self, previous: np.ndarray, drives: np.ndarray, context: dict[str, Any]
    ) -> np.ndarray:
        decay = self.lambdas
        if decay.ndim == 0:
            decay = np.full(drives.shape[-1], float(decay), dtype=np.float32)
        return np.clip(decay * previous + (1.0 - decay) * drives, 0.0, 1.0)

    def should_retain(
        self, agent_energy: float, step: int, context: dict[str, Any]
    ) -> bool:
        return agent_energy > 0.0


class BasicDriveCore:
    """Placeholder drive core, to be replaced by the real drive core."""

    names = (
        "novelty",
        "uncertainty",
        "threat",
        "integrity",
        "agency",
        "learning_progress",
        "arousal",
        "homeostatic_deviation",
    )
    needs_observations = True

    def __init__(self, retention: RetentionRule | None = None) -> None:
        self.retention = retention or FixedDecay()
        self._previous = np.zeros((0, len(self.names)), dtype=np.float32)
        self._last_rays: list[np.ndarray] = []

    def reset(self, n_agents: int, rng: Any) -> None:
        self._previous = np.zeros((n_agents, len(self.names)), dtype=np.float32)
        self._last_rays = [np.zeros(0, dtype=np.float32) for _ in range(n_agents)]

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray:
        if self._previous.shape[0] != len(obs_batch):
            self.reset(len(obs_batch), info.get("rng"))
        values = np.zeros_like(self._previous)
        for index, obs in enumerate(obs_batch):
            rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
            energy, health, speed = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
            hazard_dist = min(
                (
                    float(ray)
                    for ray, kind in zip(rays, getattr(obs, "ray_types", ()))
                    if kind == 2
                ),
                default=20.0,
            )
            values[index] = (
                0.0,
                float(np.clip(np.var(rays) if rays.size else 0.0, 0.0, 1.0)),
                float(np.clip(1.0 - hazard_dist / 20.0, 0.0, 1.0)),
                float(np.clip(1.0 - health, 0.0, 1.0)),
                float(np.clip(speed, 0.0, 1.0)),
                0.0,
                0.0,
                float(np.clip(1.0 - energy, 0.0, 1.0)),
            )
            if index < len(self._last_rays) and self._last_rays[index].size:
                values[index, 5] = float(
                    np.clip(np.mean(np.abs(rays - self._last_rays[index])), 0.0, 1.0)
                )
            if index < len(self._last_rays):
                self._last_rays[index] = rays.copy()
        updated = self.retention.step(self._previous, values, info)
        self._previous = updated.copy()
        return updated

    def compute_drives(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        retention_rule: RetentionRule | None = None,
    ) -> dict[str, float]:
        energy = float(agent_state.get("energy", 100.0)) / 100.0
        position = agent_state.get("position", (0.0, 0.0))
        hazards = world_state.get("hazard_locations", [])
        nearest = min(
            (np.hypot(h[0] - position[0], h[1] - position[1]) for h in hazards),
            default=20.0,
        )
        values = np.array(
            [
                [
                    0.0,
                    0.0,
                    np.clip(1.0 - nearest / 20.0, 0.0, 1.0),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 - energy,
                ]
            ],
            dtype=np.float32,
        )
        if retention_rule is not None and hasattr(retention_rule, "step"):
            values = retention_rule.step(np.zeros_like(values), values, {})
        return dict(zip(self.names, values[0]))

    def get_state(self) -> dict[str, Any]:
        """Return internal state so snapshots can capture the drive core."""
        return {
            "previous": self._previous.copy(),
            "last_rays": [r.copy() for r in self._last_rays],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a captured snapshot."""
        self._previous = np.asarray(state["previous"], dtype=np.float32)
        self._last_rays = [np.asarray(r, dtype=np.float32) for r in state["last_rays"]]
