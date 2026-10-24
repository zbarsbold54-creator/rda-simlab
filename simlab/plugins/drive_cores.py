"""Numpy-only drive cores for architecture-agnostic learners.

These implement the world-facing ``DriveCore`` port (``step(obs_batch, info)``
returns an ``(n_live, 8)`` array) and expose ``get_state()/set_state()`` so a
learned policy's snapshot/replay can capture their recurrence. Torch is never
imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simlab.core.registry import register
from simlab.drives.definitions import DRIVE_NAMES, N_DRIVES, raw_drive_vector
from simlab.training.arch_spec import add_spec


class NoneDriveCore:
    """Zero drive vector; no observation dependence."""

    names = DRIVE_NAMES
    needs_observations = False

    def __init__(self, decay: Any = None) -> None:
        self._previous = np.zeros((0, N_DRIVES), dtype=np.float32)

    def reset(self, n_agents: int, rng: Any) -> None:
        self._previous = np.zeros((n_agents, N_DRIVES), dtype=np.float32)

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray:
        return np.zeros((len(obs_batch), N_DRIVES), dtype=np.float32)

    def get_state(self) -> dict[str, Any]:
        return {"previous": self._previous.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._previous = np.asarray(state["previous"], dtype=np.float32)


class ConstantDriveCore:
    """Constant drive vector, configurable via ``value`` (scalar or length-8)."""

    names = DRIVE_NAMES
    needs_observations = False

    def __init__(self, value: Any = 0.5) -> None:
        v = np.asarray(value, dtype=np.float32)
        if v.ndim == 0:
            v = np.full(N_DRIVES, float(v), dtype=np.float32)
        self._value = np.clip(v[:N_DRIVES], 0.0, 1.0).astype(np.float32)

    def reset(self, n_agents: int, rng: Any) -> None:
        return None

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray:
        return np.tile(self._value[None, :], (len(obs_batch), 1))

    def get_state(self) -> dict[str, Any]:
        return {"value": self._value.copy()}

    def set_state(self, state: dict[str, Any]) -> None:
        self._value = np.asarray(state["value"], dtype=np.float32)


class RDA8DriveCore:
    """Recurrent drive array: E[t] = decay*E[t-1] + (1-decay)*drive[t].

    ``decay`` may be a scalar or a length-8 per-drive vector. The raw drive at
    step ``t`` depends only on information up to step ``t`` (causal).
    """

    names = DRIVE_NAMES
    needs_observations = True

    def __init__(self, decay: Any = 0.9) -> None:
        d = np.asarray(decay, dtype=np.float32)
        self.decay = (
            np.full(N_DRIVES, float(d), dtype=np.float32)
            if d.ndim == 0
            else d[:N_DRIVES]
        )
        self._previous = np.zeros((0, N_DRIVES), dtype=np.float32)
        self._last_rays: list[np.ndarray] = []

    def reset(self, n_agents: int, rng: Any) -> None:
        self._previous = np.zeros((n_agents, N_DRIVES), dtype=np.float32)
        self._last_rays = [np.zeros(0, dtype=np.float32) for _ in range(n_agents)]

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray:
        if self._previous.shape[0] != len(obs_batch):
            self.reset(len(obs_batch), info.get("rng"))
        raw = np.zeros((len(obs_batch), N_DRIVES), dtype=np.float32)
        for i, obs in enumerate(obs_batch):
            raw[i] = raw_drive_vector(obs, self._last_rays[i])
            self._last_rays[i] = np.asarray(
                getattr(obs, "rays", ()), dtype=np.float32
            ).copy()
        decay = self.decay[None, :]
        updated = np.clip(decay * self._previous + (1.0 - decay) * raw, 0.0, 1.0)
        self._previous = updated.copy()
        return updated

    def get_state(self) -> dict[str, Any]:
        return {
            "previous": self._previous.copy(),
            "last_rays": [r.copy() for r in self._last_rays],
            "decay": self.decay.copy(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._previous = np.asarray(state["previous"], dtype=np.float32)
        self._last_rays = [np.asarray(r, dtype=np.float32) for r in state["last_rays"]]
        self.decay = np.asarray(state["decay"], dtype=np.float32)


class InstantDriveCore:
    """Instantaneous drive array: ``d_t = raw_drive_vector(obs)``.

    Unlike ``RDA8DriveCore`` this applies no EMA; it returns the raw
    instantaneous 8-drive signal at step ``t``. The learner-side Retention
    consumes this signal.
    """

    names = DRIVE_NAMES
    needs_observations = True

    def __init__(self, decay: Any = None) -> None:
        self._last_rays: list[np.ndarray] = []

    def reset(self, n_agents: int, rng: Any) -> None:
        self._last_rays = [np.zeros(0, dtype=np.float32) for _ in range(n_agents)]

    def step(self, obs_batch: list[Any], info: dict[str, Any]) -> np.ndarray:
        if len(self._last_rays) != len(obs_batch):
            self.reset(len(obs_batch), info.get("rng"))
        raw = np.zeros((len(obs_batch), N_DRIVES), dtype=np.float32)
        for i, obs in enumerate(obs_batch):
            raw[i] = raw_drive_vector(obs, self._last_rays[i])
            self._last_rays[i] = np.asarray(
                getattr(obs, "rays", ()), dtype=np.float32
            ).copy()
        return raw

    def get_state(self) -> dict[str, Any]:
        return {"last_rays": [r.copy() for r in self._last_rays]}

    def set_state(self, state: dict[str, Any]) -> None:
        self._last_rays = [np.asarray(r, dtype=np.float32) for r in state["last_rays"]]


register("drive", "none")(NoneDriveCore)
register("drive", "constant")(ConstantDriveCore)
register("drive", "rda8")(RDA8DriveCore)
register("drive", "instant")(InstantDriveCore)


@dataclass
class ConstantConfig:
    value: float = field(
        default=0.5,
        metadata={"min": 0.0, "max": 1.0, "help": "Constant drive value (0..1)."},
    )


@dataclass
class RDA8Config:
    decay: float = field(
        default=0.9,
        metadata={"min": 0.0, "max": 1.0, "help": "EMA decay for the drive array."},
    )


add_spec(
    "drive_core",
    "none",
    description="Zero drive vector; no observation dependence.",
    config=None,
    loader=lambda: NoneDriveCore,
    source=__name__,
)
add_spec(
    "drive_core",
    "constant",
    description="Constant drive vector, configurable via value.",
    config=ConstantConfig,
    loader=lambda: ConstantDriveCore,
    source=__name__,
)
add_spec(
    "drive_core",
    "rda8",
    description="Recurrent drive array E[t]=decay*E[t-1]+(1-decay)*drive[t].",
    config=RDA8Config,
    loader=lambda: RDA8DriveCore,
    source=__name__,
)
add_spec(
    "drive_core",
    "instant",
    description="Instantaneous drive array d_t (no EMA); consumed by the learner Retention.",
    config=None,
    loader=lambda: InstantDriveCore,
    source=__name__,
)
