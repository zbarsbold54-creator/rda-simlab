"""Small data objects shared by the simulation core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    """Convert supported state values to deterministic JSON primitives."""
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": [int(v) for v in value.shape],
            "data": value.tobytes().hex(),
        }
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"unsupported state value for canonical serialization: {type(value).__name__}"
    )


def freeze(value: Any) -> Any:
    """Recursively freeze containers and copy arrays into read-only storage."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Return mutable containers and independent arrays for restoration."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class Action:
    turn: float = 0.0
    thrust: float = 0.0
    eat: bool = False
    action_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn", float(np.clip(self.turn, -1.0, 1.0)))
        object.__setattr__(self, "thrust", float(np.clip(self.thrust, 0.0, 1.0)))

    @property
    def move_direction(self) -> tuple[int, int]:
        return (
            round(self.thrust * np.cos(self.turn)),
            round(self.thrust * np.sin(self.turn)),
        )

    @property
    def energy_spent(self) -> float:
        return self.thrust


@dataclass(frozen=True)
class AgentState:
    agent_id: str
    position: tuple[float, float]
    heading: float = 0.0
    speed: float = 0.0
    energy: float = 100.0
    max_energy: float = 100.0
    health: float = 100.0
    age: float = 0.0
    alive: bool = True
    food_eaten: int = 0

    @property
    def is_alive(self) -> bool:
        return self.alive


@dataclass(frozen=True)
class Observation:
    rays: tuple[float, ...] = ()
    ray_types: tuple[int, ...] = ()
    proprioception: tuple[float, float, float] = (1.0, 1.0, 0.0)
    agent_position: tuple[float, float] = (0.0, 0.0)
    agent_energy: float = 100.0
    step_count: int = 0

    @property
    def world_state(self) -> np.ndarray:
        return np.asarray(self.rays, dtype=np.float32)

    @property
    def food_locations(self) -> list[tuple[float, float]]:
        return []

    @property
    def hazard_locations(self) -> list[tuple[float, float]]:
        return []


@dataclass(frozen=True)
class Snapshot:
    """Complete world state, suitable for exact restoration."""

    step: int
    agent_states: Mapping[str, AgentState]
    food_patches: tuple[Mapping[str, Any], ...]
    hazard_patches: tuple[Mapping[str, Any], ...]
    rng_state: Mapping[str, Any]
    width: float
    height: float
    dt: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_states", freeze(self.agent_states))
        object.__setattr__(self, "food_patches", freeze(self.food_patches))
        object.__setattr__(self, "hazard_patches", freeze(self.hazard_patches))
        object.__setattr__(self, "rng_state", freeze(self.rng_state))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @property
    def world_state(self) -> np.ndarray:
        raster = np.zeros((int(self.height) + 1, int(self.width) + 1), dtype=np.int8)
        for state in self.agent_states.values():
            if state.alive:
                x, y = int(state.position[0]), int(state.position[1])
                if 0 <= x < raster.shape[1] and 0 <= y < raster.shape[0]:
                    raster[y, x] = 1
        raster.setflags(write=False)
        return raster

    def to_dict(self) -> dict[str, Any]:
        """Return every field needed to restore and hash this snapshot."""
        return {
            "step": self.step,
            "agent_states": {
                agent_id: {
                    "agent_id": state.agent_id,
                    "position": state.position,
                    "heading": state.heading,
                    "speed": state.speed,
                    "energy": state.energy,
                    "max_energy": state.max_energy,
                    "health": state.health,
                    "age": state.age,
                    "alive": state.alive,
                    "food_eaten": state.food_eaten,
                }
                for agent_id, state in self.agent_states.items()
            },
            "food_patches": [thaw(patch) for patch in self.food_patches],
            "hazard_patches": [thaw(patch) for patch in self.hazard_patches],
            "rng_state": thaw(self.rng_state),
            "width": self.width,
            "height": self.height,
            "dt": self.dt,
            "metadata": thaw(self.metadata),
        }

    def save(self, base_path: str) -> tuple[str, str]:
        """Write the snapshot as ``base_path.json`` and ``base_path.npz``."""
        import json
        from pathlib import Path

        base = Path(base_path)
        json_path, npz_path = base.with_suffix(".json"), base.with_suffix(".npz")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(
                canonicalize(self.to_dict()),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
        np.savez_compressed(npz_path, world_state=self.world_state)
        return str(json_path), str(npz_path)

    @classmethod
    def load(cls, base_path: str) -> Snapshot:
        import json
        from pathlib import Path

        base = Path(base_path)
        with base.with_suffix(".json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        with np.load(base.with_suffix(".npz")) as archive:
            if "world_state" not in archive:
                raise ValueError("snapshot npz is missing world_state")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        agents = {
            aid: AgentState(**{**state, "position": tuple(state["position"])})
            for aid, state in data["agent_states"].items()
        }
        food = tuple(
            MappingProxyType({**patch, "position": tuple(patch["position"])})
            for patch in data.get("food_patches", [])
        )
        hazards = tuple(
            MappingProxyType({**patch, "position": tuple(patch["position"])})
            for patch in data.get("hazard_patches", [])
        )
        return cls(
            int(data["step"]),
            MappingProxyType(agents),
            food,
            hazards,
            data["rng_state"],
            float(data["width"]),
            float(data["height"]),
            float(data["dt"]),
            MappingProxyType(dict(data.get("metadata", {}))),
        )


@dataclass(frozen=True)
class StepResult:
    agent_states: Mapping[str, AgentState]
    observations: Mapping[str, Observation]
    snapshot: Snapshot | None
    terminated: bool
    truncated: bool = False
    reward: float | Mapping[str, float] = 0.0
    info: Mapping[str, Any] = field(default_factory=dict)
