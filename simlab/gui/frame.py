"""Immutable GUI data published by the simulation worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from simlab.core.types import Action, Observation, Snapshot


@dataclass(frozen=True)
class AgentFrame:
    action: Action | None
    observation: Observation | None
    drives: tuple[float, ...]
    retention: dict[str, float] | None = None


@dataclass(frozen=True)
class SimulationFrame:
    snapshot: Snapshot
    agents: Mapping[str, AgentFrame]
    drive_names: tuple[str, ...]
    # agent_id -> (death_step, cause)
    deaths: Mapping[str, tuple[int, str]] = field(default_factory=dict)
    # agent_id -> ((step, energy), ...) bounded series for the selected agent plot
    energies: Mapping[str, tuple[tuple[int, float], ...]] = field(default_factory=dict)
