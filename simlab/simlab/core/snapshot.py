"""Snapshot storage and JSON friendly action log records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simlab.core.types import Action, Snapshot


@dataclass(frozen=True)
class ActionLogEntry:
    step: int
    agent_id: str
    action: Action

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "action": vars(self.action),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionLogEntry:
        return cls(
            int(data["step"]),
            str(data["agent_id"]),
            Action(**data["action"]),
        )


@dataclass(frozen=True)
class ActionLog:
    """Serializable seed and per-agent action records for deterministic replay."""

    seed: int
    entries: tuple[ActionLogEntry, ...]

    @classmethod
    def from_steps(cls, seed: int, action_steps: list[dict[str, Action]]) -> ActionLog:
        return cls(
            int(seed),
            tuple(
                ActionLogEntry(step, agent_id, action)
                for step, actions in enumerate(action_steps)
                for agent_id, action in sorted(actions.items())
            ),
        )

    def actions_by_step(self) -> dict[int, dict[str, Action]]:
        actions: dict[int, dict[str, Action]] = {}
        for entry in self.entries:
            actions.setdefault(entry.step, {})[entry.agent_id] = entry.action
        return actions

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seed": self.seed,
                    "entries": [entry.to_dict() for entry in self.entries],
                },
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
        return str(destination)

    @classmethod
    def load(cls, path: str | Path) -> ActionLog:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            int(data["seed"]),
            tuple(ActionLogEntry.from_dict(entry) for entry in data["entries"]),
        )


class SnapshotManager:
    def __init__(self, max_snapshots: int = 2000) -> None:
        self._snapshots: dict[int, Snapshot] = {}
        self.max_snapshots = max_snapshots

    def add(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.step] = snapshot
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(min(self._snapshots))

    @property
    def latest_snapshot(self) -> Snapshot | None:
        return self._snapshots[max(self._snapshots)] if self._snapshots else None

    def get_snapshot(self, step: int) -> Snapshot | None:
        return self._snapshots.get(step)

    def capture_snapshot(self, world: Any) -> Snapshot:
        snapshot = world.snapshot()
        self.add(snapshot)
        return snapshot
