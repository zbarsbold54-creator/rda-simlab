"""Bounded snapshots and recorded actions for scrubbing and replay."""

from __future__ import annotations

from collections import OrderedDict, deque

from simlab.core.types import Action, Snapshot


class WorkerHistory:
    def __init__(self, capacity: int, initial: Snapshot) -> None:
        self.capacity = capacity
        self.actions: OrderedDict[int, dict[str, Action]] = OrderedDict()
        self.snapshots: deque[Snapshot] = deque(maxlen=capacity)
        self.pending: list[Snapshot] = []
        self.recording = True
        self.replay_until = 0
        self.reset(initial)

    def reset(self, initial: Snapshot) -> None:
        self.actions.clear()
        self.snapshots.clear()
        self.pending.clear()
        self.replay_until = 0
        self.add_snapshot(initial)

    def add_snapshot(self, snapshot: Snapshot) -> None:
        self.snapshots.append(snapshot)
        self.pending.append(snapshot)

    def take_batch(self) -> tuple[Snapshot, ...]:
        batch = tuple(self.pending)
        self.pending.clear()
        return batch

    def action_for(self, step: int) -> dict[str, Action] | None:
        if step >= self.replay_until:
            return None
        return self.actions.get(step)

    def record(self, step: int, actions: dict[str, Action]) -> None:
        if self.recording:
            self.actions[step] = dict(actions)

    def resume_from(self, step: int) -> None:
        future = [action_step for action_step in self.actions if action_step >= step]
        self.replay_until = max(future, default=step - 1) + 1

    def replay_origin(self) -> Snapshot | None:
        if not self.actions or not self.snapshots:
            return None
        oldest_action = min(self.actions)
        return next(
            (snapshot for snapshot in self.snapshots if snapshot.step >= oldest_action),
            None,
        )

    def replay_to(self, final_step: int) -> None:
        self.replay_until = final_step
