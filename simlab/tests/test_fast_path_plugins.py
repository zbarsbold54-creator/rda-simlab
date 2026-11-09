"""Fast-path correctness with stateful registered plugins.

Checks that the GUI Max path and the headless Runner step stateful
components (a recurrent DriveCore using the FixedDecay retention rule and a
Policy that reads obs + drive_state) identically, and that skipping a drive
step breaks determinism.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import numpy as np

from simlab.core.config import load_config
from simlab.core.pipeline import compute_actions
from simlab.core.registry import get, register
from simlab.core.runner import Runner
from simlab.core.types import Action
from simlab.gui.frame import AgentFrame
from simlab.gui.sim_worker import SimWorker
from simlab.gui.worker_components import WorkerComponents
from simlab.interfaces.drive_core import FixedDecay


class RecurrentTestDriveCore:
    """Recurrent drive core: fixed-decay retention of observation-derived drives."""

    names = ("novelty", "threat")
    needs_observations = True

    def __init__(self) -> None:
        self.retention = FixedDecay([0.9, 0.5])
        self._previous = np.zeros((0, 2), dtype=np.float32)
        self.history: list[np.ndarray] = []

    def reset(self, n_agents: int, rng) -> None:
        self._previous = np.zeros((n_agents, 2), dtype=np.float32)
        self.history = []

    def step(self, obs_batch: list, info: dict) -> np.ndarray:
        values = np.zeros((len(obs_batch), 2), dtype=np.float32)
        for index, obs in enumerate(obs_batch):
            rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
            values[index, 0] = float(np.mean(rays)) if rays.size else 0.0
            hazard_dist = min(
                (
                    float(ray)
                    for ray, kind in zip(rays, getattr(obs, "ray_types", ()))
                    if kind == 2
                ),
                default=20.0,
            )
            values[index, 1] = float(np.clip(1.0 - hazard_dist / 20.0, 0.0, 1.0))
        updated = self.retention.step(self._previous, values, info)
        self._previous = updated.copy()
        self.history.append(updated.copy())
        return updated


register("drive", "test_recurrent")(RecurrentTestDriveCore)


class StatefulTestPolicy:
    """Policy that reads obs + drive_state and keeps internal state."""

    needs_observations = True

    def __init__(self, seed: int = 0) -> None:
        self._calls = 0
        self.history: list[tuple[int, float]] = []

    def act(self, obs, drive_state=None) -> Action:
        self._calls += 1
        drive_total = float(sum(drive_state)) if drive_state else 0.0
        self.history.append((self._calls, drive_total))
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())
        best_dist = None
        best_index = None
        for index, (distance, kind) in enumerate(zip(rays, ray_types)):
            if kind == 1 and (best_dist is None or distance < best_dist):
                best_dist = distance
                best_index = index
        if best_dist is None:
            return Action(0.25, 0.5, False)
        count = len(rays)
        middle = count // 2
        offset_deg = (
            (best_index - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
        )
        turn = max(-1.0, min(1.0, offset_deg / 135.0))
        thrust = 0.8 * (0.5 + 0.5 * float(np.clip(drive_total, 0.0, 1.0)))
        return Action(turn, thrust, best_dist < 1.5)


register("policy", "test_stateful_policy")(StatefulTestPolicy)


def _plugin_config():
    return replace(load_config(), policy="test_stateful_policy", drive="test_recurrent")


def test_stateful_plugins_fast_path_matches_headless() -> None:
    config = _plugin_config()
    seed = 91021
    assert get("drive", "test_recurrent") is not None
    assert get("policy", "test_stateful_policy") is not None

    headless = Runner(config, seed=seed, agents=20)
    headless.setup()
    worker = SimWorker(config, seed=seed, agents=20)
    worker.set_speed("Max")
    assert worker._speed == "Max"

    hashes = []
    for step in range(1000):
        advanced = worker._advance_one_step()
        assert advanced is not None
        headless.run_step()
        wh = worker._runner._world.state_hash()
        hh = headless._world.state_hash()
        hashes.append((wh, hh))
        assert wh == hh, f"hash divergence at step {step}"

    worker_drive = worker._components.drive_core.history
    headless_drive = headless._drive_core.history
    assert len(worker_drive) == len(headless_drive) == 1000
    for index, (wd, hd) in enumerate(zip(worker_drive, headless_drive)):
        assert np.array_equal(wd, hd), f"drive divergence at step {index}"

    worker_policy = worker._components.policy.history
    headless_policy = headless._policy.history
    assert worker_policy == headless_policy


def test_skipping_drive_step_breaks_determinism() -> None:
    config = _plugin_config()
    seed = 91021

    reference = Runner(config, seed=seed, agents=20)
    reference.setup()
    reference_hashes = []
    for _ in range(200):
        reference.run_step()
        reference_hashes.append(reference._world.state_hash())

    original_actions_for = WorkerComponents.actions_for
    calls = {"n": 0}

    def broken_actions_for(self, world, step):
        # Simulate a buggy fast path that skips the drive update every other step.
        if calls["n"] % 2 == 1:
            actions, drives, observations = compute_actions(
                world, self.policy, None, step
            )
        else:
            actions, drives, observations = compute_actions(
                world, self.policy, self.drive_core, step
            )
        calls["n"] += 1
        details = MappingProxyType(
            {
                agent_id: AgentFrame(
                    actions.get(agent_id),
                    observations[agent_id],
                    drives.get(agent_id, ()),
                )
                for agent_id in observations
            }
        )
        return actions, details

    worker = SimWorker(config, seed=seed, agents=20)
    worker.set_speed("Max")
    try:
        WorkerComponents.actions_for = broken_actions_for
        broken_hashes = []
        for _ in range(200):
            advanced = worker._advance_one_step()
            assert advanced is not None
            broken_hashes.append(worker._runner._world.state_hash())
    finally:
        WorkerComponents.actions_for = original_actions_for

    diverged = any(b != r for b, r in zip(broken_hashes, reference_hashes))
    assert diverged, "expected skipped drive step to diverge world state"
