"""Headless simulation runner."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

from simlab.core.config import SimLabConfig, load_config
from simlab.core.pipeline import compute_actions
from simlab.core.registry import get, load_plugins
from simlab.core.snapshot import ActionLog, SnapshotManager
from simlab.core.types import Action, AgentState, Snapshot, StepResult
from simlab.core.world import World


class Runner:
    def __init__(
        self,
        config: SimLabConfig | None = None,
        seed: int | None = None,
        rng: Any = None,
        agents: int = 1,
    ) -> None:
        self._config = config or load_config()
        self._seed = self._config.world.seed if seed is None else seed
        self._agent_count = max(1, int(agents))
        self._world = World(
            self._config.world.width,
            self._config.world.height,
            self._seed,
            food_regrow_rate=self._effective_regrow_rate(),
            eat_dist=self._config.food.eat_dist,
            max_turn_rate=self._config.agent.max_turn_rate,
            speed_factor=self._config.agent.speed_factor,
        )
        self._step_count = 0
        self._max_steps = 100000
        self._is_running = False
        self._snapshot_manager = SnapshotManager()
        self._snapshots: dict[int, Snapshot] = self._snapshot_manager._snapshots
        self.action_log: list[dict[str, Action]] = []
        self._policy = None
        self._drive_core = None
        self._components_loaded = False

    def _effective_regrow_rate(self) -> float:
        """Resolve the world regrow rate from config.

        If ``food.supply_ratio`` is set, regrowth is scaled so that a given
        per-agent supply (in units per step per agent) is maintained as the
        agent count changes: ``rate = supply_ratio * n_agents * 0.0336 /
        n_patches``. Otherwise the raw ``food.regrow_rate`` is used.
        """
        supply = self._config.food.supply_ratio
        if supply is None:
            return float(self._config.food.regrow_rate)
        n_patches = min(
            self._config.food.max_food,
            max(
                1,
                int(
                    self._config.world.width
                    * self._config.world.height
                    * self._config.food.spawn_rate
                    / 10
                ),
            ),
        )
        return float(supply) * self._agent_count * 0.0336 / n_patches

    def setup(self) -> None:
        self._world.reset(self._seed)
        self._step_count = 0
        self._snapshots.clear()
        self.action_log.clear()
        for index in range(self._agent_count):
            x = float(self._world._rng.uniform(10.0, self._world.width - 10.0))
            y = float(self._world._rng.uniform(10.0, self._world.height - 10.0))
            heading = float(self._world._rng.uniform(-3.14, 3.14))
            self._world.add_agent(
                f"agent_{index}",
                (x, y),
                self._config.agent.initial_energy,
                heading=heading,
            )
        count = min(
            self._config.food.max_food,
            max(
                1,
                int(
                    self._world.width
                    * self._world.height
                    * self._config.food.spawn_rate
                    / 10
                ),
            ),
        )
        for _ in range(count):
            self._world.add_food(
                (
                    float(self._world._rng.uniform(2, self._world.width - 2)),
                    float(self._world._rng.uniform(2, self._world.height - 2)),
                )
            )
        for _ in range(self._config.hazard.count):
            self._world.add_hazard(
                (
                    float(self._world._rng.uniform(2, self._world.width - 2)),
                    float(self._world._rng.uniform(2, self._world.height - 2)),
                ),
                damage=self._config.hazard.damage_per_step,
                detection_radius=self._config.hazard.detection_radius,
            )
        self._is_running = True
        self._snapshot_manager.add(self._world.snapshot())

    def _default_action(self, aid: str) -> Action:
        state = self._world.get_agent(aid)
        if state is None or not state["is_alive"]:
            return Action()
        patches = [p for p in self._world._food_patches if p["amount"] > 0.0]
        if patches:
            target = min(
                patches,
                key=lambda patch: (
                    (patch["position"][0] - state["position"][0]) ** 2
                    + (patch["position"][1] - state["position"][1]) ** 2
                ),
            )
            desired = math.atan2(
                target["position"][1] - state["position"][1],
                target["position"][0] - state["position"][0],
            )
            delta = (desired - state["heading"] + math.pi) % (2 * math.pi) - math.pi
            return Action(float(max(-1.0, min(1.0, delta))), 0.8, True)
        return Action(0.25, 0.5, False)

    def _ensure_components(self) -> bool:
        """Lazily load the registered policy and drive core from the config."""
        if self._components_loaded:
            return self._policy is not None or self._drive_core is not None
        self._components_loaded = True
        load_plugins()
        policy_type = get("policy", self._config.policy)
        if policy_type is not None:
            try:
                self._policy = policy_type(seed=self._seed, config=self._config)
            except TypeError:
                try:
                    self._policy = policy_type(seed=self._seed)
                except TypeError:
                    self._policy = policy_type()
        drive_type = get("drive", self._config.drive)
        if drive_type is not None:
            self._drive_core = drive_type()
            self._drive_core.reset(self._agent_count, self._world._rng)
        return self._policy is not None or self._drive_core is not None

    def _component_actions(self) -> dict[str, Action]:
        self._ensure_components()
        actions, _, _ = compute_actions(
            self._world, self._policy, self._drive_core, self._step_count
        )
        return actions

    def run_step(
        self,
        agent_actions: dict[str, Action] | None = None,
        *,
        include_observations: bool = True,
        include_snapshot: bool = True,
        include_agent_states: bool = True,
    ) -> StepResult:
        if not self._is_running:
            raise RuntimeError("Runner is not running. Call setup() first.")
        if agent_actions is not None:
            actions = dict(agent_actions)
        elif self._ensure_components():
            actions = self._component_actions()
        else:
            actions = {
                aid: self._default_action(aid) for aid in self._world._agents
            }
        action_index = self._step_count
        result = self._world.step(
            actions,
            include_observations=include_observations,
            include_snapshot=include_snapshot,
            include_agent_states=include_agent_states,
        )
        self._step_count = self._world.get_step_count()
        if action_index < len(self.action_log):
            self.action_log[action_index] = actions
        else:
            self.action_log.append(actions)
        if result.snapshot is not None:
            self._snapshot_manager.add(result.snapshot)
        return result

    def run_episode(
        self,
        agent_actions_generator: Callable[
            [int, dict[str, AgentState]], dict[str, Action]
        ]
        | None = None,
    ) -> list[StepResult]:
        results = []
        while self._is_running and self._step_count < self._max_steps:
            actions = (
                agent_actions_generator(
                    self._step_count, dict(self._world._state_view())
                )
                if agent_actions_generator
                else None
            )
            result = self.run_step(actions)
            results.append(result)
            if result.terminated:
                break
        return results

    def run_bench(self, num_steps: int) -> list[StepResult]:
        """Benchmark both world-only and full-pipeline step throughput."""
        self.run_bench_world_only(num_steps)
        return self.run_bench_full(num_steps)

    def run_bench_full(self, num_steps: int) -> list[StepResult]:
        """Full pipeline: sensing + registered DriveCore + registered Policy each step."""
        started = time.perf_counter()
        results = [
            self.run_step(
                include_observations=False,
                include_snapshot=False,
                include_agent_states=False,
            )
            for _ in range(num_steps)
        ]
        self.last_bench_steps_per_second = num_steps / max(
            time.perf_counter() - started, 1e-9
        )
        return results

    def run_bench_world_only(self, num_steps: int) -> None:
        """World stepping only: no sensing, drives, policies or snapshots."""
        actions = {aid: Action(0.0, 0.5, False) for aid in self._world._agents}
        started = time.perf_counter()
        for _ in range(num_steps):
            self._world.step(
                dict(actions),
                include_observations=False,
                include_snapshot=False,
                include_agent_states=False,
            )
        self.last_bench_world_steps_per_second = num_steps / max(
            time.perf_counter() - started, 1e-9
        )

    def snapshot(self) -> Snapshot | None:
        return self._world.snapshot() if self._is_running else None

    def restore(self, snapshot: Snapshot, preserve_action_log: bool = True) -> None:
        """Restore a checkpoint while optionally retaining recorded actions."""
        self._world.restore(snapshot)
        self._world.width = snapshot.width
        self._world.height = snapshot.height
        self._world.dt = snapshot.dt
        self._step_count = snapshot.step
        self._is_running = True
        if not preserve_action_log:
            self.action_log.clear()
            self._snapshots.clear()
        self._snapshot_manager.add(snapshot)

    def export_action_log(self) -> ActionLog:
        return ActionLog.from_steps(self._seed, self.action_log)

    def reset(self) -> None:
        self._is_running = False
        self._world.reset(self._seed)
        self._step_count = 0
        self._snapshots.clear()
        self.action_log.clear()

    def stop(self) -> None:
        self._is_running = False

    def get_stats(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "is_running": self._is_running,
            "world_size": f"{self._world.width:g}x{self._world.height:g}",
        }


class SimEnv:
    """Thin single-agent environment with Gymnasium-like signatures."""

    def __init__(
        self, config: SimLabConfig | None = None, seed: int | None = None
    ) -> None:
        self.runner = Runner(config, seed)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.runner = Runner(
            self.runner._config, self.runner._seed if seed is None else seed
        )
        self.runner.setup()
        result = self.runner._world.snapshot()
        return result.agent_states, {"seed": self.runner._seed}

    def step(self, action: Action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.runner.run_step({"agent_0": action})
        reward = (
            float(result.reward.get("agent_0", 0.0))
            if hasattr(result.reward, "get")
            else float(result.reward)
        )
        return (
            result.observations.get("agent_0"),
            reward,
            result.terminated,
            result.truncated,
            dict(result.info),
        )
