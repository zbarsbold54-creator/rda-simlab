"""Deterministic continuous 2D survival world."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from simlab.core.rng import SimRNG
from simlab.core.sensors import ray_sensors, ray_sensors_batch
from simlab.core.types import (
    Action,
    AgentState,
    Observation,
    Snapshot,
    StepResult,
    canonicalize,
    thaw,
)


class World:
    """A walled continuous arena whose state is owned by this object."""

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        seed: int = 0,
        dt: float = 0.1,
        food_regrow_rate: float = 0.0,
        eat_dist: float = 1.5,
        max_turn_rate: float = 3.0,
        speed_factor: float = 8.0,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.dt = float(dt)
        self._seed = int(seed)
        self._rng = SimRNG(self._seed)
        self._agents: dict[str, dict[str, Any]] = {}
        self._food_patches: list[dict[str, Any]] = []
        self._hazard_patches: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}
        self._step_count = 0
        self.food_regrow_rate = float(food_regrow_rate)
        self.eat_dist = float(eat_dist)
        self.max_turn_rate = float(max_turn_rate)
        self.speed_factor = float(speed_factor)

    @property
    def agents(self) -> MappingProxyType:
        return MappingProxyType(self._agents)

    @property
    def food_patches(self) -> tuple[MappingProxyType, ...]:
        return tuple(MappingProxyType(dict(patch)) for patch in self._food_patches)

    @property
    def hazard_patches(self) -> tuple[MappingProxyType, ...]:
        return tuple(MappingProxyType(dict(patch)) for patch in self._hazard_patches)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = SimRNG(self._seed)
        self._agents.clear()
        self._food_patches.clear()
        self._hazard_patches.clear()
        self._metadata.clear()
        self._step_count = 0

    def add_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        energy: float = 100.0,
        heading: float = 0.0,
        health: float = 100.0,
    ) -> None:
        capacity = 100.0
        clipped_energy = float(np.clip(energy, 0.0, capacity))
        self._agents[agent_id] = {
            "position": self._clamp_point(position),
            "heading": float(heading),
            "speed": 0.0,
            "energy": clipped_energy,
            "max_energy": capacity,
            "health": float(np.clip(health, 0.0, 100.0)),
            "age": 0.0,
            "is_alive": bool(health > 0 and clipped_energy > 0),
            "food_eaten": 0,
        }

    def remove_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def add_food(
        self,
        position: tuple[float, float],
        count: int = 1,
        amount: float = 20.0,
        max_amount: float | None = None,
    ) -> None:
        point = self._clamp_point(position)
        capacity = float(max(0.0, amount if max_amount is None else max_amount))
        for _ in range(max(0, int(count))):
            self._food_patches.append(
                {
                    "position": point,
                    "amount": float(max(0.0, amount)),
                    "max_amount": capacity,
                }
            )

    def add_hazard(
        self,
        position: tuple[float, float],
        count: int = 1,
        damage: float = 12.0,
        radius: float = 1.5,
        detection_radius: float = 30.0,
    ) -> None:
        point = self._clamp_point(position)
        for _ in range(max(0, int(count))):
            self._hazard_patches.append(
                {
                    "position": point,
                    "damage": float(max(0.0, damage)),
                    "radius": float(max(0.1, radius)),
                    "detection_radius": float(max(0.1, detection_radius)),
                    "heading": 0.0,
                }
            )

    def step(
        self,
        actions: dict[str, Action],
        *,
        include_observations: bool = True,
        include_snapshot: bool = True,
        include_agent_states: bool = True,
    ) -> StepResult:
        self._step_count += 1
        self._move_hazards()
        rewards: dict[str, float] = {}
        for agent_id, state in self._agents.items():
            if not state["is_alive"]:
                rewards[agent_id] = 0.0
                continue
            action = actions.get(agent_id, Action())
            was_full = state["energy"] >= state["max_energy"]
            self._move_agent(state, action)
            reward = -0.01
            if action.eat:
                reward += self._try_eat(state, was_full)
            self._apply_hazard_damage(state)
            if state["energy"] <= 0.0:
                state["health"] = max(0.0, state["health"] - 0.5)
            if state["health"] <= 0.0:
                state["is_alive"] = False
            rewards[agent_id] = reward
        states = self._state_view() if include_agent_states else {}
        observations = self._observation_batch() if include_observations else {}
        self._regrow_food()
        snapshot = self.snapshot() if include_snapshot else None
        terminated = not any(state["is_alive"] for state in self._agents.values())
        return StepResult(
            MappingProxyType(states),
            MappingProxyType(observations),
            snapshot,
            terminated,
            reward=MappingProxyType(rewards),
            info=MappingProxyType({"step": self._step_count}),
        )

    def _regrow_food(self) -> None:
        rate = self.food_regrow_rate
        if rate <= 0.0:
            return
        for patch in self._food_patches:
            max_amount = patch.get("max_amount", patch["amount"])
            if patch["amount"] < max_amount:
                patch["amount"] = min(max_amount, patch["amount"] + rate)

    def _move_agent(self, state: dict[str, Any], action: Action) -> None:
        state["heading"] = (
            state["heading"] + action.turn * self.max_turn_rate * self.dt
        ) % (2.0 * math.pi)
        state["speed"] = action.thrust * self.speed_factor
        x, y = state["position"]
        state["position"] = self._clamp_point(
            (
                x + math.cos(state["heading"]) * state["speed"] * self.dt,
                y + math.sin(state["heading"]) * state["speed"] * self.dt,
            )
        )
        state["energy"] = max(
            0.0, state["energy"] - (0.08 + 0.04 * state["speed"]) * self.dt
        )
        state["age"] += self.dt

    def _move_hazards(self) -> None:
        alive = [state for state in self._agents.values() if state["is_alive"]]
        for hazard in self._hazard_patches:
            hx, hy = hazard["position"]
            nearest_distance = hazard["detection_radius"] ** 2
            target = None
            for state in alive:
                dx = state["position"][0] - hx
                dy = state["position"][1] - hy
                distance_squared = dx * dx + dy * dy
                if distance_squared <= nearest_distance:
                    nearest_distance = distance_squared
                    target = state
            if target is not None:
                hazard["heading"] = math.atan2(
                    target["position"][1] - hy, target["position"][0] - hx
                )
            else:
                hazard["heading"] += float(self._rng.uniform(-0.4, 0.4)) * self.dt
            speed = 2.0
            hazard["position"] = self._clamp_point(
                (
                    hx + math.cos(hazard["heading"]) * speed * self.dt,
                    hy + math.sin(hazard["heading"]) * speed * self.dt,
                )
            )

    def _try_eat(self, state: dict[str, Any], was_full: bool = False) -> float:
        if was_full:
            return 0.0
        for patch in self._food_patches:
            dx = state["position"][0] - patch["position"][0]
            dy = state["position"][1] - patch["position"][1]
            if patch["amount"] <= 0.0 or dx * dx + dy * dy > self.eat_dist**2:
                continue
            available = min(patch["amount"], state["max_energy"] - state["energy"])
            if available <= 0.0:
                return 0.0
            state["energy"] += available
            patch["amount"] -= available
            state["food_eaten"] += 1
            return 1.0
        return 0.0

    def _apply_hazard_damage(self, state: dict[str, Any]) -> None:
        for hazard in self._hazard_patches:
            dx = state["position"][0] - hazard["position"][0]
            dy = state["position"][1] - hazard["position"][1]
            if dx * dx + dy * dy <= hazard["radius"] ** 2:
                state["health"] = max(0.0, state["health"] - hazard["damage"] * self.dt)

    def _state_view(self) -> dict[str, AgentState]:
        return {
            agent_id: AgentState(
                agent_id,
                tuple(state["position"]),
                state["heading"],
                state["speed"],
                state["energy"],
                state["max_energy"],
                state["health"],
                state["age"],
                state["is_alive"],
                state["food_eaten"],
            )
            for agent_id, state in self._agents.items()
        }

    def _observation(self, agent_id: str) -> Observation:
        return self._observation_batch()[agent_id]

    def _observation_batch(self) -> dict[str, Observation]:
        if not self._agents:
            return {}
        agent_ids = tuple(self._agents)
        states = tuple(self._agents[agent_id] for agent_id in agent_ids)
        positions = np.asarray(
            [state["position"] for state in states], dtype=np.float64
        )
        headings = np.asarray([state["heading"] for state in states], dtype=np.float64)
        # Depleted food patches (amount <= 0) are invisible to sensors so that
        # policies cannot chase empty patches forever.
        food_positions = [
            patch["position"] for patch in self._food_patches if patch["amount"] > 0.0
        ]
        food_arr = (
            np.asarray(food_positions, dtype=np.float64).reshape(-1, 2)
            if food_positions
            else np.empty((0, 2), dtype=np.float64)
        )
        distances, types = ray_sensors_batch(
            positions,
            headings,
            self.width,
            self.height,
            food_arr,
            np.asarray(
                [patch["position"] for patch in self._hazard_patches],
                dtype=np.float64,
            ).reshape(-1, 2),
            positions,
        )
        observations = {}
        for index, (agent_id, state) in enumerate(zip(agent_ids, states)):
            energy = state["energy"]
            observations[agent_id] = Observation(
                tuple(distances[index].tolist()),
                tuple(types[index].tolist()),
                (
                    energy / state["max_energy"],
                    state["health"] / 100.0,
                    state["speed"] / self.speed_factor,
                ),
                tuple(state["position"]),
                energy,
                self._step_count,
            )
        return observations

    def _ray_sensors(
        self, position: tuple[float, float], heading: float, count: int = 16
    ) -> tuple[list[float], list[int]]:
        return ray_sensors(
            position,
            heading,
            self.width,
            self.height,
            tuple(patch["position"] for patch in self._food_patches),
            tuple(patch["position"] for patch in self._hazard_patches),
            tuple(state["position"] for state in self._agents.values()),
            count,
        )

    def snapshot(self) -> Snapshot:
        food = tuple(
            MappingProxyType(copy.deepcopy(patch)) for patch in self._food_patches
        )
        hazards = tuple(
            MappingProxyType(copy.deepcopy(patch)) for patch in self._hazard_patches
        )
        return Snapshot(
            self._step_count,
            MappingProxyType(self._state_view()),
            food,
            hazards,
            copy.deepcopy(self._rng.state),
            self.width,
            self.height,
            self.dt,
            MappingProxyType(
                {
                    "seed": self._seed,
                    "food_regrow_rate": self.food_regrow_rate,
                    "eat_dist": self.eat_dist,
                    "max_turn_rate": self.max_turn_rate,
                    "speed_factor": self.speed_factor,
                    **copy.deepcopy(self._metadata),
                }
            ),
        )

    def restore(self, snapshot: Snapshot) -> None:
        self._seed = int(snapshot.metadata.get("seed", self._seed))
        self._step_count = snapshot.step
        self._food_patches = [thaw(patch) for patch in snapshot.food_patches]
        self._hazard_patches = [thaw(patch) for patch in snapshot.hazard_patches]
        self._agents = {
            agent_id: {
                "position": tuple(state.position),
                "heading": state.heading,
                "speed": state.speed,
                "energy": state.energy,
                "max_energy": state.max_energy,
                "health": state.health,
                "age": state.age,
                "is_alive": state.alive,
                "food_eaten": state.food_eaten,
            }
            for agent_id, state in snapshot.agent_states.items()
        }
        self._rng = SimRNG(0)
        self._rng.state = thaw(snapshot.rng_state)
        self.food_regrow_rate = float(snapshot.metadata.get("food_regrow_rate", 0.0))
        self.eat_dist = float(snapshot.metadata.get("eat_dist", 1.5))
        self.max_turn_rate = float(snapshot.metadata.get("max_turn_rate", 3.0))
        self.speed_factor = float(snapshot.metadata.get("speed_factor", 8.0))
        self._metadata = {
            key: thaw(value)
            for key, value in snapshot.metadata.items()
            if key
            not in ("seed", "food_regrow_rate", "eat_dist", "max_turn_rate", "speed_factor")
        }

    def state_hash(self) -> str:
        canonical = canonicalize(self.snapshot().to_dict())
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _clamp_point(self, point: tuple[float, float]) -> tuple[float, float]:
        return (
            float(min(max(point[0], 0.0), self.width)),
            float(min(max(point[1], 0.0), self.height)),
        )

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        return self._agents.get(agent_id)

    def get_step_count(self) -> int:
        return self._step_count

    def get_world_state(self) -> np.ndarray:
        return self.snapshot().world_state
