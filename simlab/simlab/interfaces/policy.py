"""Policy protocol and deterministic built-in policies."""

from __future__ import annotations

from typing import Any, Protocol

from simlab.core.types import Action


class Policy(Protocol):
    needs_observations: bool

    def act(self, obs: Any, drive_state: Any = None) -> Action: ...


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def derive_eat_thrust(eat_dist, max_turn_rate, speed_factor) -> float:
    """Thrust s.t. min turn radius v/omega_max <= 0.7 * eat_range.

    A thrust above this lets the agent orbit a patch just outside eat range
    and never connect. Defaults mirror the historical hardcoded constant.
    """
    if max_turn_rate <= 0.0 or speed_factor <= 0.0:
        return 0.4
    return (0.7 * float(eat_dist) * float(max_turn_rate)) / float(speed_factor)


class HeuristicPolicyNoflee:
    """Sensor-based heuristic without hazard fleeing (historic behavior).

    Uses only obs (rays + proprioception), no World access. Near-food thrust
    is derived from the world's eat range and max turn rate (see
    ``derive_eat_thrust``) rather than hardcoded.
    """

    needs_observations = True

    CLOSE_DIST = 5.0
    THRUST_SEEK = 0.8
    STEER_GAIN = 60.0

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        self._seed = seed
        food = getattr(config, "food", None)
        agent = getattr(config, "agent", None)
        self.eat_dist = float(getattr(food, "eat_dist", 1.5))
        self.max_turn_rate = float(getattr(agent, "max_turn_rate", 3.0))
        self.speed_factor = float(getattr(agent, "speed_factor", 8.0))
        self.eat_thrust = derive_eat_thrust(
            self.eat_dist, self.max_turn_rate, self.speed_factor
        )
        # All near-food branches share the derived thrust and steering gain.
        self.close_thrust = self.eat_thrust
        self.eat_steer_divisor = self.STEER_GAIN
        self.close_steer_divisor = self.STEER_GAIN

    @staticmethod
    def _best_food(rays, ray_types):
        best_dist = None
        best_index = None
        for index, (distance, kind) in enumerate(zip(rays, ray_types)):
            if kind == 1 and (best_dist is None or distance < best_dist):
                best_dist = distance
                best_index = index
        return best_dist, best_index

    def _act(self, obs):
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())
        best_dist, best_index = self._best_food(rays, ray_types)
        if best_dist is None or best_index is None:
            return Action(0.25, 0.5, False)

        count = len(rays)
        middle = count // 2
        offset_deg = (
            (best_index - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
        )

        if best_dist < self.eat_dist:
            # Decelerate so the minimum turning radius (v / omega_max) stays
            # below the eat range; otherwise the agent orbits just outside it.
            turn = _clamp(offset_deg / self.eat_steer_divisor)
            return Action(turn, self.eat_thrust, True, "heuristic")
        if best_dist < self.CLOSE_DIST:
            # Near food: steer toward it and ease off to converge instead of
            # orbiting just outside eat range.
            turn = _clamp(offset_deg / self.close_steer_divisor)
            return Action(turn, self.close_thrust, False, "heuristic")
        turn = _clamp(offset_deg / 135.0)
        return Action(turn, self.THRUST_SEEK, False, "heuristic")

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return self._act(obs)


class HeuristicPolicy(HeuristicPolicyNoflee):
    """Default sensor heuristic: flee hazards when threat > 0.3, else seek/wander."""

    @staticmethod
    def _threat(rays, ray_types) -> float:
        hazards = [
            distance
            for distance, kind in zip(rays, ray_types)
            if kind == 2
        ]
        nearest = min(hazards) if hazards else 20.0
        return max(0.0, min(1.0, 1.0 - nearest / 20.0))

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if obs is None:
            return Action(0.25, 0.5, False)
        rays = getattr(obs, "rays", ())
        ray_types = getattr(obs, "ray_types", ())
        if self._threat(rays, ray_types) > 0.3:
            count = len(rays)
            middle = count // 2
            best = -1
            nearest = None
            for index, (distance, kind) in enumerate(zip(rays, ray_types)):
                if kind == 2 and (nearest is None or distance < nearest):
                    nearest = distance
                    best = index
            offset_deg = (
                (best - middle) * (270.0 / (count - 1)) if count > 1 else 0.0
            )
            if abs(offset_deg) < 15.0:
                turn = -1.0
            else:
                turn = max(-1.0, min(1.0, -offset_deg / 135.0))
            return Action(turn, 0.8, False, "heuristic_flee")
        return self._act(obs)


class HeuristicPolicyV2(HeuristicPolicy):
    """Sensor heuristic from the world-v1 era (pre round-5 eat fix).

    Eat branch used thrust 0.6 with steer /135; close branch thrust 0.5.
    Frozen for reproduction of the world-v1 baseline table.
    """

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        super().__init__(seed, config)
        self.eat_thrust = 0.6
        self.close_thrust = 0.5
        self.eat_steer_divisor = 135.0


class RandomPolicy:
    needs_observations = False

    def __init__(self, seed: int = 0) -> None:
        self._rng = __import__("numpy").random.default_rng(seed)

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action(
            float(self._rng.uniform(-1, 1)), float(self._rng.random()), False, "random"
        )


class LearnedPolicy:
    needs_observations = False

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action()
