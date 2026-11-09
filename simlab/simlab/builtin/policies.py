"""Registered policies."""

import math
from typing import Any

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.interfaces.policy import (
    HeuristicPolicy,
    HeuristicPolicyNoflee,
    HeuristicPolicyV2,
    RandomPolicy,
    _clamp,
    derive_eat_thrust,
)

register("policy", "heuristic")(HeuristicPolicy)
register("policy", "heuristic_noflee")(HeuristicPolicyNoflee)
register("policy", "heuristic_v2")(HeuristicPolicyV2)
register("policy", "random")(RandomPolicy)


class OracleHeuristicPolicyNoflee:
    """World-state heuristic without hazard fleeing (historic behavior)."""

    needs_observations = False
    CLOSE_DIST = 5.0
    THRUST_SEEK = 0.8
    STEER_GAIN = 60.0

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        food = getattr(config, "food", None)
        agent = getattr(config, "agent", None)
        self.eat_dist = float(getattr(food, "eat_dist", 1.5))
        self.max_turn_rate = float(getattr(agent, "max_turn_rate", 3.0))
        self.speed_factor = float(getattr(agent, "speed_factor", 8.0))
        self.eat_thrust = derive_eat_thrust(
            self.eat_dist, self.max_turn_rate, self.speed_factor
        )
        self.close_thrust = self.eat_thrust
        self.eat_steer_divisor = self.STEER_GAIN
        self.close_steer_divisor = self.STEER_GAIN

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        return Action(0.0, 0.8, True)

    def select_action(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        available_actions: list[Any],
        step: int,
    ) -> Action:
        # Mirror the sensor HeuristicPolicy control law, using world geometry
        # instead of rays. The sensor sees no depleted patches (they are
        # filtered from observations), so the oracle must apply the same filter.
        position = agent_state.get("position", (0.0, 0.0))
        food = [
            patch
            for patch in world_state.get("food_patches", [])
            if patch[1] > 0.0
        ]
        if not food:
            return Action(0.25, 0.5, False)
        target, _amount = min(
            food, key=lambda p: (p[0][0] - position[0]) ** 2 + (p[0][1] - position[1]) ** 2
        )
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        heading = agent_state.get("heading", 0.0)
        delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
        dist2 = (position[0] - target[0]) ** 2 + (position[1] - target[1]) ** 2
        dist = math.sqrt(dist2)
        if dist < self.eat_dist:
            turn = _clamp(math.degrees(delta) / self.eat_steer_divisor)
            return Action(turn, self.eat_thrust, True, "heuristic")
        if dist < self.CLOSE_DIST:
            turn = _clamp(math.degrees(delta) / self.close_steer_divisor)
            return Action(turn, self.close_thrust, False, "heuristic")
        turn = _clamp(math.degrees(delta) / 135.0)
        return Action(turn, self.THRUST_SEEK, False, "heuristic")


class OracleHeuristicPolicy(OracleHeuristicPolicyNoflee):
    """Default world-state heuristic: flee hazards when threat > 0.3."""

    def select_action(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        available_actions: list[Any],
        step: int,
    ) -> Action:
        position = agent_state.get("position", (0.0, 0.0))
        hazards = world_state.get("hazard_locations", [])
        nearest = None
        for hx, hy in hazards:
            dist2 = (hx - position[0]) ** 2 + (hy - position[1]) ** 2
            if nearest is None or dist2 < nearest:
                nearest = dist2
        threat = 1.0 - (math.sqrt(nearest) if nearest is not None else 20.0) / 20.0
        if threat > 0.3:
            hx, hy = min(
                hazards,
                key=lambda h: (h[0] - position[0]) ** 2 + (h[1] - position[1]) ** 2,
            )
            desired = math.atan2(position[1] - hy, position[0] - hx)  # away
            heading = agent_state.get("heading", 0.0)
            delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
            turn = _clamp(math.degrees(delta) / 135.0)
            return Action(turn, 0.8, False, "oracle_flee")
        return super().select_action(agent_state, world_state, available_actions, step)


class OracleHeuristicPolicyV2(OracleHeuristicPolicy):
    """World-state heuristic from the world-v1 era (pre principled-thrust).

    Eat branch used thrust 0.6 with steer /135; close branch thrust 0.5.
    Frozen for reproduction of the world-v1 baseline table.
    """

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        super().__init__(seed, config)
        self.eat_thrust = 0.6
        self.close_thrust = 0.5
        self.eat_steer_divisor = 135.0


class OracleHeuristicAvoid(OracleHeuristicPolicyNoflee):
    """World-state upper-bound probe that actively avoids hazards.

    Reference only - not a baseline and not claimed optimal. It forages like
    the sensor heuristic but (a) prefers food targets outside any hazard's
    detection radius, (b) when a hazard is within detection radius plus a
    margin, flees at thrust 1.0 along a direction derived from ALL hazards,
    sliding tangentially along walls and pushing out of corners, and (c) never
    idles inside a hazard's detection radius.
    """

    needs_observations = False
    FLEE_THRUST = 1.0
    FLEE_MARGIN = 4.0
    WALL_MARGIN = 8.0

    def __init__(self, seed: int = 0, config: Any = None) -> None:
        super().__init__(seed, config)
        hazard = getattr(config, "hazard", None)
        world = getattr(config, "world", None)
        self.hazard_detection = float(
            getattr(hazard, "detection_radius", 30.0)
        )
        self.arena_w = float(getattr(world, "width", 50.0))
        self.arena_h = float(getattr(world, "height", 50.0))

    def _danger_zone(self, position: tuple[float, float], hazards) -> bool:
        r = self.hazard_detection + self.FLEE_MARGIN
        r2 = r * r
        return any(
            (h[0] - position[0]) ** 2 + (h[1] - position[1]) ** 2 <= r2
            for h in hazards
        )

    # Patches below this remaining amount are treated as crumbs: the energy
    # gained per bite (the amount) is less than the movement cost of grazing
    # them, so they are only selected when no fuller patch exists.
    AMOUNT_EPSILON = 0.5

    @staticmethod
    def _pick_seed_target(pool):
        # pool: list of (dist2, target, amount). Prefer the nearest patch
        # with a meaningful remaining amount; if every candidate is a crumb,
        # fall back to the largest remaining amount (least depleted).
        big = [c for c in pool if c[2] > OracleHeuristicAvoid.AMOUNT_EPSILON]
        if big:
            return min(big, key=lambda s: s[0])[1]
        return max(pool, key=lambda s: s[2])[1]

    def _select_food(self, position, food, hazards):
        r = self.hazard_detection + self.FLEE_MARGIN
        r2 = r * r
        safe = []
        unsafe = []
        for target, amount in food:
            if amount <= 0.0:
                continue
            d2 = (target[0] - position[0]) ** 2 + (
                target[1] - position[1]
            ) ** 2
            risky = any(
                (h[0] - target[0]) ** 2 + (h[1] - target[1]) ** 2 <= r2
                for h in hazards
            )
            (unsafe if risky else safe).append((d2, target, amount))
        if safe:
            return self._pick_seed_target(safe)
        if unsafe:
            return self._pick_seed_target(unsafe)
        return None

    def _escape_dir(self, position, hazards) -> tuple[float, float]:
        vx = 0.0
        vy = 0.0
        for h in hazards:
            dx = position[0] - h[0]
            dy = position[1] - h[1]
            d = max(math.hypot(dx, dy), 1e-6)
            w = 1.0 / (d * d)
            vx += dx / d * w
            vy += dy / d * w
        n = math.hypot(vx, vy)
        ux = vx / n if n > 1e-9 else 0.0
        uy = vy / n if n > 1e-9 else 0.0
        if position[0] < self.WALL_MARGIN and ux < 0.0:
            ux = 0.0
        if self.arena_w - position[0] < self.WALL_MARGIN and ux > 0.0:
            ux = 0.0
        if position[1] < self.WALL_MARGIN and uy < 0.0:
            uy = 0.0
        if self.arena_h - position[1] < self.WALL_MARGIN and uy > 0.0:
            uy = 0.0
        n2 = math.hypot(ux, uy)
        if n2 > 1e-9:
            return ux / n2, uy / n2
        dx = self.arena_w / 2.0 - position[0]
        dy = self.arena_h / 2.0 - position[1]
        n3 = math.hypot(dx, dy)
        if n3 > 1e-9:
            return dx / n3, dy / n3
        return 0.0, 0.0

    def select_action(
        self,
        agent_state: dict[str, Any],
        world_state: dict[str, Any],
        available_actions: list[Any],
        step: int,
    ) -> Action:
        position = agent_state.get("position", (0.0, 0.0))
        heading = agent_state.get("heading", 0.0)
        hazards = world_state.get("hazard_locations", [])
        if self._danger_zone(position, hazards):
            ux, uy = self._escape_dir(position, hazards)
            desired = math.atan2(uy, ux)
            delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
            turn = _clamp(math.degrees(delta) / 135.0)
            return Action(turn, self.FLEE_THRUST, False, "oracle_avoid_flee")
        target = self._select_food(position, world_state.get("food_patches", []), hazards)
        if target is None:
            return Action(0.25, 0.5, False, "oracle_avoid_wander")
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        delta = (desired - heading + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(target[0] - position[0], target[1] - position[1])
        if dist < self.eat_dist:
            turn = _clamp(math.degrees(delta) / self.eat_steer_divisor)
            return Action(turn, self.eat_thrust, True, "oracle_avoid_eat")
        if dist < self.CLOSE_DIST:
            turn = _clamp(math.degrees(delta) / self.close_steer_divisor)
            return Action(turn, self.close_thrust, False, "oracle_avoid_close")
        turn = _clamp(math.degrees(delta) / 135.0)
        return Action(turn, self.THRUST_SEEK, False, "oracle_avoid_seek")


register("policy", "oracle_noflee")(OracleHeuristicPolicyNoflee)
register("policy", "oracle_heuristic")(OracleHeuristicPolicyNoflee)
register("policy", "oracle")(OracleHeuristicPolicy)
register("policy", "oracle_v2")(OracleHeuristicPolicyV2)
register("policy", "oracle_avoid")(OracleHeuristicAvoid)
