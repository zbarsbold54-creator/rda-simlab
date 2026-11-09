"""Minimal task protocol implementation."""

from simlab.core.registry import register


@register("task", "survive")
class SurviveTask:
    """Sparse survival reward: +1 per alive step, 0 at termination."""

    def reward(self, result: object) -> float:
        return 0.0 if getattr(result, "terminated", False) else 1.0


@register("task", "survival_shaped")
class SurvivalShapedTask:
    """Shaped survival reward for the learner.

    Per agent per step:
      - +0.01 for staying alive;
      - -1.0 on the death step;
      - +0.05 * (delta energy fraction) on any step with a positive energy
        gain (eating), where delta energy fraction is relative to max energy.

    The task is stateful: it remembers each agent's previous energy to size
    the gain term. Use ``reset(n_agents, max_energy)`` at episode start and
    ``reward(index, energy, alive_now, alive_prev)`` per step.
    """

    ALIVE = 0.01
    DEATH = -1.0
    GAIN = 0.05

    def __init__(self) -> None:
        self._max_energy = 100.0
        self._prev: list[float | None] = []

    def reset(self, n_agents: int, max_energy: float = 100.0) -> None:
        self._max_energy = float(max_energy)
        self._prev = [None] * int(n_agents)

    def reward(
        self,
        index: int,
        energy: float,
        alive_now: bool,
        alive_prev: bool = True,
    ) -> float:
        r = self.ALIVE if alive_now else 0.0
        if not alive_now and alive_prev:
            r += self.DEATH
        prev = self._prev[index]
        if prev is not None and energy > prev:
            r += self.GAIN * (energy - prev) / self._max_energy
        self._prev[index] = float(energy)
        return r
