"""Minimal metrics implementation."""

from simlab.core.registry import register


@register("metric", "survival_steps")
class SurvivalSteps:
    def compute(self, history: list[object]) -> float:
        return float(len(history))
