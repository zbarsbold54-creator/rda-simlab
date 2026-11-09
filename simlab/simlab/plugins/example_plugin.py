"""Example plugin proving that registration requires no core edits."""

from simlab.core.registry import register
from simlab.core.types import Action


@register("policy", "example_idle")
class ExampleIdlePolicy:
    def act(self, obs: object, drive_state: object = None) -> Action:
        return Action()
