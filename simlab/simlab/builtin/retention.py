"""Registered retention rules."""

from simlab.core.registry import register
from simlab.interfaces.drive_core import FixedDecay


@register("retention", "fixed_decay")
class RegisteredFixedDecay(FixedDecay):
    pass


@register("retention", "passthrough")
class PassthroughRetention:
    def step(self, previous, drives, context):
        return drives.copy()
