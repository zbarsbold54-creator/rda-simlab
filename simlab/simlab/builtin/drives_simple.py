"""Registered placeholder drive core."""

from simlab.core.registry import register
from simlab.interfaces.drive_core import BasicDriveCore


@register("drive", "simple")
class SimpleDriveCore(BasicDriveCore):
    """Placeholder, to be replaced by the real drive core."""
