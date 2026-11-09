"""Phase 2 survival, sensing, drives, and plugin gates."""

import importlib
from pathlib import Path

import numpy as np

from simlab.core.config import SimLabConfig
from simlab.core.registry import get, load_plugins
from simlab.core.types import Action
from simlab.core.world import World
from simlab.interfaces.drive_core import BasicDriveCore, FixedDecay
from simlab.interfaces.policy import HeuristicPolicy


def test_hand_built_scene_has_known_ray_distances_and_types() -> None:
    world = World(seed=4)
    world.add_agent("agent_0", (10.0, 10.0), heading=0.0)
    world.add_food((15.0, 10.0))
    world.add_hazard((14.455, 12.270), detection_radius=0.1)
    observation = world._observation("agent_0")
    assert (observation.ray_types[8], observation.rays[8]) == (1, 5.0)
    assert observation.ray_types[9] == 2
    assert np.isclose(observation.rays[9], 5.0, atol=0.02)
    assert observation.ray_types[0] == 3
    assert all(0.0 <= value <= 20.0 for value in observation.rays)


def test_placeholder_drives_are_bounded() -> None:
    world = World(seed=5)
    world.add_agent("agent_0", (10.0, 10.0), energy=50.0, health=70.0)
    world.add_food((12.0, 12.0))
    world.add_hazard((14.0, 13.0), detection_radius=0.1)
    drives = BasicDriveCore()
    drives.reset(1, world._rng)
    for _ in range(2000):
        observation = world.step({"agent_0": Action(0.1, 0.2)}).observations["agent_0"]
        values = drives.step([observation], {"rng": world._rng})
        assert np.all((values >= 0.0) & (values <= 1.0))


def test_retention_can_be_swapped_by_registered_name(tmp_path) -> None:
    assert get("retention", "fixed_decay") is not None
    assert get("retention", "passthrough") is not None
    fixed = FixedDecay(0.5)
    previous = np.zeros((1, 8), dtype=np.float32)
    drives = np.ones((1, 8), dtype=np.float32)
    assert np.allclose(fixed.step(previous, drives, {}), 0.5)
    config_path = tmp_path / "retention_test.toml"
    config_path.write_text(
        '[components]\nretention = "passthrough"\n', encoding="utf-8"
    )
    try:
        config = SimLabConfig.from_toml(str(config_path))
    finally:
        config_path.unlink(missing_ok=True)
    retention_type = get("retention", config.retention)
    swapped = retention_type()
    changed = swapped.step(previous, drives, {})
    assert config.retention == "passthrough" and not np.array_equal(
        changed, fixed.step(previous, drives, {})
    )


def test_dropped_plugin_is_discovered_and_selectable() -> None:
    import simlab.plugins

    plugin_dir = Path(next(iter(simlab.plugins.__path__)))
    module_path = plugin_dir / "temporary_test_plugin.py"
    module_path.write_text(
        "from simlab.core.registry import register\n"
        "from simlab.core.types import Action\n"
        '@register("policy", "temporary_selectable")\n'
        "class TemporaryPolicy:\n"
        "    def act(self, obs, drive_state=None):\n"
        "        return Action()\n",
        encoding="utf-8",
    )
    try:
        importlib.invalidate_caches()
        load_plugins()
        selected_type = get("policy", "temporary_selectable")
        assert selected_type is not None and selected_type().act(None).thrust == 0.0
    finally:
        module_path.unlink(missing_ok=True)
        sys_module = "simlab.plugins.temporary_test_plugin"
        import sys

        sys.modules.pop(sys_module, None)
        importlib.invalidate_caches()
    assert get("policy", "heuristic") is HeuristicPolicy
