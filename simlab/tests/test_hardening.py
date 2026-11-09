"""Phase 1 hardening coverage."""

import numpy as np

from simlab.core.types import Action
from simlab.core.world import World


def base_world() -> World:
    world = World(seed=21)
    world.add_agent("agent_0", (10.0, 10.0), energy=90.0)
    world.add_food((10.0, 10.0), amount=20.0)
    world.add_hazard((20.0, 20.0))
    return world


def test_agent_energy_is_hashed() -> None:
    world = base_world()
    before = world.state_hash()
    world._agents["agent_0"]["energy"] += 1.0
    assert world.state_hash() != before


def test_food_amount_is_hashed() -> None:
    world = base_world()
    before = world.state_hash()
    world._food_patches[0]["amount"] += 1.0
    assert world.state_hash() != before


def test_hazard_position_is_hashed() -> None:
    world = base_world()
    before = world.state_hash()
    world._hazard_patches[0]["position"] = (21.0, 20.0)
    assert world.state_hash() != before


def test_step_counter_is_hashed() -> None:
    world = base_world()
    before = world.state_hash()
    world._step_count += 1
    assert world.state_hash() != before


def test_rng_state_is_hashed() -> None:
    world = base_world()
    before = world.state_hash()
    world._rng.random()
    assert world.state_hash() != before


def test_last_large_array_element_is_hashed() -> None:
    world = base_world()
    world._metadata["large_array"] = np.zeros((1024, 1024), dtype=np.float32)
    before = world.state_hash()
    world._metadata["large_array"][-1, -1] = 1.0
    assert world.state_hash() != before


def test_snapshot_disk_round_trip_replays_future(tmp_path) -> None:
    world = base_world()
    for _ in range(10):
        world.step({"agent_0": Action(0.2, 0.5, False)})
    snapshot = world.snapshot()
    base = tmp_path / "simlab_snapshot"
    snapshot.save(str(base))
    loaded = snapshot.load(str(base))
    restored = World(seed=999)
    restored.restore(loaded)
    assert restored.state_hash() == world.state_hash()
    for index in range(200):
        action = Action(((index % 5) - 2) / 2.0, 0.6, index % 17 == 0)
        world.step({"agent_0": action})
        restored.step({"agent_0": action})
        assert restored.state_hash() == world.state_hash()


def test_full_or_missed_food_does_not_count() -> None:
    world = World(seed=1)
    world.add_agent("agent_0", (10.0, 10.0), energy=100.0)
    world.add_food((10.0, 10.0), amount=20.0)
    world.step({"agent_0": Action(0.0, 0.0, True)})
    full_count = world._agents["agent_0"]["food_eaten"]
    world.add_food((50.0, 50.0), amount=20.0)
    world._agents["agent_0"]["energy"] = 90.0
    world._agents["agent_0"]["position"] = (30.0, 30.0)
    world.step({"agent_0": Action(0.0, 0.0, True)})
    assert [full_count, world._agents["agent_0"]["food_eaten"]] == [0, 0]
