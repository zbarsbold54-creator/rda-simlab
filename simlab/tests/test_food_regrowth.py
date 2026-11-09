"""Food regrowth: deterministic, hashed, and preserved by snapshot/restore."""

from __future__ import annotations

from simlab.core.world import World


def _world(rate: float = 0.0) -> World:
    world = World(50, 50, seed=7, food_regrow_rate=rate)
    world.add_food((10.0, 10.0), amount=5.0, max_amount=20.0)
    return world


def _step(world: World, n: int) -> None:
    for _ in range(n):
        world.step(
            {},
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )


def test_no_regrowth_by_default() -> None:
    world = _world(0.0)
    _step(world, 50)
    assert world._food_patches[0]["amount"] == 5.0
    assert world._food_patches[0]["max_amount"] == 20.0


def test_regrowth_is_deterministic_and_capped() -> None:
    a = _world(0.01)
    b = _world(0.01)
    _step(a, 2000)
    _step(b, 2000)
    amount_a = a._food_patches[0]["amount"]
    amount_b = b._food_patches[0]["amount"]
    assert amount_a == amount_b
    assert amount_a == 20.0  # regrown up to max_amount
    assert a.state_hash() == b.state_hash()


def test_regrowth_rate_is_part_of_state_hash() -> None:
    base = _world(0.0)
    regrowing = _world(0.01)
    assert base.state_hash() != regrowing.state_hash()


def test_snapshot_restore_preserves_regrowth() -> None:
    world = _world(0.01)
    _step(world, 100)
    mid = world.snapshot()
    expected_after_mid = []
    continue_world = _world(0.01)
    continue_world.restore(mid)
    # Expected regrowth continuing from the restored snapshot.
    for _ in range(100):
        continue_world.step(
            {},
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        expected_after_mid.append(continue_world._food_patches[0]["amount"])

    fresh = _world(0.01)
    _step(fresh, 100)
    fresh.restore(mid)
    actual_after_mid = []
    for _ in range(100):
        fresh.step(
            {},
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        actual_after_mid.append(fresh._food_patches[0]["amount"])
    assert actual_after_mid == expected_after_mid
