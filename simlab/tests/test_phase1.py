"""Phase 1 deterministic-core gates."""

from simlab.core.types import Action
from simlab.core.world import World


def actions(n: int) -> list[dict[str, Action]]:
    return [
        {"agent_0": Action(turn=((i % 7) - 3) / 3, thrust=0.75, eat=False)}
        for i in range(n)
    ]


def make_world(seed: int) -> World:
    world = World(seed=seed)
    world.add_agent("agent_0", (50.0, 50.0), 100.0, heading=0.2)
    world.add_food((65.0, 50.0), 3)
    return world


def test_agent_moves() -> None:
    world = make_world(1)
    start = world.snapshot().agent_states["agent_0"].position
    for action in actions(200):
        world.step(action)
    end = world.snapshot().agent_states["agent_0"].position
    assert abs(end[0] - start[0]) + abs(end[1] - start[1]) > 0.1


def test_seeds_differ() -> None:
    first, second = make_world(1), make_world(2)
    hashes = []
    for _ in range(20):
        hashes.append((first.state_hash(), second.state_hash()))
        first.step({"agent_0": Action(0.1, 0.5)})
        second.step({"agent_0": Action(0.1, 0.5)})
    assert any(a != b for a, b in hashes)


def test_same_seed_same_hashes() -> None:
    a, b = make_world(7), make_world(7)
    for action in actions(100):
        a.step(action)
        b.step(action)
        assert a.state_hash() == b.state_hash()


def test_snapshot_restores_future() -> None:
    world = make_world(11)
    for action in actions(500):
        world.step(action)
    snap = world.snapshot()
    future = []
    for action in actions(500):
        world.step(action)
        future.append(world.state_hash())
    restored = World(seed=999)
    restored.restore(snap)
    replay = []
    for action in actions(500):
        restored.step(action)
        replay.append(restored.state_hash())
    assert replay == future


def test_action_log_replay() -> None:
    world = make_world(13)
    log = actions(100)
    for action in log:
        world.step(action)
    expected = world.state_hash()
    replay = make_world(13)
    for action in log:
        replay.step(action)
    assert replay.state_hash() == expected
