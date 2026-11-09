"""Scrub-back replay with the example learned-policy plugin."""

from dataclasses import replace

from simlab.core.config import load_config
from simlab.core.runner import Runner


def _cfg() -> object:
    return replace(load_config(), policy="example_learned", drive="simple")


def test_scrub_back_replay_matches_when_policy_and_drive_state_restored() -> None:
    agents = 3
    n = 300
    k = 200
    runner = Runner(_cfg(), seed=202, agents=agents)
    runner.setup()
    runner._ensure_components()
    runner._policy.reset(agents, None)
    runner._drive_core.reset(agents, None)

    for _ in range(n):
        runner.run_step()
    saved_policy = runner._policy.get_state()
    saved_drive = runner._drive_core.get_state()

    expected_hashes = []
    for _ in range(k):
        runner.run_step()
        expected_hashes.append(runner._world.state_hash())

    checkpoint = runner._snapshot_manager.get_snapshot(n)
    assert checkpoint is not None
    runner.restore(checkpoint)
    runner._policy.set_state(saved_policy)
    runner._drive_core.set_state(saved_drive)

    for offset in range(k):
        runner.run_step()
        assert runner._world.state_hash() == expected_hashes[offset]


def test_scrub_back_breaks_if_policy_state_not_restored() -> None:
    agents = 3
    n = 300
    k = 60
    runner = Runner(_cfg(), seed=202, agents=agents)
    runner.setup()
    runner._ensure_components()
    runner._policy.reset(agents, None)
    runner._drive_core.reset(agents, None)

    for _ in range(n):
        runner.run_step()
    saved_drive = runner._drive_core.get_state()

    expected_hashes = []
    for _ in range(k):
        runner.run_step()
        expected_hashes.append(runner._world.state_hash())

    checkpoint = runner._snapshot_manager.get_snapshot(n)
    runner.restore(checkpoint)
    # Fresh policy state (zeros) instead of the saved recurrent state.
    runner._policy.reset(agents, None)
    runner._drive_core.set_state(saved_drive)

    replay_hashes = []
    for _ in range(k):
        runner.run_step()
        replay_hashes.append(runner._world.state_hash())

    assert replay_hashes != expected_hashes
