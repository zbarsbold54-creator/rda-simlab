"""Snapshot scrubbing, deterministic replay, and action-log coverage."""

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.core.snapshot import ActionLog
from simlab.core.types import Action


def test_scrub_back_500_steps_and_replay_matches_original_future() -> None:
    runner = Runner(load_config(), seed=71, agents=2)
    runner.setup()
    expected_hashes = []
    for step in range(700):
        actions = {
            agent_id: Action(
                turn=((step + index) % 9 - 4) / 4,
                thrust=((step + 2 * index) % 5) / 4,
                eat=step % 13 == 0,
            )
            for index, agent_id in enumerate(runner._world._agents)
        }
        runner.run_step(actions)
        expected_hashes.append(runner._world.state_hash())

    scrub_step = 700 - 500
    checkpoint = runner._snapshot_manager.get_snapshot(scrub_step)
    assert checkpoint is not None
    recorded_future = runner.action_log[scrub_step:]
    runner.restore(checkpoint)

    for offset, actions in enumerate(recorded_future):
        runner.run_step(actions)
        assert runner._world.state_hash() == expected_hashes[scrub_step + offset]


def test_restore_with_regrowth_matches_at_multiple_scrub_points() -> None:
    runner = Runner(load_config(), seed=71, agents=2)
    runner.setup()
    # Default config now has supply_ratio 2.0 -> effective regrow_rate > 0.
    assert runner._world.food_regrow_rate > 0.0

    total = 800
    expected_hashes = []
    for step in range(total):
        actions = {
            agent_id: Action(
                turn=((step + index) % 9 - 4) / 4,
                thrust=((step + 2 * index) % 5) / 4,
                eat=step % 7 == 0,
            )
            for index, agent_id in enumerate(runner._world._agents)
        }
        runner.run_step(actions)
        expected_hashes.append(runner._world.state_hash())

    # Use more frequent eating to guarantee at least one food patch is
    # partially depleted, so regrowth is observable at every scrub point.
    scrub_points = (100, 300, 550)
    futures = {
        s: runner.action_log[s : s + 200] for s in scrub_points
    }
    for scrub_step in scrub_points:
        checkpoint = runner._snapshot_manager.get_snapshot(scrub_step)
        assert checkpoint is not None
        runner.restore(checkpoint)
        for offset, actions in enumerate(futures[scrub_step]):
            runner.run_step(actions)
            assert runner._world.state_hash() == expected_hashes[scrub_step + offset]


def test_action_log_round_trips_json_and_groups_by_step(tmp_path) -> None:
    steps = [
        {"agent_0": Action(0.25, 0.5, True), "agent_1": Action(-0.5, 0.75)},
        {"agent_0": Action(-1.0, 1.0)},
    ]
    recording = ActionLog.from_steps(19, steps)
    destination = tmp_path / "run.json"
    recording.save(destination)
    loaded = ActionLog.load(destination)
    assert loaded.seed == 19
    assert loaded.actions_by_step() == dict(enumerate(steps))
