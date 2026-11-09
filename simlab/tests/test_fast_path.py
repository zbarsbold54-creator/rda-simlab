"""Determinism checks for the GUI worker's Max-speed simulation path."""

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.gui import theme
from simlab.gui.sim_worker import SimWorker


def test_max_batch_matches_headless_hashes_for_1000_steps() -> None:
    config = load_config()
    seed = 3407
    headless = Runner(config, seed=seed, agents=20)
    headless.setup()
    worker = SimWorker(config, seed=seed, agents=20)
    worker.set_speed("Max")

    assert worker._speed == "Max"
    assert worker._runner._world.state_hash() == headless._world.state_hash()
    for batch_start in range(0, 1000, theme.MAX_BATCH_STEPS):
        batch_size = min(theme.MAX_BATCH_STEPS, 1000 - batch_start)
        for _ in range(batch_size):
            advanced = worker._advance_one_step()
            assert advanced is not None
            headless.run_step()
            assert worker._runner._world.state_hash() == headless._world.state_hash()
