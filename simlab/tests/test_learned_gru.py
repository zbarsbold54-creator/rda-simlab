"""Tests for the LearnedGRU plugin and torch isolation."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from simlab.core.config import load_config
from simlab.core.runner import Runner


def _cfg() -> object:
    return replace(load_config(), policy="learned_gru", drive="simple")


def test_importing_core_and_cli_does_not_load_torch() -> None:
    code = (
        "import sys; import simlab.core; import simlab.cli; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", out.stderr


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_learned_gru_scrub_back_replay_matches_when_hidden_restored() -> None:
    from simlab.plugins.learned_gru import LearnedGRU

    agents = 3
    n = 200
    k = 100
    runner = Runner(_cfg(), seed=77, agents=agents)
    runner.setup()
    runner._ensure_components()
    runner._policy = LearnedGRU(seed=77, config=None)
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


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_learned_gru_eval_is_deterministic(tmp_path: Path) -> None:
    from simlab.eval_lib import REPO, aggregate, run_episode_metrics
    from simlab.plugins.learned_gru import LearnedGRU

    ckpt = tmp_path / "p.pt"
    LearnedGRU(seed=0, config=None).save_checkpoint(ckpt)
    cfg_path = REPO / "configs" / "balanced.toml"

    def run() -> dict[str, object]:
        results = [
            run_episode_metrics(
                "learned_gru",
                seed,
                5,
                500,
                str(cfg_path),
                drive="simple",
                checkpoint=str(ckpt),
            )
            for seed in range(4)
        ]
        return aggregate(results, 1.0, 500)

    r1 = run()
    r2 = run()
    assert r1["hash"] == r2["hash"]
    assert r1["cap_pct"] == r2["cap_pct"]


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
def test_learned_gru_scrub_back_breaks_if_hidden_not_restored() -> None:
    from simlab.plugins.learned_gru import LearnedGRU

    agents = 3
    n = 200
    k = 60
    runner = Runner(_cfg(), seed=77, agents=agents)
    runner.setup()
    runner._ensure_components()
    runner._policy = LearnedGRU(seed=77, config=None)
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
    runner._policy.reset(agents, None)  # zeros, wrong
    runner._drive_core.set_state(saved_drive)

    replay_hashes = []
    for _ in range(k):
        runner.run_step()
        replay_hashes.append(runner._world.state_hash())

    assert replay_hashes != expected_hashes
