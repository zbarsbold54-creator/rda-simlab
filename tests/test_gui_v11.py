"""gui-v1.1: policy banner fields, Apply switches policy, per-agent info.

Runs offscreen where Qt is required; the Apply-switches test uses the core
Runner directly (no Qt) and therefore works in a plain pytest process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace

from simlab.core.config import load_config
from simlab.core.pipeline import compute_actions
from simlab.core.runner import Runner
from simlab.training.arch_spec import build_policy_cfg
from simlab.training.torch_policy import make_policy

STEPS = 200
SEED = 0
REPO = Path(__file__).resolve().parents[1]
HARD = str(REPO / "configs" / "hard.toml")


def _selection(backbone: str, fusion: str, obs_mode: str, seed: int) -> dict[str, Any]:
    return {
        "backbone": backbone,
        "drive_core": "rda8",
        "fusion": fusion,
        "obs_mode": obs_mode,
        "seed": seed,
        "backbone_config": {"hidden": 64} if backbone == "gru_legacy" else {},
        "drive_config": {},
    }


POLICIES: dict[str, tuple[dict[str, Any] | None, int]] = {
    # name -> (selection, seed) ; heuristic has no selection
    "heuristic": (None, 0),
    "gru_legacy_init1": (_selection("gru_legacy", "none", "obs_only", 1), 1),
    "gru_legacy_init2": (_selection("gru_legacy", "none", "obs_only", 2), 2),
    "transformer": (_selection("transformer", "concat", "obs_only", 0), 0),
    "rda_gru_surprise": (
        _selection("rda_gru_surprise", "film", "obs_drives", 0),
        0,
    ),
}


def _make_runner(name: str) -> Runner:
    selection = POLICIES[name][0]
    if name == "heuristic":
        cfg = replace(
            load_config(HARD),
            policy="heuristic",
            drive="rda8",
            drive_config=None,
            policy_checkpoint=None,
        )
    else:
        cfg = replace(
            load_config(HARD),
            policy="learned_torch",
            drive="rda8",
            drive_config=None,
            policy_checkpoint=None,
            policy_cfg=build_policy_cfg(selection),
        )
    runner = Runner(cfg, seed=SEED, agents=1)
    runner.setup()
    runner._ensure_components()  # builds the drive core
    if name != "heuristic":
        # Build the net directly from the selection so its init seed is used.
        selection = POLICIES[name][0]
        runner._policy = make_policy(build_policy_cfg(selection))
    return runner


def _collect_path(name: str) -> dict[int, tuple[float, float, bool]]:
    """Return {step: action_tuple} for the single agent while alive."""
    runner = _make_runner(name)
    world = runner._world
    path: dict[int, tuple[float, float, bool]] = {}
    for step in range(STEPS):
        policy = runner._policy
        actions, _drives, _obs = compute_actions(
            world, policy, runner._drive_core, step, batched=True
        )
        runner.run_step(
            actions,
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        for agent_id in actions:
            a = actions[agent_id]
            path[step] = (a.turn, a.thrust, bool(a.eat))
    return path


def test_apply_switches_policy_actions_differ() -> None:
    """Same Hard world/seed/ticks: action paths differ across policies."""
    paths = {name: _collect_path(name) for name in POLICIES}
    names = list(POLICIES)
    print("\n=== actions-differ table (fraction of ticks where actions differ) ===")
    header = f"{'policy':<20}" + "".join(f"{n:>20}" for n in names)
    print(header)
    table: dict[tuple[str, str], float] = {}
    for a in names:
        row = []
        for b in names:
            common = sorted(set(paths[a]) & set(paths[b]))
            if not common:
                row.append(float("nan"))
                continue
            diff = sum(paths[a][s] != paths[b][s] for s in common)
            fraction = diff / len(common)
            table[(a, b)] = fraction
            row.append(fraction)
        print(f"{a:<20}" + "".join(f"{v:>20.3f}" for v in row))

    # Guarantees demanded by the task.
    for name in names[1:]:
        assert table[("heuristic", name)] > 0.0, f"heuristic vs {name} identical"
    assert table[("gru_legacy_init1", "gru_legacy_init2")] > 0.0, (
        "gru_legacy different init seeds produce identical actions"
    )


def test_one_agent_hard_seed0_playout() -> None:
    """Report death step/cause/food for heuristic vs untrained gru_legacy."""
    for name in ("heuristic", "gru_legacy_init1"):
        runner = _make_runner(name)
        world = runner._world
        death_step = None
        cause = None
        food = None
        for step in range(STEPS):
            policy = runner._policy
            actions, _drives, _obs = compute_actions(
                world, policy, runner._drive_core, step, batched=True
            )
            runner.run_step(
                actions,
                include_observations=False,
                include_snapshot=False,
                include_agent_states=False,
            )
            for st in world._agents.values():
                food = int(st["food_eaten"])
                if death_step is None and not st["is_alive"]:
                    death_step = step + 1
                    cause = "starvation" if st["energy"] <= 0.0 else "hazard"
        print(f"{name}: death_step={death_step} cause={cause} food_eaten={food}")


def test_policy_banner_fields() -> None:
    """TorchPolicy.metadata exposes every banner field."""
    for name in ("gru_legacy_init1", "transformer", "rda_gru_surprise"):
        selection = POLICIES[name][0]
        policy = make_policy(build_policy_cfg(selection))
        meta = policy.metadata()
        for key in (
            "backbone",
            "fusion",
            "obs_mode",
            "param_count",
            "arch_hash",
            "trained",
            "checkpoint_path",
            "train_step",
        ):
            assert key in meta, f"{name}: missing {key}"
        assert meta["trained"] is False
        assert meta["param_count"] > 0
        assert meta["arch_hash"]


def test_checkpoint_metadata_trained_flags(tmp_path: Path) -> None:
    """A saved/loaded checkpoint flips trained flags and stores step."""
    selection = POLICIES["gru_legacy_init1"][0]
    policy = make_policy(build_policy_cfg(selection))
    ckpt = tmp_path / "run.pt"
    policy.save_checkpoint(ckpt, step=1234)
    reloaded = make_policy(build_policy_cfg(selection))
    reloaded._load_checkpoint(ckpt)
    assert reloaded.metadata()["trained"] is True
    assert reloaded.metadata()["train_step"] == 1234
    assert Path(reloaded.metadata()["checkpoint_path"]).name == "run.pt"


def test_arch_select_from_ckpt_restores_config() -> None:
    """The Architecture dock rebuilds a selection from a stored arch config."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from simlab.gui.architecture import ArchitectureWidget

    app = QApplication.instance() or QApplication([])
    widget = ArchitectureWidget(load_config())
    # A stored arch config (as written by TorchPolicy.config_dict()).
    arch = {
        "backbone": "rda_gru_surprise",
        "fusion": "film",
        "obs_mode": "obs_drives",
        "kwargs": {"hidden": 64},
    }
    selection = widget._selection_from_arch(arch, with_drives=True)
    assert selection["backbone"] == "rda_gru_surprise"
    assert selection["fusion"] == "film"
    assert selection["obs_mode"] == "obs_drives"
    assert selection["drive_core"] in ("rda8", "none")
    widget.close()
    app.processEvents()
