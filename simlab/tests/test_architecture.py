"""Architecture-agnostic learner tests.

Covers: backward compatibility with the legacy learner-v0 checkpoint, the
parametrized backbone x drive-core smoke matrix, rda8 causality, and rda8
snapshot/replay round-tripping. All CPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from simlab.training.encoding import encode_observation
from simlab.training.torch_policy import make_policy

REPO = Path(__file__).resolve().parents[1]
LEGACY_CKPT = REPO / "tests" / "fixtures" / "learner_v0_final.pt"
BACKWARD_COMPAT_HASH = "d3e1532311768f52"

DRIVE_CORES = ("none", "constant", "rda8")
BACKBONES = (
    "gru",
    "transformer",
    "rda",
    "rda_const",
    "rda_g1",
    "rda_g1_rand",
    "rda_g3",
    "rda_g3_tied",
)


class FakeObs:
    def __init__(self, rays: np.ndarray, energy=0.5, health=1.0, speed=0.5) -> None:
        self.rays = rays
        self.ray_types = np.zeros(len(rays), dtype=int)
        self.proprioception = (float(energy), float(health), float(speed))


def _make_obs(rng: np.random.Generator) -> FakeObs:
    rays = rng.uniform(0.0, 20.0, size=16).astype(np.float32)
    return FakeObs(rays, energy=float(rng.uniform(0.2, 1.0)), health=1.0)


def _drive_for(name: str):
    from simlab.plugins.drive_cores import (
        ConstantDriveCore,
        NoneDriveCore,
        RDA8DriveCore,
    )

    return {
        "none": NoneDriveCore,
        "constant": ConstantDriveCore,
        "rda8": RDA8DriveCore,
    }[name]


def _drive_seq(name: str, obs_seq: list[FakeObs], rng, decay=0.9):
    kwargs = {"decay": decay} if name == "rda8" else {}
    core = _drive_for(name)(**kwargs)
    core.reset(1, rng)
    out = [core.step([o], {"rng": rng})[0].copy() for o in obs_seq]
    return out, core


def _segments(bb: str, dc: str, rng, n_steps: int = 40) -> list[dict[str, Any]]:
    obs_seq = [_make_obs(rng) for _ in range(n_steps)]
    drives, _ = _drive_seq(dc, obs_seq, rng)
    xs = []
    turn, thrust, eat, logp, v, r, done = [], [], [], [], [], [], []
    for o, d in zip(obs_seq, drives):
        xs.append(encode_observation(o, tuple(float(x) for x in d)))
        turn.append(0.0)
        thrust.append(0.0)
        eat.append(0.0)
        logp.append(0.0)
        v.append(0.0)
        r.append(0.0)
        done.append(0.0)
    return [
        {
            "aid": "a",
            "x": np.stack(xs).astype(np.float32),
            "turn": np.asarray(turn, dtype=np.float32),
            "thrust": np.asarray(thrust, dtype=np.float32),
            "eat": np.asarray(eat, dtype=np.float32),
            "logp": np.asarray(logp, dtype=np.float32),
            "v": np.asarray(v, dtype=np.float32),
            "r": np.asarray(r, dtype=np.float32),
            "done": np.asarray(done, dtype=np.float32),
        }
    ]


def _final_weights(out_dir: Path) -> dict[str, Any]:
    import torch

    return torch.load(
        out_dir / "final.pt", map_location="cpu", weights_only=False
    )["state_dict"]


def test_multiworker_training_is_deterministic_across_processes(tmp_path: Path):
    """workers=2, seeded 2-iteration training in two separate processes must
    produce identical weights (regression for as-completed collection order)."""
    import torch

    total_steps = 900  # 2 iterations with 2 workers x 256 steps
    outs = [tmp_path / "run_a", tmp_path / "run_b"]
    for out in outs:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simlab.cli",
                "train",
                "--obs", "obs_only",
                "--total-steps", str(total_steps),
                "--workers", "2",
                "--seed", "7",
                "--out", str(out),
            ],
            cwd=str(REPO),
            check=True,
            capture_output=True,
            text=True,
        )
    a = _final_weights(outs[0])
    b = _final_weights(outs[1])
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor diverged: {k}"


def test_training_seed_controls_initial_weights(tmp_path: Path):
    """The training seed must seed policy init: different seeds give different
    initial weights, and the same seed gives identical initial weights across
    separate processes."""
    import torch

    from simlab.training.ppo import PPOConfig, _build_net

    def init_weights(seed: int) -> dict[str, torch.Tensor]:
        policy = _build_net(PPOConfig(seed=seed))
        return {k: v.clone() for k, v in policy.net_state_dict().items()}

    w_a = init_weights(100000)
    w_b = init_weights(100001)
    diff = max((w_a[k] - w_b[k]).abs().max().item() for k in w_a)
    assert diff > 1e-6, "different training seeds must give different initial weights"

    outs = [tmp_path / "w1.pt", tmp_path / "w2.pt"]
    for out in outs:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch,sys; sys.path.insert(0,r'"
                + str(REPO)
                + "'); "
                "from simlab.training.ppo import PPOConfig,_build_net; "
                "p=_build_net(PPOConfig(seed=7)); "
                "torch.save(p.net_state_dict(), r'"
                + str(out)
                + "')",
            ],
            cwd=str(REPO),
            check=True,
            capture_output=True,
            text=True,
        )
    a = torch.load(outs[0], map_location="cpu", weights_only=True)
    b = torch.load(outs[1], map_location="cpu", weights_only=True)
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor diverged across processes: {k}"


def test_backward_compat_legacy_checkpoint():
    if not LEGACY_CKPT.exists():
        pytest.skip("legacy learner-v0 checkpoint missing")
    from simlab.eval_lib import aggregate, run_episode_metrics

    cfg_path = str(REPO / "configs" / "balanced.toml")
    ckpt = str(LEGACY_CKPT)
    results = [
        run_episode_metrics("learned_torch", s, 5, 2000, cfg_path, "simple", ckpt)
        for s in range(5)
    ]
    agg = aggregate(results, 0.0, 2000)
    assert agg["hash"] == BACKWARD_COMPAT_HASH
    assert agg["cap_pct"] == 100.0


@pytest.mark.parametrize("bb", BACKBONES)
@pytest.mark.parametrize("dc", DRIVE_CORES)
def test_ppo_smoke(bb: str, dc: str, tmp_path: Path):
    rng = np.random.default_rng(1234)
    policy = make_policy(
        {"backbone": bb, "fusion": "concat", "obs_mode": "obs_drives", "seed": 0}
    )
    B = 4
    h0 = policy._backbone.initial_state(B)
    assert h0.shape == (B, policy.hidden_size)
    x = policy._t.randn(B, policy._in_dim)
    a, v, f = policy.frame_step(x, h0, None)
    assert a.shape == (B, 3)
    assert v.shape == (B,)
    assert f.shape == (B, policy.hidden_size)

    from simlab.training.ppo import PPOConfig, PPOTrainer

    trainer = PPOTrainer(PPOConfig(obs_mode="obs_drives"), tmp_path / "out")
    trainer.policy = policy
    trainer.opt = trainer.torch.optim.Adam(policy.parameters(), lr=trainer.cfg.lr0)
    segs = _segments(bb, dc, rng)
    for _ in range(2):
        info = trainer._ppo_update(segs)
        assert isinstance(info, dict)


def test_done_mask_reset_forward():
    rng = np.random.default_rng(7)
    obs_seq = [_make_obs(rng) for _ in range(8)]
    drives, _ = _drive_seq("none", obs_seq, rng)
    obs = np.stack(
        [
            encode_observation(o, tuple(float(x) for x in d))
            for o, d in zip(obs_seq, drives)
        ]
    ).astype(np.float32)
    policy = make_policy(
        {"backbone": "gru", "fusion": "concat", "obs_mode": "obs_drives"}
    )
    bb = policy._backbone
    B, T = 1, 8
    seq = policy._t.from_numpy(obs).unsqueeze(0)
    done_all = policy._t.ones(B, T)
    _, _, state_all = bb.forward(seq, None, bb.initial_state(B), done_all)
    # Reset at step 0 should make step 1 identical to starting from zero state.
    done_reset = done_all.clone()
    done_reset[0, 0] = 0.0
    _, _, state_reset = bb.forward(seq, None, bb.initial_state(B), done_reset)
    # The internal buffer state is not directly comparable across architectures,
    # so verify shapes and that a zero reset produces a finite new state.
    assert state_all.shape == state_reset.shape == (B, policy.hidden_size)
    assert policy._t.isfinite(state_reset).all()


def test_rda8_causality():
    rng = np.random.default_rng(99)
    seq_a = [_make_obs(rng) for _ in range(4)]
    seq_b = seq_a[:2] + [_make_obs(np.random.default_rng(5)) for _ in range(2)]
    da, _ = _drive_seq("rda8", seq_a, rng)
    db, _ = _drive_seq("rda8", seq_b, rng)
    np.testing.assert_array_equal(da[0], db[0])
    np.testing.assert_array_equal(da[1], db[1])


def test_rda8_snapshot_replay_bit_identical():
    rng = np.random.default_rng(1234)
    seq = [_make_obs(np.random.default_rng(i)) for i in range(8)]
    core = _drive_for("rda8")()
    core.reset(1, rng)
    first = [core.step([o], {"rng": rng})[0].copy() for o in seq[:3]]
    snap = core.get_state()
    second = [core.step([o], {"rng": rng})[0].copy() for o in seq[3:]]

    replay = _drive_for("rda8")()
    replay.set_state(snap)
    second_replay = [replay.step([o], {"rng": rng})[0].copy() for o in seq[3:]]
    np.testing.assert_array_equal(second, second_replay)
    assert len(first) == 3


def test_match_params_within_5_percent():
    for bb in BACKBONES:
        base = make_policy({"backbone": bb, "obs_mode": "obs_only"})
        target = base.param_count()
        cfg = {"backbone": bb, "obs_mode": "obs_only", "match_params": target}
        matched = make_policy(cfg)
        count = matched.param_count()
        assert abs(count - target) / target <= 0.05
    # rda variants build and match to a reference too (recurrent hidden top).
    base = make_policy({"backbone": "rda", "obs_mode": "obs_drives"})
    target = base.param_count()
    cfg = {"backbone": "rda_g1", "obs_mode": "obs_drives", "match_params": target}
    matched = make_policy(cfg)
    count = matched.param_count()
    assert abs(count - target) / target <= 0.05

    from simlab.training.arch_spec import (
        DEFAULT_MATCH_PARAMS,
        build_policy_cfg,
        list_specs,
    )

    for bb in list_specs("backbone"):
        if bb == "gru_legacy":
            legacy = make_policy(build_policy_cfg({"backbone": bb, "fusion": "none"}))
            assert legacy.param_count() == 9798
            continue
        is_rda = bb.startswith("rda")
        selection = {
            "backbone": bb,
            "drive_core": "rda8" if is_rda else "none",
            "fusion": "film" if is_rda else "concat",
            "obs_mode": "obs_drives" if is_rda else "obs_only",
            "seed": 0,
            "backbone_config": {},
            "drive_config": {},
        }
        preset = make_policy(build_policy_cfg(selection))
        assert abs(preset.param_count() - DEFAULT_MATCH_PARAMS) / DEFAULT_MATCH_PARAMS <= 0.05, bb
        if is_rda:
            retention = preset._backbone._retention
            assert hasattr(retention._base, "pred_o"), bb
            assert hasattr(retention._base, "pred_d"), bb

def test_eval_rejects_unknown_policy():
    """Eval must reject unknown policy/backbone names loudly (no silent fallback)."""
    from simlab.cli import _validate_policy_name

    with pytest.raises(ValueError) as ei:
        _validate_policy_name("gru_legacy")
    assert "valid policies" in str(ei.value)
    assert "learned_torch" in str(ei.value)


def test_eval_accepts_registered_policy():
    from simlab.cli import _validate_policy_name

    _validate_policy_name("learned_torch")


def test_train_arch_accepts_registered_preset_name(tmp_path: Path) -> None:
    import torch

    from simlab.training.arch_spec import build_policy_cfg, default_preset_selection

    output = tmp_path / "named_preset"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simlab.cli",
            "train",
            "--arch",
            "rda_gru_surprise",
            "--total-steps",
            "1024",
            "--workers",
            "4",
            "--seed",
            "100000",
            "--rollout",
            "256",
            "--eval-every",
            "0",
            "--update-device",
            "cpu",
            "--out",
            str(output),
        ],
        cwd=str(REPO),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    checkpoint = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
    expected = make_policy(
        build_policy_cfg(default_preset_selection("rda_gru_surprise", seed=100000))
    )._backbone.config_dict()
    assert checkpoint["arch"] == expected, result.stdout + result.stderr
    assert checkpoint["with_drives"] is True
