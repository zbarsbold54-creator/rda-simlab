"""RDA-specific tests.

Covers the STEP 4 contract: causality, gate timing, ``surprise_rand``
independence, ``constant`` E observation-independence, transformer
vectorized-vs-incremental equivalence, multiworker determinism, and the
param-count table. All CPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from simlab.training.encoding import encode_observation
from simlab.training.retention import Retention
from simlab.training.torch_policy import make_policy

REPO = Path(__file__).resolve().parents[1]

RDA_VARIANTS = (
    "rda",
    "rda_const",
    "rda_g1",
    "rda_g1_rand",
    "rda_g3",
    "rda_g3_tied",
)

N_DRIVES = 8
OBS_DRIVES_DIM = 91
HIDDEN = 64


class FakeObs:
    def __init__(self, rays: np.ndarray, energy=0.5, health=1.0, speed=0.5) -> None:
        self.rays = rays
        self.ray_types = np.zeros(len(rays), dtype=int)
        self.proprioception = (float(energy), float(health), float(speed))


def _make_obs(rng: np.random.Generator) -> FakeObs:
    rays = rng.uniform(0.0, 20.0, size=16).astype(np.float32)
    return FakeObs(rays, energy=float(rng.uniform(0.2, 1.0)), health=1.0)


def _obs_stream(rng: np.random.Generator, n: int, drives: np.ndarray) -> np.ndarray:
    outs = []
    for k in range(n):
        o = _make_obs(rng)
        d = tuple(float(x) for x in drives[k])
        outs.append(encode_observation(o, d))
    return np.stack(outs).astype(np.float32)


def _drive_stream(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, N_DRIVES)).astype(np.float32)


def _policy(bb: str) -> Any:
    return make_policy(
        {"backbone": bb, "fusion": "concat", "obs_mode": "obs_drives", "seed": 0}
    )


def _to_np(v) -> np.ndarray:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().copy()
    return np.asarray(v).copy()

def _run_incr(policy, seq: np.ndarray, t_stop: int | None = None):
    """Incremental frame_step; returns per-step metric snapshots."""
    t = torch
    B = 1
    h = policy._backbone.initial_state(B)
    cap = len(seq) if t_stop is None else t_stop
    # Fix the RNG so stochastic controls (surprise_rand) draw identical values
    # across the two compared runs; freeze running stats too.
    torch.manual_seed(12345)
    policy._backbone._retention.eval()
    snaps = []
    for k in range(cap):
        x = t.from_numpy(seq[k : k + 1]).float()
        with t.no_grad():
            a, v, h = policy.frame_step(x, h, None)
        snap = {"a": _to_np(a[0]), "v": float(v[0].item())}
        mm = policy._backbone._last_metrics
        for key in ("E", "lam", "m", "s", "film_scale"):
            if key in mm:
                snap[key] = _to_np(mm[key])
        snaps.append(snap)
    return snaps


# ------------------------------------------------------------------ causality
@pytest.mark.parametrize("bb", RDA_VARIANTS)
def test_causality_future_obs_do_not_affect_past(bb: str) -> None:
    rng = np.random.default_rng(101)
    drives = _drive_stream(rng, 8)
    seq_a = _obs_stream(rng, 8, drives)
    seq_b = seq_a.copy()
    rng2 = np.random.default_rng(999)
    seq_b[4:] = _obs_stream(rng2, 4, drives[4:])
    policy = _policy(bb)
    sa = _run_incr(policy, seq_a, t_stop=4)
    sb = _run_incr(policy, seq_b, t_stop=4)
    for k in range(4):
        np.testing.assert_array_equal(sa[k]["a"], sb[k]["a"])
        assert sa[k]["v"] == sb[k]["v"]
        for key in ("E", "lam", "m"):
            if key in sa[k] and key in sb[k]:
                np.testing.assert_allclose(sa[k][key], sb[k][key], atol=1e-6)


# ------------------------------------------------------------ gate timing
def test_gate_timing_injected_surprise_leaves_lam_t_unchanged_and_raises_next() -> None:
    ret = Retention(mode="surprise", n_drives=N_DRIVES, hidden=16)
    ret.eval()
    n, h = N_DRIVES, 16
    B = 1
    rs = ret.initial_state(B)
    d = torch.rand(B, n)
    # A "quiet" observation equal to the aux head's prediction of the zero
    # embedding, so the first step produces ~zero surprise.
    o_norm = ret._base.pred_o.bias.detach().clone().unsqueeze(0)
    o_surp = torch.full((B, h), 50.0)
    _, rs, met0 = ret.step(o_norm, d, rs)
    _, rs, met1 = ret.step(o_surp, d, rs)
    _, _, met2 = ret.step(o_norm, d, rs)
    # lam0 uses m_{-1}=0 -> baseline; lam1 uses m_0 (small) -> still baseline.
    assert abs(float(met0["lam"].item()) - ret.baseline) < 1e-6
    assert abs(float(met1["lam"].item()) - ret.baseline) < 1e-3
    # the surprise is detected at t=1
    assert float(met1["s"].item()) > 1.0
    # lam2 uses m_1 (large) -> rises above baseline
    assert float(met2["lam"].item()) > ret.baseline + 0.05


# ----------------------------------------------------- surprise_rand independence
def test_surprise_rand_gate_values_independent_of_data() -> None:
    ret = Retention(mode="surprise_rand", n_drives=N_DRIVES, hidden=16)
    ret.eval()
    n, h = N_DRIVES, 16
    T = 12
    torch.manual_seed(2024)

    # Pre-fill a nontrivial empirical marginal so sampling uses the histogram.
    with torch.no_grad():
        ret.hist_counts.fill_(1.0)

    def gate_seq(o_stream, d_stream):
        # Same torch RNG seed for both streams: demonstrates the gate values
        # depend only on the RNG, not on the observations/drives.
        torch.manual_seed(2024)
        rs = ret.initial_state(1)
        out = []
        for k in range(T):
            _, rs, met = ret.step(o_stream[k], d_stream[k], rs)
            out.append(_to_np(met["lam"]))
        return out

    o_a = torch.randn(T, 1, h)
    o_b = torch.randn(T, 1, h) * 100.0  # radically different observation stream
    d_a = torch.rand(T, 1, n)
    d_b = torch.rand(T, 1, n) * 7.0
    ga = gate_seq(o_a, d_a)
    gb = gate_seq(o_b, d_b)
    for k in range(T):
        np.testing.assert_array_equal(ga[k], gb[k])


# ------------------------------------------------- constant E independence
def test_constant_E_does_not_depend_on_observations() -> None:
    ret = Retention(mode="constant", n_drives=N_DRIVES, hidden=16)
    n, h = N_DRIVES, 16
    rs = ret.initial_state(1)
    E_a, _, _ = ret.step(torch.zeros(1, h), torch.rand(1, n), rs)
    E_b, _, _ = ret.step(torch.full((1, h), 99.0), torch.rand(1, n), rs)
    np.testing.assert_allclose(
        E_a.detach().numpy(), E_b.detach().numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        E_a.detach().numpy().reshape(-1),
        ret._base.e_const.detach().numpy(),
        atol=1e-7,
    )


# ------------------------------------------------- transformer vectorized equiv
def test_transformer_vectorized_equals_incremental() -> None:
    policy = _policy("transformer")
    bb = policy._backbone
    B, T = 2, 8
    rng = np.random.default_rng(5)
    drives = _drive_stream(rng, T)
    seq = _obs_stream(rng, T, drives)
    x = torch.from_numpy(seq).unsqueeze(0).expand(B, T, -1).float()
    done = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]]
    )
    state = bb.initial_state(B)
    logits_v, val_v, state_v = bb.forward(x, None, state, done)
    h = state
    logits_i, val_i = [], []
    for t in range(T):
        h = h * done[:, t : t + 1]
        a, v, h = bb.frame_step(x[:, t], h, None)
        logits_i.append(a)
        val_i.append(v)
    logits_i = torch.stack(logits_i, dim=1)
    val_i = torch.stack(val_i, dim=1)
    assert (logits_v - logits_i).abs().max().item() < 1e-5
    assert (val_v - val_i).abs().max().item() < 1e-5
    assert (state_v - h).abs().max().item() < 1e-5


# --------------------------------------------------------------- param counts
def test_param_count_table_print() -> None:
    counts: dict[str, int] = {}
    for bb in RDA_VARIANTS:
        counts[bb] = _policy(bb).param_count()
    # within 5% of the shared "rda" reference
    ref = counts["rda"]
    for bb, c in counts.items():
        assert abs(c - ref) / ref <= 0.05
    line = ", ".join(f"{bb}={c}" for bb, c in counts.items())
    print("PARAM_COUNTS " + line)


# ----------------------------------------------------- multiworker determinism
def _arch_toml(tmp: Path, bb: str) -> Path:
    p = tmp / f"{bb}.toml"
    p.write_text(
        "[architecture]\n"
        f'backbone = "{bb}"\n'
        'drive_core = "rda8"\n'
        'fusion = "concat"\n'
        'obs_mode = "obs_drives"\n'
        "seed = 7\n"
        "backbone_config = { hidden = 64 }\n",
        encoding="utf-8",
    )
    return p


@pytest.mark.parametrize("bb", RDA_VARIANTS)
def test_multiworker_rda_training_is_deterministic(bb: str, tmp_path: Path) -> None:
    total_steps = 2048  # 2 iterations x 4 workers x 256 env steps
    arch = _arch_toml(tmp_path, bb)
    outs = [tmp_path / f"{bb}_a", tmp_path / f"{bb}_b"]
    for out in outs:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "simlab.cli",
                "train",
                "--arch",
                str(arch),
                "--total-steps",
                str(total_steps),
                "--workers",
                "4",
                "--eval-every",
                "1000",
                "--out",
                str(out),
            ],
            cwd=str(REPO),
            check=True,
            capture_output=True,
            text=True,
        )
    a = torch.load(outs[0] / "final.pt", map_location="cpu", weights_only=False)["state_dict"]
    b = torch.load(outs[1] / "final.pt", map_location="cpu", weights_only=False)["state_dict"]
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"tensor diverged for {bb}: {k}"
