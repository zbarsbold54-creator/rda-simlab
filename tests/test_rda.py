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


FILM_PRESETS = (
    "gru",
    "transformer",
    "rda_gru_fixed",
    "rda_tf_fixed",
)


def test_film_fusion_with_obs_drives_forward() -> None:
    """Film fusion + obs_drives does not crash (regression for rda-0.2:
    in_dim was clipped to 83 for non-concat fusion, so feeding the 91-wide
    obs_drives vector raised a matmul shape error during training)."""
    t = torch
    rng = np.random.default_rng(7)
    drives = _drive_stream(rng, 4)
    seq = _obs_stream(rng, 4, drives)  # (4, 91)
    assert seq.shape[1] == OBS_DRIVES_DIM
    for bb in FILM_PRESETS:
        pol = make_policy(
            {"backbone": bb, "fusion": "film", "obs_mode": "obs_drives", "seed": 0}
        )
        h = pol._backbone.initial_state(4)
        x = t.from_numpy(seq).float()
        with t.no_grad():
            a, v, h2 = pol.frame_step(x, h, None)
        assert a.shape == (4, 3), bb
        assert v.shape == (4,), bb
        assert h2.shape == h.shape, bb


@pytest.mark.parametrize("bb", ("rda_gru_fixed", "rda_tf_fixed"))
def test_film_rda_match_params_within_5_percent(bb: str) -> None:
    """RDA film presets are size-matchable within 5% of the transformer."""
    target = make_policy({"backbone": "transformer", "obs_mode": "obs_only"}).param_count()
    matched = make_policy(
        {"backbone": bb, "obs_mode": "obs_drives", "fusion": "film",
         "match_params": target}
    )
    count = matched.param_count()
    assert abs(count - target) / target <= 0.05


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


# --------------------------------------------------------- aux-loss parity
def test_aux_loss_parity_across_variants() -> None:
    """Aux loss is nonzero and equal in expectation across ALL retention modes."""
    modes = (
        "fixed",
        "constant",
        "surprise",
        "surprise_rand",
        "per_drive",
        "per_drive_tied",
    )
    n, h = N_DRIVES, 16
    B, T = 4, 6
    torch.manual_seed(0)

    rets = {m: Retention(mode=m, n_drives=n, hidden=h) for m in modes}
    # Share the aux-head weights across all variants so only the gate differs.
    canon = rets["fixed"]._base
    for m, r in rets.items():
        r._base.pred_o.load_state_dict(canon.pred_o.state_dict())
        r._base.pred_d.load_state_dict(canon.pred_d.state_dict())

    o = torch.randn(T, B, h)
    d = torch.rand(T, B, n)
    losses: dict[str, float] = {}
    for m, r in rets.items():
        r.eval()
        rs = r.initial_state(B)
        acc = 0.0
        with torch.no_grad():
            for k in range(T):
                _, rs, met = r.step(o[k], d[k], rs)
                acc += float(met["aux_loss"])
        losses[m] = acc / T

    for m, v in losses.items():
        assert v > 1e-6, f"{m} aux loss is zero"
    base = losses["fixed"]
    for m, v in losses.items():
        assert abs(v - base) < 1e-5, f"aux loss not equal across variants: {losses}"


# ------------------------------------------------- gru_legacy fixture identity
def test_gru_legacy_loads_fixture_identically() -> None:
    """The learner-v0 fixture loads into gru_legacy with identical weights."""
    fix = torch.load(
        REPO / "tests/fixtures/learner_v0_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    pol = make_policy(
        {"backbone": "gru_legacy", "fusion": "none", "obs_mode": "obs_only", "seed": 0}
    )
    pol._load_checkpoint(REPO / "tests/fixtures/learner_v0_final.pt")
    assert pol.param_count() == 9798
    sd = pol._backbone._net.state_dict()
    fsd = fix["state_dict"] if isinstance(fix, dict) and "state_dict" in fix else fix
    assert set(sd.keys()) == set(fsd.keys()), (
        set(sd.keys()) ^ set(fsd.keys())
    )
    for k in sd:
        assert torch.equal(
            sd[k].detach().reshape(-1).float(),
            torch.as_tensor(fsd[k]).reshape(-1).float(),
        ), k


# ------------------------------------------------- gru_legacy never size-matched
def test_gru_legacy_never_size_matched() -> None:
    """gru_legacy is fixed at 9798 params and must ignore any match_params."""
    pol = make_policy(
        {
            "backbone": "gru_legacy",
            "fusion": "none",
            "obs_mode": "obs_only",
            "seed": 0,
            "match_params": 74118,  # a foreign target must be ignored
        }
    )
    assert pol.param_count() == 9798


# ------------------------------------------------- fixture eval golden pin
def test_gru_legacy_fixture_eval_golden() -> None:
    """Pin the learner-v0 reference eval through gru_legacy.

    Balanced 5, dev seeds 0-19, cap 10000. Expected: 5.0% cap, median 3705,
    and the canonical results hash.
    """
    from simlab.eval_lib import aggregate, run_episode_metrics

    fix = REPO / "tests/fixtures/learner_v0_final.pt"
    cfg = REPO / "configs/balanced.toml"
    results = []
    for seed in range(20):
        results.append(
            run_episode_metrics(
                policy="learned_torch",  # registered policy name; loads the fixture
                seed=seed,
                agents=5,
                cap=10000,
                config_path=str(cfg),
                drive="simple",
                checkpoint=str(fix),
            )
        )
    agg = aggregate(results, wall_time=0.0, cap=10000)
    assert agg["cap_pct"] == pytest.approx(5.0, abs=1e-9), agg
    assert agg["median_survival"] == 3705, agg
    assert agg["hash"] == "3e73e6d3232b6285", agg
    assert agg["death_cause"] == {"starvation": 94, "hazard": 1, "truncation": 5}


# --------------------------------------------------------- forward parity (net)
def _gru_net_forward(net, x, h):
    f = torch.tanh(net.gru_ih(x) + net.gru_hh(h))
    return net.policy_head(f), net.value_head(f), f


def test_gru_legacy_forward_parity_with_learned_gru_plugin() -> None:
    """Same fixed obs stream + hidden -> bit-identical net outputs (both paths)."""
    from simlab.plugins.learned_gru import LearnedGRU

    fix = REPO / "tests/fixtures/learner_v0_final.pt"

    old = LearnedGRU()
    old._load_checkpoint(fix)

    pol = make_policy(
        {"backbone": "gru_legacy", "fusion": "none", "obs_mode": "obs_only", "seed": 0}
    )
    pol._load_checkpoint(fix)

    rng = np.random.default_rng(2024)
    x_seq = rng.uniform(-1, 1, size=(300, 83)).astype(np.float32)
    h_old = torch.zeros(1, 64)
    h_new = torch.zeros(1, 64)
    first_diff = None
    for k in range(300):
        x = torch.from_numpy(x_seq[k : k + 1]).float()
        a_old, v_old, f_old = _gru_net_forward(old._net, x, h_old)
        a_new, v_new, f_new = _gru_net_forward(pol._backbone._net, x, h_new)
        for name, l, r_ in (
            ("hidden", f_old, f_new),
            ("logits", a_old, a_new),
            ("value", v_old, v_new),
        ):
            md = (l - r_).abs().max().item()
            assert md == 0.0, f"{name} diverged at step {k}: {md}"
        h_old = f_old.detach()
        h_new = f_new.detach()
    assert first_diff is None
    print("GRU_PARITY maxdiff=0 over 300 steps")


# ------------------------------------------------------- golden retention tests
def _identity_ret(mode, n, h, **kw):
    r = Retention(mode=mode, n_drives=n, hidden=h, rho=0.5, **kw)
    with torch.no_grad():
        r._base.pred_o.weight.copy_(torch.eye(h))
        r._base.pred_o.bias.zero_()
        r._base.pred_d.weight.copy_(torch.eye(n))
        r._base.pred_d.bias.zero_()
    r.eval()
    return r


def test_golden_fixed_retention() -> None:
    r = _identity_ret("fixed", 2, 1)
    d = [[0.2, 0.4], [0.3, 0.5]]
    rs = r.initial_state(1)
    E = []
    lam = []
    with torch.no_grad():
        for k in range(2):
            _, rs, met = r.step(
                torch.zeros(1, 1), torch.tensor([d[k]]), rs
            )
            E.append(met["E"][0].tolist())
            lam.append(met["lam"][0].tolist())
    assert lam[0] == pytest.approx([0.9, 0.9], abs=1e-5)
    assert lam[1] == pytest.approx([0.9, 0.9], abs=1e-5)
    assert E[0] == pytest.approx([0.02, 0.04], abs=1e-5)
    assert E[1] == pytest.approx([0.048, 0.086], abs=1e-5)


def test_golden_constant_retention() -> None:
    r = _identity_ret("constant", 2, 1)
    with torch.no_grad():
        r._base.e_const.copy_(torch.tensor([0.5, 0.25]))
    rs = r.initial_state(1)
    with torch.no_grad():
        for k, dk in enumerate(([0.2, 0.4], [0.9, 0.1], [0.0, 0.0])):
            _, rs, met = r.step(
                torch.full((1, 1), float(k)), torch.tensor([dk]), rs
            )
            assert met["E"][0].tolist() == pytest.approx(
                [0.5, 0.25], abs=1e-12
            ), f"constant E depends on obs at step {k}"
            assert met["lam"][0].tolist() == pytest.approx(
                [0.0, 0.0], abs=1e-12
            )


def test_golden_surprise_gate_timing_and_E() -> None:
    # h=1, n=1, rho=0.5, baseline=0.9, max=0.9984375, m_scale=4.
    # Surprise injected at step 3 (o_emb=3 while prev_e=0).
    r = _identity_ret("surprise", 1, 1)
    o = [[0.0], [0.0], [3.0], [3.0]]
    d = [[0.2], [0.4], [0.6], [0.8]]
    rs = r.initial_state(1)
    got = []
    with torch.no_grad():
        for k in range(4):
            _, rs, met = r.step(
                torch.tensor(o[k]).unsqueeze(0),
                torch.tensor(d[k]).unsqueeze(0),
                rs,
            )
            got.append({key: float(met[key][0]) for key in ("E", "lam", "m", "s")})
    lam4 = 0.9 + (0.9984375 - 0.9) * (3.0 / 4.0)  # 0.973828125
    # lam at step 3 unchanged (baseline); m rises to ~3 so lam jumps at step 4.
    # The surprise is normalized by sqrt(1+EPS), so m is 3/sqrt(1+1e-6).
    assert got[2]["lam"] == pytest.approx(0.9, abs=1e-5)
    assert got[2]["m"] == pytest.approx(3.0, abs=1e-4)
    assert got[3]["lam"] == pytest.approx(lam4, abs=1e-5)
    # hand-computed E_t = lam*E_{t-1} + (1-lam)*d_t
    E1 = 0.1 * 0.2
    E2 = 0.9 * E1 + 0.1 * 0.4
    E3 = 0.9 * E2 + 0.1 * 0.6
    E4 = lam4 * E3 + (1.0 - lam4) * 0.8
    assert got[0]["E"] == pytest.approx(E1, abs=1e-5)
    assert got[1]["E"] == pytest.approx(E2, abs=1e-5)
    assert got[2]["E"] == pytest.approx(E3, abs=1e-5)
    assert got[3]["E"] == pytest.approx(E4, abs=1e-5)


def test_golden_per_drive_retention() -> None:
    r = _identity_ret("per_drive", 2, 1)
    d = [[0.2, 0.4], [0.3, 0.5]]
    rs = r.initial_state(1)
    got = []
    with torch.no_grad():
        for k in range(2):
            _, rs, met = r.step(torch.zeros(1, 1), torch.tensor([d[k]]), rs)
            got.append(
                {
                    "E": met["E"][0].tolist(),
                    "lam": met["lam"][0].tolist(),
                    "m": met["m"][0].tolist(),
                }
            )
    # step 1: prev_d=0 -> s=[0.2,0.4], m=[0.2,0.4], lam=baseline
    assert got[0]["lam"] == pytest.approx([0.9, 0.9], abs=1e-5)
    assert got[0]["m"] == pytest.approx([0.2, 0.4], abs=1e-5)
    assert got[0]["E"] == pytest.approx([0.02, 0.04], abs=1e-5)
    # step 2: s=[0.1,0.1]; m=0.5*[0.2,0.4]+[0.1,0.1]=[0.2,0.3].
    # Past-only rule: lam at step 2 uses m_prev (state m after step 1 =
    # [0.2,0.4]), NOT the newly computed m.
    m2 = [0.2, 0.3]
    m_prev = [0.2, 0.4]
    lam2 = [
        0.9 + 0.0984375 * (m_prev[0] / 4.0),
        0.9 + 0.0984375 * (m_prev[1] / 4.0),
    ]
    E2 = [
        lam2[0] * got[0]["E"][0] + (1 - lam2[0]) * d[1][0],
        lam2[1] * got[0]["E"][1] + (1 - lam2[1]) * d[1][1],
    ]
    assert got[1]["m"] == pytest.approx(m2, abs=1e-5)
    assert got[1]["lam"] == pytest.approx(lam2, abs=1e-5)
    assert got[1]["E"] == pytest.approx(E2, abs=1e-5)
    # per-drive gates differ because each drive's trace differs.
    assert got[1]["lam"][0] != got[1]["lam"][1]


def test_golden_per_drive_tied_retention() -> None:
    # Shared G1 gate sets a scalar lam broadcast to all drives; per-drive m/s
    # still tracked. Surprise at step 3 leaves lam at baseline, rises at step 4.
    r = _identity_ret("per_drive_tied", 2, 1)
    o = [[0.0], [0.0], [3.0], [3.0]]
    d = [[0.2, 0.4], [0.3, 0.5], [0.4, 0.6], [0.5, 0.7]]
    rs = r.initial_state(1)
    got = []
    with torch.no_grad():
        for k in range(4):
            _, rs, met = r.step(
                torch.tensor(o[k]).unsqueeze(0),
                torch.tensor([d[k]]),
                rs,
            )
            got.append(
                {
                    "E": met["E"][0].tolist(),
                    "lam": met["lam"][0].tolist(),
                    "m": met["m"][0].tolist(),
                    "m_g": float(met["m_g"][0]),
                }
            )
    lam4 = 0.9 + 0.0984375 * (3.0 / 4.0)
    assert got[2]["lam"] == pytest.approx([0.9, 0.9], abs=1e-5)  # unchanged at surprise
    assert got[3]["lam"] == pytest.approx([lam4, lam4], abs=1e-5)  # rises
    # tied lam is scalar-broadcast: both drives equal at every step
    for k in range(4):
        assert got[k]["lam"][0] == got[k]["lam"][1]
    # E recursion uses the broadcast lam and each drive's own d
    E1 = [0.1 * 0.2, 0.1 * 0.4]
    E2 = [0.9 * E1[0] + 0.1 * 0.3, 0.9 * E1[1] + 0.1 * 0.5]
    E3 = [0.9 * E2[0] + 0.1 * 0.4, 0.9 * E2[1] + 0.1 * 0.6]
    E4 = [lam4 * E3[0] + (1 - lam4) * 0.5, lam4 * E3[1] + (1 - lam4) * 0.7]
    assert got[0]["E"] == pytest.approx(E1, abs=1e-5)
    assert got[1]["E"] == pytest.approx(E2, abs=1e-5)
    assert got[2]["E"] == pytest.approx(E3, abs=1e-5)
    assert got[3]["E"] == pytest.approx(E4, abs=1e-5)


def test_golden_surprise_rand_retention() -> None:
    # lam sampled from the empirical histogram; pin bin indexing + reproducibility.
    r = _identity_ret("surprise_rand", 2, 2)
    with torch.no_grad():
        r.hist_counts.zero_()
        r.hist_counts[3] = 10.0
    o = torch.zeros(1, 2)
    d = torch.full((1, 2), 0.25)
    torch.manual_seed(7)
    rs = r.initial_state(1)
    with torch.no_grad():
        _, rs, met = r.step(o, d, rs)
    lam = met["lam"][0]
    lo = r.hist_edges[3].item()
    hi = r.hist_edges[4].item()
    for i in range(2):
        assert lo <= float(lam[i]) <= hi, "sampled lam outside its histogram bin"
    assert torch.equal(met["lam"][0][0], met["lam"][0][1])  # scalar broadcast
    # E must use the sampled lam
    expected_E = (1.0 - float(lam[0])) * 0.25
    np.testing.assert_allclose(met["E"][0].tolist(), [expected_E, expected_E], atol=1e-6)
    # reproducibility under the same seed
    torch.manual_seed(7)
    rs2 = r.initial_state(1)
    with torch.no_grad():
        _, _, met2 = r.step(torch.zeros(1, 2), torch.full((1, 2), 0.25), rs2)
    assert torch.equal(met2["lam"], met["lam"])


# ------------------------------------------------------- checkpoint safety
def _policy_from_checkpoint(path) -> Any:
    import types

    from simlab.training.torch_policy import TorchPolicy

    return TorchPolicy(seed=0, config=types.SimpleNamespace(policy_checkpoint=str(path)))


def _policy_frame_produces(pol) -> None:
    # A usable policy must reset and emit one action without raising.
    pol.reset(1)
    x = torch.zeros(1, pol._in_dim)
    h = pol._backbone.initial_state(1)
    with torch.no_grad():
        a, v, h2 = pol.frame_step(x, h, None)
    assert a is not None and v is not None and h2 is not None


def test_checkpoint_save_is_atomic(tmp_path: Path) -> None:
    pol = _policy("gru")
    path = tmp_path / "ckpt.pt"
    pol.save_checkpoint(path)
    assert path.exists()
    # No partial temp file remains after a successful save.
    assert not list(tmp_path.glob("*.tmp"))
    # Saving again overwrites atomically and stays loadable.
    pol.save_checkpoint(path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    loaded = _policy_from_checkpoint(path)
    assert loaded._load_error is None
    _policy_frame_produces(loaded)


@pytest.mark.parametrize("backbone", ["gru", "transformer"])
def test_non_rda_checkpoint_loads_via_runner(tmp_path: Path, backbone: str) -> None:
    """Regression: _load_checkpoint must not require retention_mode."""
    from dataclasses import replace

    from simlab.core.config import load_config
    from simlab.core.runner import Runner

    pol = make_policy({"backbone": backbone, "fusion": "concat", "obs_mode": "obs_only", "seed": 0})
    path = tmp_path / "ckpt.pt"
    pol.save_checkpoint(path)
    base = load_config("configs/balanced.toml")
    cfg = replace(base, policy="learned_torch", drive="simple", policy_checkpoint=str(path))
    runner = Runner(cfg, seed=0, agents=1)
    runner.setup()
    runner._ensure_components()
    assert runner._policy is not None
    assert getattr(runner._policy, "_load_error", None) is None


def test_truncated_checkpoint_does_not_crash_policy(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    path.write_bytes(b"PK\x03\x04 truncated-garbage-not-a-real-checkpoint")
    pol = _policy_from_checkpoint(path)
    # No exception: falls back to a fresh policy and records the load error.
    assert pol._load_error is not None
    _policy_frame_produces(pol)


def test_missing_checkpoint_falls_back_to_fresh_policy(tmp_path: Path) -> None:
    path = tmp_path / "missing.pt"
    pol = _policy_from_checkpoint(path)
    assert pol._load_error is not None
    _policy_frame_produces(pol)


@pytest.mark.parametrize(
    ("retention", "has_gate"),
    [("surprise", True), ("fixed", False), ("constant", False)],
)
def test_collect_gate_trace_reports_documented_metrics(
    tmp_path: Path, retention: str, has_gate: bool
) -> None:
    from simlab.research.replay import collect_gate_trace

    policy = make_policy(
        {
            "backbone": f"rda_gru_{retention}",
            "drive_core": "rda8",
            "fusion": "film",
            "obs_mode": "obs_drives",
            "seed": 0,
        }
    )
    checkpoint = tmp_path / f"rda_gru_{retention}.pt"
    policy.save_checkpoint(checkpoint)

    trace = collect_gate_trace(checkpoint, cap=8, seed=3, agents=1)

    assert trace["ticks"] == 8
    assert trace["steps"] == 8
    for key in ("d", "E"):
        assert trace[key].shape == (8, 1, N_DRIVES)
        assert np.isfinite(trace[key]).all()
    lam_width = 1 if has_gate else N_DRIVES
    assert trace["lam"].shape == (8, 1, lam_width)
    assert np.isfinite(trace["lam"]).all()
    assert np.all(trace["lam"] >= 0.0)
    assert np.all(trace["lam"] <= 0.9984375)
    if has_gate:
        for key in ("m", "s"):
            assert trace[key].shape == (8, 1, 1)
            assert np.isfinite(trace[key]).all()
    else:
        assert trace["m"].size == 0
        assert trace["s"].size == 0
        if retention == "constant":
            assert np.all(trace["lam"] == 0.0)
        else:
            assert np.allclose(trace["lam"], 0.9)
