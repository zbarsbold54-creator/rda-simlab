"""Decay probe driver (decay-v1).

Measures whether the RDA retention gate decays/retains memory sensibly.
Probes A/B/C/D as specified. CPU only, workers=4, OMP/MKL=1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LOG_ROOT = REPO / "logs" / "decay_probe"
TRAIN_ROOT = LOG_ROOT / "train"

HARD = ["rda_gru_surprise", "rda_gru_per_drive", "rda_tf_surprise", "rda_tf_per_drive"]
SOFT = ["rda_gru_surprise_soft", "rda_gru_per_drive_soft", "rda_tf_surprise_soft", "rda_tf_per_drive_soft"]
CONSTANTS = ["rda_gru_constant", "rda_tf_constant"]
CONTROLS = ["gru", "transformer"]
VARIANTS = HARD + SOFT + CONTROLS
RDA_VARIANTS = HARD + SOFT
GATED = HARD + SOFT
_POLICY_CACHE: dict[str, Any] = {}
EVAL_SEEDS = list(range(20))
TICKS = 512
BALANCED = "configs/balanced.toml"
TRAIN_STEPS = 40000
TRAIN_SEED = 100000
WORKERS = 4
BASELINE = 0.9
LAM_MAX = 0.9984375
M_SCALE = 4.0
NOISE_SEED = 12345
NOISE_STD = 0.01

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ProbeA:
    variant: str
    checkpoint: str
    lam_p5: float
    lam_p50: float
    lam_p95: float
    lam_std: float
    pct_low: float
    pct_high: float
    pct_m_in: float
    n_steps: int


@dataclass
class ProbeB:
    variant: str
    checkpoint: str
    mean_lam_after: float
    mean_lam_base: float
    diff: float
    spearman: float


@dataclass
class ProbeC:
    variant: str
    checkpoint: str
    peak_mean: float
    half_mean: float
    half_str: str
    introspect_exposes: bool


def _env() -> dict[str, str]:
    e = dict(os.environ)
    e["OMP_NUM_THREADS"] = "1"
    e["MKL_NUM_THREADS"] = "1"
    return e


def _git_commit() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, cwd=str(REPO),
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _find_existing(variant: str) -> Path | None:
    """Find a 10-iteration checkpoint (learning.csv with 10 rows)."""
    for csv in (REPO / "runs").rglob("learning.csv"):
        try:
            rows = [
                ln for ln in csv.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln and not ln.startswith("#") and not ln.startswith("iteration")
            ]
        except OSError:
            continue
        if len(rows) != 10:
            continue
        # exact match: variant must be a whole path component or prefix after NN_
        parts = csv.parts
        if not any(p == variant or p.startswith((f"00_{variant}_", f"01_{variant}_", f"02_{variant}_", f"03_{variant}_", f"04_{variant}_", f"05_{variant}_", f"06_{variant}_", f"07_{variant}_", f"08_{variant}_", f"09_{variant}_")) for p in parts):
            continue
        final = csv.parent / "final.pt"
        if final.exists():
            return final
    return None


def _train(variant: str) -> Path:
    import time as _time
    out = TRAIN_ROOT / variant
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final.pt"
    if final.exists():
        return final
    cmd = [
        sys.executable, "-m", "simlab.cli", "train",
        "--arch", variant,
        "--total-steps", str(TRAIN_STEPS),
        "--workers", str(WORKERS),
        "--seed", str(TRAIN_SEED),
        "--out", str(out),
    ]
    log.info("training %s -> %s", variant, out)
    start = _time.time()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=_env())
    last_print = 0
    csv = out / "learning.csv"
    while proc.poll() is None:
        _time.sleep(2)
        if csv.exists():
            try:
                lines = csv.read_text(encoding="utf-8", errors="replace").splitlines()
                steps_done = 0
                for ln in lines:
                    if ln and not ln.startswith("#") and not ln.startswith("iteration"):
                        try:
                            steps_done = int(ln.split(",")[1])
                        except Exception:
                            log.debug("parse learning.csv failed", exc_info=True)
                if steps_done - last_print >= 5000:
                    print(f"[{variant}] {steps_done}/{TRAIN_STEPS} steps elapsed={_time.time()-start:.1f}s", flush=True)
                    last_print = steps_done
            except Exception:
                log.debug("read learning.csv failed", exc_info=True)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {variant}")
    print(f"[{variant}] done elapsed={_time.time()-start:.1f}s", flush=True)
    return final


def _train_at_steps(variant: str, steps: int) -> Path:
    out = TRAIN_ROOT / f"{variant}_s{steps}"
    out.mkdir(parents=True, exist_ok=True)
    final = out / "final.pt"
    if final.exists():
        return final
    if steps == 0:
        from simlab.training.arch_spec import build_policy_cfg, default_preset_selection
        from simlab.training.torch_policy import make_policy
        sel = default_preset_selection(variant, seed=TRAIN_SEED)
        cfg = build_policy_cfg(sel)
        pol = make_policy(cfg)
        pol.save_checkpoint(final)
        return final
    cmd = [
        sys.executable, "-m", "simlab.cli", "train",
        "--arch", variant,
        "--total-steps", str(steps),
        "--workers", str(WORKERS),
        "--seed", str(TRAIN_SEED),
        "--out", str(out),
    ]
    log.info("training %s steps=%d -> %s", variant, steps, out)
    r = subprocess.run(cmd, check=False, cwd=str(REPO), env=_env())
    if r.returncode != 0:
        raise RuntimeError(f"training failed for {variant} steps={steps}")
    return final


def _can_reuse_soft(checkpoint: Path, variant: str) -> bool:
    try:
        from simlab.training.arch_spec import build_policy_cfg, default_preset_selection
        from simlab.training.torch_policy import make_policy
        sel = default_preset_selection(variant, seed=TRAIN_SEED)
        cfg = build_policy_cfg(sel)
        fresh = make_policy(cfg)
        loaded = make_policy({**cfg, "checkpoint": str(checkpoint)})
        if getattr(loaded, "_load_error", None):
            return False
        # sample parameter differs from fresh?
        import torch as _t
        sd_f = fresh._backbone._net.state_dict()
        sd_l = loaded._backbone._net.state_dict()
        for k in sd_f:
            if not _t.allclose(sd_f[k].cpu(), sd_l[k].cpu(), atol=1e-6):
                return True
        return False
    except Exception:
        log.debug("reuse check failed", exc_info=True)
        return False


def _get_checkpoint(variant: str) -> Path:
    fresh = TRAIN_ROOT / variant / "final.pt"
    if fresh.exists():
        if variant in SOFT:
            if _can_reuse_soft(fresh, variant):
                log.info("reuse soft checkpoint (loads and differs) for %s: %s", variant, fresh)
                return fresh
            log.info("soft checkpoint invalid for %s, retraining", variant)
            fresh.unlink(missing_ok=True)
        else:
            log.info("reuse fresh checkpoint for %s: %s", variant, fresh)
            return fresh
    if variant not in SOFT:
        existing = _find_existing(variant)
        if existing is not None and variant != "rda_gru_surprise":
            log.info("reuse existing 10-iter checkpoint for %s: %s", variant, existing)
            return existing
    log.info("training 40k steps for %s", variant)
    return _train(variant)


def _make_runner(checkpoint: Path, seed: int, variant: str):
    from dataclasses import replace

    from simlab.core.config import load_config
    from simlab.core.runner import Runner

    base = load_config(BALANCED)
    is_rda = variant.startswith("rda")
    drive = "rda8" if is_rda else "simple"
    cfg = replace(
        base,
        policy="learned_torch",
        drive=drive,
        agent=replace(base.agent, count=5),
        policy_checkpoint=str(checkpoint),
    )
    runner = Runner(cfg, seed=seed, agents=5)
    runner.setup()
    runner._ensure_components()
    pol = getattr(runner, "_policy", None)
    if pol is not None and getattr(pol, "_load_error", None):
        raise RuntimeError(f"checkpoint load failed for {variant}: {pol._load_error}")
    # cache policy weights per variant (load once, reuse across seeds)
    key = str(checkpoint)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = pol
    else:
        # reuse cached weights to avoid reload; keep runner's policy object but copy weights
        cached = _POLICY_CACHE[key]
        if cached is not pol:
            try:
                pol._backbone._net.load_state_dict(cached._backbone._net.state_dict())
                _ret = getattr(pol._backbone, "_retention", None)
                _cret = getattr(cached._backbone, "_retention", None)
                if _ret is not None and _cret is not None:
                    _ret.load_state_dict(_cret.state_dict())
            except Exception:
                log.debug("cache copy failed", exc_info=True)
    return runner


def _collect_ab(checkpoint: Path, variant: str) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[list[list[float]]]]:
    """Collect lam, m_norm, s, raw m, m_hat, u, per-episode per-channel lam."""
    lams: list[float] = []
    ms: list[float] = []
    ss: list[float] = []
    raw_ms: list[float] = []
    mhats: list[float] = []
    us: list[float] = []
    tick_lam: list[float] = []
    tick_s: list[float] = []
    per_episode: list[list[list[float]]] = []
    for seed in EVAL_SEEDS:
        ep_channels: dict[int, list[float]] = {}
        runner = _make_runner(checkpoint, seed, variant)
        policy = runner._policy
        world = runner._world
        drive_core = runner._drive_core
        # reset policy hidden
        policy.reset(5, np.random.default_rng(seed))
        for tick in range(TICKS):
            obs_batch = world._observation_batch()
            live = [aid for aid, st in world._agents.items() if st["is_alive"]]
            drives: dict[str, Any] = {}
            if drive_core is not None:
                live_obs = [obs_batch[aid] for aid in live]
                vals = drive_core.step(live_obs, {"rng": world._rng})
                drives = {aid: tuple(float(v) for v in vals[i]) for i, aid in enumerate(live)}
            actions: dict[str, Any] = {}
            tick_lams: list[float] = []
            tick_ss: list[float] = []
            tick_ms: list[float] = []
            for aid in live:
                obs = obs_batch[aid]
                drv = drives.get(aid, ())
                act = policy.act(obs, drv)
                actions[aid] = act
                # introspect
                bb = getattr(policy, "_backbone", None)
                if bb is not None and hasattr(bb, "introspect"):
                    try:
                        m = bb.introspect()
                    except Exception:  # noqa: BLE001
                        m = {}
                    if "lam" in m:
                        lam_arr = np.asarray(m["lam"], dtype=np.float64).reshape(-1)
                        lams.extend(lam_arr.tolist())
                        tick_lams.extend(lam_arr.tolist())
                        for ci, v in enumerate(lam_arr.tolist()):
                            ep_channels.setdefault(ci, []).append(float(v))
                    if "m" in m:
                        m_arr = np.asarray(m["m"], dtype=np.float64).reshape(-1)
                        mn = m_arr / M_SCALE
                        ms.extend(mn.tolist())
                        raw_ms.extend(m_arr.tolist())
                        tick_ms.extend(mn.tolist())
                    if "m_hat" in m:
                        mhats.extend(np.asarray(m["m_hat"], dtype=np.float64).reshape(-1).tolist())
                    if "u" in m:
                        us.extend(np.asarray(m["u"], dtype=np.float64).reshape(-1).tolist())
                    if "s" in m:
                        s_arr = np.asarray(m["s"], dtype=np.float64).reshape(-1)
                        ss.extend(s_arr.tolist())
                        tick_ss.extend(s_arr.tolist())
            if tick_lams:
                tick_lam.append(float(np.mean(tick_lams)))
            if tick_ss:
                tick_s.append(float(np.mean(tick_ss)))
            # step world
            runner._world.step(actions, include_observations=False, include_snapshot=False, include_agent_states=False)
            runner._step_count = runner._world.get_step_count()
            if runner._world._agents and all(not st["is_alive"] for st in runner._world._agents.values()):
                break
        per_episode.append([ep_channels.get(k, []) for k in sorted(ep_channels)])
    return lams, ms, ss, tick_lam, raw_ms, mhats, us, per_episode


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    # rank
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # handle ties by averaging (simple)
        return r
    rx = rank(xa)
    ry = rank(ya)
    # Pearson on ranks
    mx = rx.mean()
    my = ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    return float(num / den) if den > 0 else 0.0


def _probe_a(variant: str, checkpoint: Path, lams: list[float], ms: list[float]) -> ProbeA:
    arr = np.asarray(lams, dtype=np.float64) if lams else np.array([0.0])
    p5, p50, p95 = np.percentile(arr, [5, 50, 95])
    std = float(arr.std())
    # within 5% of bound value (relative)
    pct_low = float((np.abs(arr - BASELINE) / BASELINE <= 0.05).mean() * 100.0)
    pct_high = float((np.abs(arr - LAM_MAX) / LAM_MAX <= 0.05).mean() * 100.0)
    if ms:
        marr = np.asarray(ms, dtype=np.float64)
        pct_m = float(((marr >= 0.0) & (marr <= 1.0)).mean() * 100.0)
    else:
        pct_m = 0.0
    return ProbeA(variant, str(checkpoint), float(p5), float(p50), float(p95), std, pct_low, pct_high, pct_m, len(lams))


def _probe_b(variant: str, checkpoint: Path, tick_lam: list[float], tick_s: list[float]) -> ProbeB:
    if not tick_lam or not tick_s or len(tick_lam) != len(tick_s):
        return ProbeB(variant, str(checkpoint), 0.0, 0.0, 0.0, 0.0)
    s_arr = np.asarray(tick_s, dtype=np.float64)
    lam_arr = np.asarray(tick_lam, dtype=np.float64)
    thr = np.percentile(s_arr, 95)
    idx = np.where(s_arr >= thr)[0]
    after: list[float] = []
    for i in idx:
        for k in range(1, 11):
            if i + k < len(lam_arr):
                after.append(float(lam_arr[i + k]))
    mean_after = float(np.mean(after)) if after else 0.0
    mask = np.ones(len(lam_arr), dtype=bool)
    mask[idx] = False
    # also exclude 10 ticks after spikes from baseline? spec says all other steps
    mean_base = float(lam_arr[mask].mean()) if mask.any() else 0.0
    diff = abs(mean_after - mean_base)
    # spearman s_t vs lam_{t+1}
    if len(s_arr) > 1:
        sp = _spearman(s_arr[:-1].tolist(), lam_arr[1:].tolist())
    else:
        sp = 0.0
    return ProbeB(variant, str(checkpoint), mean_after, mean_base, diff, sp)


def _probe_d1(per_episode: list[list[list[float]]]) -> dict[str, Any]:
    # per_episode: list of episodes, each list of channels, each list of lam values
    if not per_episode:
        return {"within_std_mean": 0.0, "between_std_mean": 0.0, "per_channel": []}
    n_ch = max(len(ep) for ep in per_episode) if per_episode else 0
    within_means: list[float] = []
    between_means: list[float] = []
    per_ch: list[dict[str, float]] = []
    for ch in range(n_ch):
        ep_means: list[float] = []
        ep_stds: list[float] = []
        for ep in per_episode:
            if ch < len(ep) and ep[ch]:
                arr = np.asarray(ep[ch], dtype=np.float64)
                ep_means.append(float(arr.mean()))
                ep_stds.append(float(arr.std()))
        if ep_means:
            w = float(np.mean(ep_stds)) if ep_stds else 0.0
            b = float(np.std(ep_means)) if len(ep_means) > 1 else 0.0
            within_means.append(w)
            between_means.append(b)
            per_ch.append({"channel": ch, "within_std": w, "between_std": b, "mean": float(np.mean(ep_means))})
    return {
        "within_std_mean": float(np.mean(within_means)) if within_means else 0.0,
        "between_std_mean": float(np.mean(between_means)) if between_means else 0.0,
        "per_channel": per_ch,
    }


def _probe_d2(raw_ms: list[float], ms: list[float], ss: list[float], mhats: list[float], us: list[float], variant: str) -> dict[str, Any]:
    def pcts(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"min": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        a = np.asarray(vals, dtype=np.float64)
        return {"min": float(a.min()), "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    # gate pre-activation: hard = m/m_scale (before clamp), soft = soft_gate_k*m_hat (before sigmoid)
    if variant.endswith("_soft"):
        pre = [float(v) * 1.0 for v in mhats]  # soft_gate_k default 1.0
        clamp_pct = 0.0  # no clamp in soft path
    else:
        pre = ms  # m/m_scale is pre-activation for hard
        if ms:
            arr = np.asarray(ms, dtype=np.float64)
            clamp_pct = float(((arr < 0.0) | (arr > 1.0)).mean() * 100.0)
        else:
            clamp_pct = 0.0
    return {
        "raw_m": pcts(raw_ms),
        "norm_m": pcts(ms),
        "s": pcts(ss),
        "pre_act": pcts(pre),
        "m_hat": pcts(mhats),
        "u": pcts(us),
        "clamp_pct": clamp_pct,
    }


def _run_episode_states(checkpoint: Path, variant: str, seed: int, noisy: bool) -> list[np.ndarray]:
    """Return per-tick state arrays (E for RDA, hidden otherwise)."""
    runner = _make_runner(checkpoint, seed, variant)
    policy = runner._policy
    world = runner._world
    drive_core = runner._drive_core
    policy.reset(5, np.random.default_rng(seed))
    # noise vector
    rng = np.random.default_rng(NOISE_SEED)
    noise_full = rng.normal(0.0, NOISE_STD, size=256).astype(np.float32)
    states: list[np.ndarray] = []
    for tick in range(512):
        obs_batch = world._observation_batch()
        live = [aid for aid, st in world._agents.items() if st["is_alive"]]
        drives: dict[str, Any] = {}
        if drive_core is not None:
            live_obs = [obs_batch[aid] for aid in live]
            vals = drive_core.step(live_obs, {"rng": world._rng})
            drives = {aid: tuple(float(v) for v in vals[i]) for i, aid in enumerate(live)}
        actions: dict[str, Any] = {}
        # manual per-agent step to allow noise injection on encoded obs
        for aid in live:
            obs = obs_batch[aid]
            drv = drives.get(aid, ())
            # encode
            x = policy._encode(obs, drv)  # (1, D)
            drive_t = policy._drive_tensor(drv)
            if noisy and tick == 100:
                d = x.shape[1]
                noise = noise_full[:d]
                x = x + policy._t.from_numpy(noise).float().unsqueeze(0)
            # hidden index
            idx = policy._call_index % policy.n_agents
            h_prev = policy._hidden_t[idx:idx+1]
            with policy._t.no_grad():
                a, _v, f = policy._backbone.frame_step(x, h_prev, drive_t)
            turn = float(policy._t.tanh(a[0, 0]))
            thrust = float(policy._t.sigmoid(a[0, 1]))
            eat = bool(policy._t.sigmoid(a[0, 2]) >= 0.5)
            from simlab.core.types import Action
            actions[aid] = Action(turn, thrust, eat, "learned_torch")
            policy._hidden_t[idx] = f[0]
            policy._call_index += 1
        # record state after step
        bb = getattr(policy, "_backbone", None)
        if bb is not None and hasattr(bb, "introspect"):
            try:
                m = bb.introspect()
                if "E" in m:
                    states.append(np.asarray(m["E"], dtype=np.float64).reshape(-1).copy())
                else:
                    states.append(policy.get_state()["hidden"].reshape(-1).astype(np.float64).copy())
            except Exception:  # noqa: BLE001
                states.append(policy.get_state()["hidden"].reshape(-1).astype(np.float64).copy())
        else:
            states.append(policy.get_state()["hidden"].reshape(-1).astype(np.float64).copy())
        runner._world.step(actions, include_observations=False, include_snapshot=False, include_agent_states=False)
        runner._step_count = runner._world.get_step_count()
        if runner._world._agents and all(not st["is_alive"] for st in runner._world._agents.values()):
            break
    return states


def _probe_c(variant: str, checkpoint: Path) -> ProbeC:
    bb_exposes = variant.startswith("rda")
    peaks: list[float] = []
    halves: list[float] = []
    for seed in EVAL_SEEDS:
        clean = _run_episode_states(checkpoint, variant, seed, noisy=False)
        noisy = _run_episode_states(checkpoint, variant, seed, noisy=True)
        n = min(len(clean), len(noisy), 512)
        divs: list[float] = []
        for t in range(100, min(501, n)):
            if t < len(clean) and t < len(noisy):
                d = float(np.linalg.norm(clean[t] - noisy[t]))
                divs.append(d)
            else:
                divs.append(0.0)
        if not divs:
            continue
        peak = float(max(divs))
        peaks.append(peak)
        peak_idx = int(np.argmax(divs))
        half = 401.0
        for i in range(peak_idx + 1, len(divs)):
            if divs[i] < 0.5 * peak:
                half = float(i - peak_idx)
                break
        halves.append(half)
    peak_mean = float(np.mean(peaks)) if peaks else 0.0
    half_mean = float(np.mean(halves)) if halves else 0.0
    half_str = ">400" if half_mean > 400 else f"{half_mean:.1f}"
    return ProbeC(variant, str(checkpoint), peak_mean, half_mean, half_str, bb_exposes)


def _run_single(variant: str) -> dict[str, Any]:
    import time as _time
    t0 = _time.time()
    ckpt = _get_checkpoint(variant)
    out: dict[str, Any] = {"variant": variant, "checkpoint": str(ckpt), "probes": {}}
    if variant in RDA_VARIANTS:
        lams, ms, ss, tick_lam, raw_ms, mhats, us, per_ep = _collect_ab(ckpt, variant)
        print(f"[{variant}] A/B/D collected elapsed={_time.time()-t0:.1f}s", flush=True)
        pa = _probe_a(variant, ckpt, lams, ms)
        if ss and lams:
            s_tick = ss[:len(tick_lam)] if len(ss) >= len(tick_lam) else ss + [0.0]*(len(tick_lam)-len(ss))
            pb = _probe_b(variant, ckpt, tick_lam, s_tick)
        else:
            pb = ProbeB(variant, str(ckpt), 0.0, 0.0, 0.0, 0.0)
        d1 = _probe_d1(per_ep)
        d2 = _probe_d2(raw_ms, ms, ss, mhats, us, variant)
        out["probes"]["A"] = {"lam_p5": pa.lam_p5, "lam_p50": pa.lam_p50, "lam_p95": pa.lam_p95, "lam_std": pa.lam_std, "pct_low": pa.pct_low, "pct_high": pa.pct_high, "pct_m_in": pa.pct_m_in, "n_steps": pa.n_steps, "checkpoint": pa.checkpoint}
        out["probes"]["B"] = {"mean_lam_after": pb.mean_lam_after, "mean_lam_base": pb.mean_lam_base, "diff": pb.diff, "spearman": pb.spearman, "checkpoint": pb.checkpoint}
        out["probes"]["D1"] = d1
        out["probes"]["D2"] = d2
        print(f"[{variant}] A/B/D1/D2 done elapsed={_time.time()-t0:.1f}s", flush=True)
    pc = _probe_c(variant, ckpt)
    out["probes"]["C"] = {"peak_mean": pc.peak_mean, "half_mean": pc.half_mean, "half_str": pc.half_str, "introspect_exposes": pc.introspect_exposes, "checkpoint": pc.checkpoint}
    print(f"[{variant}] C done elapsed={_time.time()-t0:.1f}s", flush=True)
    out["wall_time_s"] = _time.time() - t0
    return out


def main() -> int:
    import time as _time
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.variant:
        res = _run_single(args.variant)
        per = LOG_ROOT / "per_variant"
        per.mkdir(parents=True, exist_ok=True)
        (per / f"{args.variant}.json").write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[{args.variant}] wrote per-variant json wall={res['wall_time_s']:.1f}s", flush=True)
        return 0
    # parent: run each variant in subprocess with 30 min timeout
    per = LOG_ROOT / "per_variant"
    per.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--variant", variant]
        print(f"[parent] starting {variant} subprocess", flush=True)
        t0 = _time.time()
        r = subprocess.run(cmd, check=False, cwd=str(REPO), env=_env(), timeout=1800)
        print(f"[parent] {variant} done rc={r.returncode} elapsed={_time.time()-t0:.1f}s", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"variant subprocess failed for {variant}")
    results: dict[str, Any] = {"commit": _git_commit(), "variants": {}, "probes": {}}
    for variant in VARIANTS:
        data = json.loads((per / f"{variant}.json").read_text(encoding="utf-8"))
        results["variants"][variant] = {"checkpoint": data["checkpoint"], "wall_time_s": data["wall_time_s"]}
        for k, v in data["probes"].items():
            results["probes"].setdefault(k, {})[variant] = v
    for v in CONSTANTS:
        results["variants"][v] = {"checkpoint": "no gate (n/a)"}
        for k in ("A", "B", "D1", "D2", "C"):
            results["probes"].setdefault(k, {})[v] = {"note": "no gate (n/a)"}
    # D3 in parent (uses _collect_ab directly)
    results["probes"]["D3"] = {}
    for steps in (0, 10000, 20000, 40000):
        ck = _train_at_steps("rda_gru_surprise", steps)
        lams, ms, _, _, _, _, _, _ = _collect_ab(ck, "rda_gru_surprise")
        pa = _probe_a("rda_gru_surprise", ck, lams, ms)
        results["probes"]["D3"][str(steps)] = {"pct_low": pa.pct_low, "pct_high": pa.pct_high, "total": pa.pct_low + pa.pct_high, "lam_p50": pa.lam_p50, "checkpoint": str(ck)}
    out = LOG_ROOT / "results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    # tables (hard vs soft)
    print("\nProbe A hard vs soft:", flush=True)
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["A"][h]
        b = results["probes"]["A"][s]
        print(f"{h} p5={a['lam_p5']:.4f} p50={a['lam_p50']:.4f} p95={a['lam_p95']:.4f} std={a['lam_std']:.4f} low={a['pct_low']:.1f}% high={a['pct_high']:.1f}% m_in={a['pct_m_in']:.1f}%", flush=True)
        print(f"{s} p5={b['lam_p5']:.4f} p50={b['lam_p50']:.4f} p95={b['lam_p95']:.4f} std={b['lam_std']:.4f} low={b['pct_low']:.1f}% high={b['pct_high']:.1f}% m_in={b['pct_m_in']:.1f}%", flush=True)
    for v in CONSTANTS:
        print(f"{v} no gate (n/a)", flush=True)
    print("\nProbe B hard vs soft:", flush=True)
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["B"][h]
        b = results["probes"]["B"][s]
        print(f"{h} after={a['mean_lam_after']:.4f} base={a['mean_lam_base']:.4f} diff={a['diff']:.4f} sp={a['spearman']:.3f}", flush=True)
        print(f"{s} after={b['mean_lam_after']:.4f} base={b['mean_lam_base']:.4f} diff={b['diff']:.4f} sp={b['spearman']:.3f}", flush=True)
    print("\nProbe C hard vs soft:", flush=True)
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["C"][h]
        b = results["probes"]["C"][s]
        print(f"{h} peak={a['peak_mean']:.4f} half={a['half_str']}", flush=True)
        print(f"{s} peak={b['peak_mean']:.4f} half={b['half_str']}", flush=True)
    for v in CONTROLS:
        c = results["probes"]["C"][v]
        print(f"{v} peak={c['peak_mean']:.4f} half={c['half_str']} introspect={c['introspect_exposes']}", flush=True)
    print("\nProbe D1 within/between std:", flush=True)
    for v in GATED:
        d = results["probes"]["D1"][v]
        print(f"{v} within={d['within_std_mean']:.4f} between={d['between_std_mean']:.4f}", flush=True)
    print("\nProbe D2 percentiles and clamp:", flush=True)
    for v in GATED:
        d = results["probes"]["D2"][v]
        print(f"{v} raw_m p50={d['raw_m']['p50']:.3f} norm_m p50={d['norm_m']['p50']:.3f} s p50={d['s']['p50']:.3f} pre p50={d['pre_act']['p50']:.3f} clamp={d['clamp_pct']:.1f}%", flush=True)
    print("\nProbe D3 saturation vs steps (rda_gru_surprise):", flush=True)
    for steps in ("0", "10000", "20000", "40000"):
        d = results["probes"]["D3"][steps]
        print(f"steps={steps} total_at_bound={d['total']:.1f}% p50={d['lam_p50']:.4f} ckpt={d['checkpoint']}", flush=True)
    print("\nPass/fail (C1 <50% at bound, C2 diff>=0.02 or |sp|>=0.1, C4 within_std>=0.02):", flush=True)
    for v in GATED:
        a = results["probes"]["A"][v]
        total = a["pct_low"] + a["pct_high"]
        c1 = total < 50.0
        b = results["probes"]["B"][v]
        c2 = b["diff"] >= 0.02 or abs(b["spearman"]) >= 0.1
        d1 = results["probes"]["D1"][v]
        c4 = d1["within_std_mean"] >= 0.02
        print(f"{v} C1={'PASS' if c1 else 'FAIL'} ({total:.1f}%) C2={'PASS' if c2 else 'FAIL'} (diff={b['diff']:.4f} sp={b['spearman']:.3f}) C4={'PASS' if c4 else 'FAIL'} (within={d1['within_std_mean']:.4f})", flush=True)
    for v in CONSTANTS:
        print(f"{v} no gate (n/a) for C1/C4", flush=True)
    return 0


def _main_old() -> int:
    # legacy in-process path (unused; kept for reference)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"commit": _git_commit(), "variants": {}, "probes": {}}
    for variant in VARIANTS:
        ckpt = _get_checkpoint(variant)
        results["variants"][variant] = {"checkpoint": str(ckpt)}
        log.info("variant %s checkpoint %s", variant, ckpt)
        if variant in RDA_VARIANTS:
            lams, ms, ss, tick_lam, raw_ms, mhats, us, per_ep = _collect_ab(ckpt, variant)
            pa = _probe_a(variant, ckpt, lams, ms)
            if ss and lams:
                s_tick = ss[:len(tick_lam)] if len(ss) >= len(tick_lam) else ss + [0.0]*(len(tick_lam)-len(ss))
                pb = _probe_b(variant, ckpt, tick_lam, s_tick)
            else:
                pb = ProbeB(variant, str(ckpt), 0.0, 0.0, 0.0, 0.0)
            d1 = _probe_d1(per_ep)
            d2 = _probe_d2(raw_ms, ms, ss, mhats, us, variant)
            results["probes"].setdefault("A", {})[variant] = {
                "lam_p5": pa.lam_p5, "lam_p50": pa.lam_p50, "lam_p95": pa.lam_p95,
                "lam_std": pa.lam_std, "pct_low": pa.pct_low, "pct_high": pa.pct_high,
                "pct_m_in": pa.pct_m_in, "n_steps": pa.n_steps, "checkpoint": pa.checkpoint,
            }
            results["probes"].setdefault("B", {})[variant] = {
                "mean_lam_after": pb.mean_lam_after, "mean_lam_base": pb.mean_lam_base,
                "diff": pb.diff, "spearman": pb.spearman, "checkpoint": pb.checkpoint,
            }
            results["probes"].setdefault("D1", {})[variant] = d1
            results["probes"].setdefault("D2", {})[variant] = d2
        pc = _probe_c(variant, ckpt)
        results["probes"].setdefault("C", {})[variant] = {
            "peak_mean": pc.peak_mean, "half_mean": pc.half_mean, "half_str": pc.half_str,
            "introspect_exposes": pc.introspect_exposes, "checkpoint": pc.checkpoint,
        }
    # D3: saturation vs training steps for rda_gru_surprise
    results["probes"]["D3"] = {}
    for steps in (0, 10000, 20000, 40000):
        ck = _train_at_steps("rda_gru_surprise", steps)
        lams, ms, _, _, _, _, _, _ = _collect_ab(ck, "rda_gru_surprise")
        pa = _probe_a("rda_gru_surprise", ck, lams, ms)
        results["probes"]["D3"][str(steps)] = {
            "pct_low": pa.pct_low, "pct_high": pa.pct_high, "total": pa.pct_low + pa.pct_high,
            "lam_p50": pa.lam_p50, "checkpoint": str(ck),
        }
    out = LOG_ROOT / "results.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    print("\nProbe A hard vs soft:")
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["A"][h]
        b = results["probes"]["A"][s]
        print(f"{h} p5={a['lam_p5']:.4f} p50={a['lam_p50']:.4f} p95={a['lam_p95']:.4f} std={a['lam_std']:.4f} low={a['pct_low']:.1f}% high={a['pct_high']:.1f}% m_in={a['pct_m_in']:.1f}%")
        print(f"{s} p5={b['lam_p5']:.4f} p50={b['lam_p50']:.4f} p95={b['lam_p95']:.4f} std={b['lam_std']:.4f} low={b['pct_low']:.1f}% high={b['pct_high']:.1f}% m_in={b['pct_m_in']:.1f}%")
    for v in CONSTANTS:
        a = results["probes"]["A"][v]
        print(f"{v} (no gate) p5={a['lam_p5']:.4f} p50={a['lam_p50']:.4f} p95={a['lam_p95']:.4f}")
    print("\nProbe B hard vs soft:")
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["B"][h]
        b = results["probes"]["B"][s]
        print(f"{h} after={a['mean_lam_after']:.4f} base={a['mean_lam_base']:.4f} diff={a['diff']:.4f} sp={a['spearman']:.3f}")
        print(f"{s} after={b['mean_lam_after']:.4f} base={b['mean_lam_base']:.4f} diff={b['diff']:.4f} sp={b['spearman']:.3f}")
    print("\nProbe C hard vs soft:")
    for h, s in zip(HARD, SOFT):
        a = results["probes"]["C"][h]
        b = results["probes"]["C"][s]
        print(f"{h} peak={a['peak_mean']:.4f} half={a['half_str']}")
        print(f"{s} peak={b['peak_mean']:.4f} half={b['half_str']}")
    for v in CONSTANTS + CONTROLS:
        c = results["probes"]["C"][v]
        print(f"{v} peak={c['peak_mean']:.4f} half={c['half_str']} introspect={c['introspect_exposes']}")
    print("\nProbe D1 within/between std:")
    for v in GATED:
        d = results["probes"]["D1"][v]
        print(f"{v} within={d['within_std_mean']:.4f} between={d['between_std_mean']:.4f}")
    print("\nProbe D2 percentiles and clamp:")
    for v in GATED:
        d = results["probes"]["D2"][v]
        print(f"{v} raw_m p50={d['raw_m']['p50']:.3f} norm_m p50={d['norm_m']['p50']:.3f} s p50={d['s']['p50']:.3f} pre p50={d['pre_act']['p50']:.3f} clamp={d['clamp_pct']:.1f}%")
    print("\nProbe D3 saturation vs steps (rda_gru_surprise):")
    for steps in ("0", "10000", "20000", "40000"):
        d = results["probes"]["D3"][steps]
        print(f"steps={steps} total_at_bound={d['total']:.1f}% p50={d['lam_p50']:.4f} ckpt={d['checkpoint']}")
    print("\nPass/fail (C1 <50% at bound, C2 diff>=0.02 or |sp|>=0.1, C4 within_std>=0.02):")
    for v in GATED:
        a = results["probes"]["A"][v]
        total = a["pct_low"] + a["pct_high"]
        c1 = total < 50.0
        b = results["probes"]["B"][v]
        c2 = b["diff"] >= 0.02 or abs(b["spearman"]) >= 0.1
        d1 = results["probes"]["D1"][v]
        c4 = d1["within_std_mean"] >= 0.02
        print(f"{v} C1={'PASS' if c1 else 'FAIL'} ({total:.1f}%) C2={'PASS' if c2 else 'FAIL'} (diff={b['diff']:.4f} sp={b['spearman']:.3f}) C4={'PASS' if c4 else 'FAIL'} (within={d1['within_std_mean']:.4f})")
    for v in CONSTANTS:
        print(f"{v} no gate (n/a) for C1/C4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
