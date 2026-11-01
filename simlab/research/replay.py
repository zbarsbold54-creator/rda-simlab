"""Replay a learned policy to collect retention/gate traces for the inspector.

This module is torch-free at import time; torch is loaded lazily inside
:func:`collect_gate_trace`. It is used by the Research Cockpit "Gates" tab to
read per-step gate metrics (``d_t``, ``E_t``, ``lam_t``, ``m_t``, ``s_t``,
FiLM gains) directly from a trained checkpoint, and to feed the pure closed
helpers in :mod:`simlab.research.gates`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.config import load_config
from simlab.core.runner import Runner

MAX_COLLECT_STEPS = 4000


def _infer_arm_from_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Best-effort read of the training ``config.json`` next to a checkpoint."""
    cfg_path = checkpoint.parent / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    arch = data.get("architecture", {})
    return {
        "drive_core": arch.get("drive_core", "rda8"),
        "drive_config": arch.get("drive_config"),
        "obs_mode": arch.get("obs_mode", "obs_drives"),
    }


def collect_gate_trace(
    checkpoint: str | Path,
    *,
    config_path: str | None = None,
    cap: int = 512,
    seed: int = 0,
    agents: int = 1,
) -> dict[str, Any]:
    """Replay ``checkpoint`` and return per-step gate metrics.

    Returns a dict with arrays for ``d``, ``E``, ``lam``, ``m``, ``s``
    (when the retention mode exposes them), ``film_scale`` (weight tensor), and
    scalar ``rail`` / ``lam_max`` hints for the inspector. Lazy-imports torch.
    """
    from simlab.training.backbones import RDABackbone  # noqa: F401

    ckpt = Path(checkpoint)
    arm = _infer_arm_from_checkpoint(ckpt)
    drive_core = arm.get("drive_core", "rda8")
    drive_config = arm.get("drive_config")

    scenario = config_path or "configs/balanced.toml"
    base = load_config(scenario)
    # The observation mode is not a SimLabConfig field; it is carried by the
    # checkpoint's own arch config and applied inside TorchPolicy when the
    # checkpoint is loaded. Only the drive core (and its config) is injected
    # here because it shapes the d_t/E_t signals the trace reads.
    cfg = replace(
        base,
        policy="learned_torch",
        drive=drive_core,
        drive_config=drive_config,
        policy_checkpoint=str(ckpt),
        agent=replace(base.agent, count=agents),
    )
    runner = Runner(cfg, seed=seed, agents=agents)
    runner.setup()

    d: list = []
    e: list = []
    lam: list = []
    m: list = []
    s: list = []
    film_scale = None
    ticks = 0

    steps = min(cap, MAX_COLLECT_STEPS)
    for _ in range(steps):
        try:
            runner.run_step(
                include_observations=False,
                include_snapshot=False,
                include_agent_states=False,
            )
        except Exception:  # noqa: BLE001
            break
        ticks += 1
        policy = getattr(runner, "_policy", None)
        if policy is None:
            break
        backbone = getattr(policy, "_backbone", None)
        introspect = getattr(backbone, "introspect", None)
        metrics = introspect() if callable(introspect) else {}
        if "E" in metrics:
            e.append(metrics["E"])
        if "d" in metrics:
            d.append(metrics["d"])
        if "lam" in metrics:
            lam.append(metrics["lam"])
        if "m" in metrics:
            m.append(metrics["m"])
        if "s" in metrics:
            s.append(metrics["s"])
        if film_scale is None and "film_scale" in metrics:
            film_scale = metrics["film_scale"]

    out: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "steps": len(e),
        "ticks": ticks,
        "rail": 1.0,
        "d": np.asarray(d),
        "E": np.asarray(e),
        "lam": np.asarray(lam),
        "m": np.asarray(m),
        "s": np.asarray(s),
        "film_scale": film_scale,
    }
    lam_max = 0.9984375
    if lam:
        lam_max = float(np.max(np.asarray(lam, dtype=float)))
    out["lam_max"] = lam_max
    return out
