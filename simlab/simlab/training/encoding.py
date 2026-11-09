"""Fixed observation encoding for learned policies.

The learner is sensor-only (it never sees absolute positions or the step
count), mirroring the sensor heuristic. Per ray we keep the normalized
distance and a one-hot of the ray type channel; we then append the three
normalized proprioceptive values. Option B appends the 8 drive values.

This module is numpy-only; torch is never imported here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RAY_RANGE = 20.0
N_RAYS = 16
N_RAY_TYPES = 4
N_DRIVES = 8

# The environment already exposes normalized proprioception
# (energy/max_energy, health/100, speed/speed_factor).
N_PROPRIO = 3


def obs_dim(with_drives: bool = False) -> int:
    base = N_RAYS * (1 + N_RAY_TYPES) + N_PROPRIO
    return base + (N_DRIVES if with_drives else 0)


def encode_observation(obs: Any, drive_state: Any = None) -> np.ndarray:
    """Encode a single observation into a float32 vector."""
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    ray_types = getattr(obs, "ray_types", ())
    parts = []
    for dist, kind in zip(rays, ray_types):
        parts.append(float(dist) / RAY_RANGE)
        one_hot = np.zeros(N_RAY_TYPES, dtype=np.float32)
        if 1 <= kind <= N_RAY_TYPES:
            one_hot[int(kind) - 1] = 1.0
        parts.extend(one_hot)
    prop = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
    parts.append(float(prop[0]))
    parts.append(float(prop[1]))
    parts.append(float(prop[2]))
    arr = np.asarray(parts, dtype=np.float32)
    if drive_state is not None:
        drive = np.asarray(drive_state, dtype=np.float32)
        if drive.size < N_DRIVES:
            drive = np.concatenate(
                [drive, np.zeros(N_DRIVES - drive.size, dtype=np.float32)]
            )
        arr = np.concatenate([arr, drive[:N_DRIVES]])
    return arr


def encode_observations(obs_list: list[Any], drive_list: list[Any]) -> np.ndarray:
    """Encode a batch of observations (one per agent) into a (N, D) array."""
    encoded = [encode_observation(o, d) for o, d in zip(obs_list, drive_list)]
    if not encoded:
        return np.zeros((0, obs_dim(with_drives=True)), dtype=np.float32)
    return np.stack(encoded)
