"""The eight RDA drive definitions, in fixed order.

This is the single source of truth for the sim-side drive signals ``d_t``.
Each drive is a deterministic numpy function of the current observation (and,
for derived quantities, the previous raw ray vector). All information used is
available at the current step ``t`` (causal).

The ``label`` fields are PLACEHOLDER strings the user edits. ``docstring`` is
the semantic description. ``compute(obs, last_rays) -> float`` returns the
instantaneous drive value in ``[0, 1]``.

The world-side drive core (``simlab.plugins.drive_cores.InstantDriveCore``)
calls :func:`raw_drive_vector`; the learner-side Retention reads these same
values from the trailing 8 dims of the ``obs_drives`` encoding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

DRIVE_NAMES = (
    "novelty",
    "uncertainty",
    "threat",
    "integrity",
    "agency",
    "learning_progress",
    "arousal",
    "homeostatic_deviation",
)
N_DRIVES = len(DRIVE_NAMES)

_RAY_RANGE = 20.0
_HAZARD_KIND = 2  # ray type index for hazards (food/agent etc. are 1/3/4)


def _hazard_distance(rays: np.ndarray, ray_types: np.ndarray) -> float:
    """Distance to the nearest hazard ray, or ray range if none."""
    best = _RAY_RANGE
    for ray, kind in zip(rays, ray_types):
        if int(kind) == _HAZARD_KIND:
            best = min(best, float(ray))
    return best


def _ray_diff(rays: np.ndarray, last_rays: np.ndarray) -> float:
    """Mean absolute ray change versus the previous observation (0..1)."""
    if rays.size and last_rays.size and rays.shape == last_rays.shape:
        return float(np.clip(np.mean(np.abs(rays - last_rays)), 0.0, 1.0))
    return 0.0


def _novelty(obs: Any, last_rays: np.ndarray) -> float:
    """Novelty: mean absolute ray change vs. previous observation."""
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    return _ray_diff(rays, last_rays)


def _uncertainty(obs: Any, last_rays: np.ndarray) -> float:
    """Uncertainty: variance of the current ray profile (0..1)."""
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    return float(np.clip(np.var(rays) if rays.size else 0.0, 0.0, 1.0))


def _threat(obs: Any, last_rays: np.ndarray) -> float:
    """Threat: 1 - normalized distance to nearest hazard (0..1)."""
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    ray_types = np.asarray(getattr(obs, "ray_types", ()), dtype=np.int64)
    hazard = _hazard_distance(rays, ray_types)
    return float(np.clip(1.0 - hazard / _RAY_RANGE, 0.0, 1.0))


def _integrity(obs: Any, last_rays: np.ndarray) -> float:
    """Integrity: 1 - health (bodily integrity deficit)."""
    _energy, health, _speed = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
    return float(np.clip(1.0 - health, 0.0, 1.0))


def _agency(obs: Any, last_rays: np.ndarray) -> float:
    """Agency: current movement speed (0..1)."""
    _energy, _health, speed = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
    return float(np.clip(speed, 0.0, 1.0))


def _learning_progress(obs: Any, last_rays: np.ndarray) -> float:
    """Learning progress: change in the ray profile between steps."""
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    return _ray_diff(rays, last_rays)


def _arousal(obs: Any, last_rays: np.ndarray) -> float:
    """Arousal: PLACEHOLDER (energy-scaled threat/uncertainty blend)."""
    energy, _health, _speed = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
    rays = np.asarray(getattr(obs, "rays", ()), dtype=np.float32)
    ray_types = np.asarray(getattr(obs, "ray_types", ()), dtype=np.int64)
    threat = float(np.clip(1.0 - _hazard_distance(rays, ray_types) / _RAY_RANGE, 0.0, 1.0))
    base = 0.5 * threat + 0.5 * float(np.clip(np.var(rays) if rays.size else 0.0, 0.0, 1.0))
    return float(np.clip(base * np.clip(energy, 0.0, 1.0), 0.0, 1.0))


def _homeostatic_deviation(obs: Any, last_rays: np.ndarray) -> float:
    """Homeostatic deviation: 1 - energy (metabolic need)."""
    energy, _health, _speed = getattr(obs, "proprioception", (1.0, 1.0, 0.0))
    return float(np.clip(1.0 - energy, 0.0, 1.0))


@dataclass(frozen=True)
class DriveDef:
    """One drive definition."""

    name: str
    label: str
    docstring: str
    compute: Callable[[Any, np.ndarray], float]


# The 8 definitions. ``label`` values are PLACEHOLDER strings for the user.
_DRIVES: tuple[DriveDef, ...] = (
    DriveDef(
        name="novelty",
        label="Novelty (PLACEHOLDER)",
        docstring="How new the current observation is relative to the recent past.",
        compute=_novelty,
    ),
    DriveDef(
        name="uncertainty",
        label="Uncertainty (PLACEHOLDER)",
        docstring="How uncertain the agent is about what it currently senses.",
        compute=_uncertainty,
    ),
    DriveDef(
        name="threat",
        label="Threat (PLACEHOLDER)",
        docstring="Proximity / danger of hazards in the sensed environment.",
        compute=_threat,
    ),
    DriveDef(
        name="integrity",
        label="Integrity (PLACEHOLDER)",
        docstring="Bodily integrity deficit (1 - health).",
        compute=_integrity,
    ),
    DriveDef(
        name="agency",
        label="Agency (PLACEHOLDER)",
        docstring="Sense of control / movement agency (speed).",
        compute=_agency,
    ),
    DriveDef(
        name="learning_progress",
        label="Learning progress (PLACEHOLDER)",
        docstring="Rate of change of the sensed world (learning signal).",
        compute=_learning_progress,
    ),
    DriveDef(
        name="arousal",
        label="Arousal (PLACEHOLDER)",
        docstring="Physiological/metabolic arousal level.",
        compute=_arousal,
    ),
    DriveDef(
        name="homeostatic_deviation",
        label="Homeostatic deviation (PLACEHOLDER)",
        docstring="Deviation from homeostatic set point (1 - energy).",
        compute=_homeostatic_deviation,
    ),
)

DRIVE_BY_NAME: dict[str, DriveDef] = {d.name: d for d in _DRIVES}


def raw_drive_vector(obs: Any, last_rays: np.ndarray) -> np.ndarray:
    """Instantaneous 8-drive vector ``d_t`` from one observation.

    Deterministic, numpy, causal (uses only information up to step ``t``).
    Returns a float32 array of shape ``(8,)``.
    """
    last = np.asarray(last_rays, dtype=np.float32)
    return np.asarray(
        [d.compute(obs, last) for d in _DRIVES],
        dtype=np.float32,
    )


def make_definitions() -> tuple[DriveDef, ...]:
    """Return the definitions tuple (for the inspector / docs)."""
    return _DRIVES
