"""Gate diagnostics helpers for the Research cockpit.

These operate on plain numpy arrays (lambda/m/s/E series, event indicators,
saturation rail) and are independent of the GUI and torch. The retention math
lives in ``simlab.training.retention``; here we only summarize/plot data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gate_stats(
    lam: np.ndarray,
    lam_max: float,
    eps: float = 1e-6,
    bins: int = 40,
) -> dict[str, Any]:
    """fraction at the saturation maximum, plus a histogram."""
    lam = np.asarray(lam, dtype=float).ravel()
    if lam.size == 0:
        return {
            "fraction_at_max": float("nan"),
            "hist": np.array([], dtype=int),
            "edges": np.array([]),
        }
    frac = float(np.mean(lam >= lam_max - eps))
    edges = np.linspace(
        np.min(lam),
        np.max(lam) if np.max(lam) > np.min(lam) else np.min(lam) + 1.0,
        bins + 1,
    )
    hist, edges = np.histogram(lam, bins=edges)
    return {"fraction_at_max": round(frac, 4), "hist": hist, "edges": edges}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 2:
        return float("nan")
    x, y = x[:n], y[:n]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlate_events(lam: np.ndarray, events: np.ndarray) -> float:
    """Pearson correlation between lambda and an event indicator."""
    return pearson(lam, events)


def rail_warning(
    magnitude: np.ndarray,
    rail: float,
    tol: float = 0.02,
    min_steps: int = 100,
) -> list[dict[str, int]]:
    """Runs where |E| stays within tol of the rail for > min_steps."""
    magnitude = np.asarray(magnitude, dtype=float).ravel()
    near = np.abs(magnitude - rail) <= tol
    runs: list[dict[str, int]] = []
    start = None
    for i, n in enumerate(near):
        if n and start is None:
            start = i
        elif not n and start is not None:
            if i - start > min_steps:
                runs.append(
                    {"start": int(start), "end": int(i - 1), "length": int(i - start)}
                )
            start = None
    if start is not None:
        length = len(near) - start
        if length > min_steps:
            runs.append(
                {"start": int(start), "end": int(len(near) - 1), "length": int(length)}
            )
    return runs
