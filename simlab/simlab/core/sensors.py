"""Vectorized ray casts for all agents in the continuous world."""

from __future__ import annotations

import math

import numpy as np


def ray_sensors(
    position: tuple[float, float],
    heading: float,
    width: float,
    height: float,
    food: tuple[tuple[float, float], ...],
    hazards: tuple[tuple[float, float], ...],
    agents: tuple[tuple[float, float], ...],
    count: int = 16,
) -> tuple[list[float], list[int]]:
    """Compatibility wrapper for one-agent sensor queries."""
    distances, types = ray_sensors_batch(
        np.asarray([position], dtype=np.float64),
        np.asarray([heading], dtype=np.float64),
        width,
        height,
        np.asarray(food, dtype=np.float64).reshape(-1, 2),
        np.asarray(hazards, dtype=np.float64).reshape(-1, 2),
        np.asarray(agents, dtype=np.float64).reshape(-1, 2),
        count,
    )
    return distances[0].tolist(), types[0].tolist()


def ray_sensors_batch(
    positions: np.ndarray,
    headings: np.ndarray,
    width: float,
    height: float,
    food: np.ndarray,
    hazards: np.ndarray,
    agents: np.ndarray,
    count: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast each agent's rays against the same scene in batched NumPy arrays."""
    offsets = np.linspace(-math.radians(135), math.radians(135), count)
    offsets[count // 2] = 0.0
    angles = headings[:, None] + offsets[None, :]
    dx, dy = np.cos(angles), np.sin(angles)
    x, y = positions[:, 0, None], positions[:, 1, None]
    nearest = np.full(angles.shape, np.inf, dtype=np.float64)
    ray_types = np.full(angles.shape, 3, dtype=np.int64)

    right = dx > 0.0
    left = dx < 0.0
    down = dy > 0.0
    up = dy < 0.0
    nearest[right] = np.broadcast_to(width - x, dx.shape)[right] / dx[right]
    nearest[left] = np.broadcast_to(-x, dx.shape)[left] / dx[left]
    nearest[down] = np.minimum(
        nearest[down], np.broadcast_to(height - y, dy.shape)[down] / dy[down]
    )
    nearest[up] = np.minimum(nearest[up], np.broadcast_to(-y, dy.shape)[up] / dy[up])

    _update_nearest(nearest, ray_types, positions, dx, dy, food, 1, False)
    _update_nearest(nearest, ray_types, positions, dx, dy, hazards, 2, False)
    _update_nearest(nearest, ray_types, positions, dx, dy, agents, 4, True)
    return np.minimum(nearest, 20.0), ray_types


def _update_nearest(
    nearest: np.ndarray,
    ray_types: np.ndarray,
    origins: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    points: np.ndarray,
    type_code: int,
    exclude_origin: bool,
) -> None:
    if not points.size:
        return
    vx = points[None, None, :, 0] - origins[:, None, None, 0]
    vy = points[None, None, :, 1] - origins[:, None, None, 1]
    projection = vx * dx[:, :, None] + vy * dy[:, :, None]
    perpendicular = np.abs(vx * dy[:, :, None] - vy * dx[:, :, None])
    valid = (projection > 0.0) & (perpendicular <= 0.75) & (projection <= 20.0)
    if exclude_origin:
        valid &= projection > 1e-6
    distances = np.min(np.where(valid, projection, np.inf), axis=2)
    hits = distances < nearest
    nearest[hits] = distances[hits]
    ray_types[hits] = type_code
