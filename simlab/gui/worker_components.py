"""Policy and drive execution for one simulation worker."""

from __future__ import annotations

import logging
from types import MappingProxyType

import numpy as np

from simlab.core.config import SimLabConfig
from simlab.core.pipeline import compute_actions, needs_observations
from simlab.core.registry import get, load_plugins
from simlab.core.types import Action
from simlab.gui.frame import AgentFrame


def limit_torch_threads() -> None:
    """Pin torch to a single thread in the GUI process.

    The GUI worker performs inference per tick; allowing several intra-op
    threads adds scheduling jitter that stalls rendering. A single thread
    keeps the frame budget deterministic. This is a no-op when torch is not
    installed or import fails.
    """
    try:
        import torch

        torch.set_num_threads(1)
    except Exception as exc:
        logging.getLogger(__name__).debug("torch thread limiting unavailable", exc_info=exc)


class WorkerComponents:
    def __init__(self, config: SimLabConfig, seed: int, agent_count: int, rng) -> None:
        self.config = config
        self.seed = seed
        self.agent_count = agent_count
        self.reset(rng)

    def reset(self, rng) -> None:
        load_plugins()
        limit_torch_threads()
        policy_type = get("policy", self.config.policy)
        if policy_type is None:
            self.policy = None
        else:
            try:
                extra = (
                    {}
                    if not getattr(self.config, "policy_cfg", None)
                    else {"cfg": self.config.policy_cfg}
                )
                self.policy = policy_type(
                    seed=self.seed, config=self.config, **extra
                )
            except TypeError:
                try:
                    self.policy = policy_type(seed=self.seed)
                except TypeError:
                    self.policy = policy_type()
        drive_type = get("drive", self.config.drive)
        drive_kwargs = dict(getattr(self.config, "drive_config", None) or {})
        self.drive_core = drive_type(**drive_kwargs) if drive_type else None
        if self.drive_core is None:
            # Drives are world-side: always expose them for the inspector and
            # drive graphs, even for baselines configured with no drive core.
            # The substituted core is instantaneous and is not consumed by the
            # policy (baselines use obs_only / no FiLM), so actions are
            # unaffected.
            display_type = get("drive", "instant")
            self.drive_core = display_type()
        self.drive_core.reset(self.agent_count, rng)
        self.drive_names = tuple(getattr(self.drive_core, "names", ()))

    def update_config(
        self, config: SimLabConfig, seed: int, agent_count: int, rng
    ) -> None:
        self.config = config
        self.seed = seed
        self.agent_count = agent_count
        self.reset(rng)

    def needs_observations(self) -> bool:
        return needs_observations(self.policy, self.drive_core)

    def _retention_rows(self, observations) -> dict[str, dict[str, float]]:
        """Per-agent retention internals (E, lam, m, s) from the policy."""
        metrics = getattr(self.policy, "_last_metrics", None)
        if not isinstance(metrics, dict):
            return {}
        rows: dict[str, dict[str, float]] = {}
        for index, agent_id in enumerate(observations):
            row: dict[str, float] = {}
            for key in ("E", "lam", "m", "s"):
                arr = metrics.get(key)
                if arr is not None and index < len(arr):
                    row[key] = float(np.mean(arr[index]))
            if row:
                rows[agent_id] = row
        return rows

    def actions_for(self, world, step: int) -> tuple[dict[str, Action], object]:
        actions, drives, observations = compute_actions(
            world, self.policy, self.drive_core, step, batched=True
        )
        retention = self._retention_rows(observations)
        details = MappingProxyType(
            {
                agent_id: AgentFrame(
                    actions.get(agent_id),
                    observations[agent_id],
                    drives.get(agent_id, ()),
                    retention.get(agent_id),
                )
                for agent_id in observations
            }
        )
        return actions, details
