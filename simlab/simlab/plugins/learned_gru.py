"""Learned GRU Policy plugin (PyTorch, CPU-only).

A registered Policy using a small recurrent cell plus actor/critic heads:

- parameters are shared across agents;
- the recurrent hidden state is per agent and is stepped every step;
- ``needs_observations = True`` (it reads ``obs`` and ``drive_state``
  through the Policy interface, never World state);
- it supports ``reset(n_agents, rng)``, ``get_state()``/``set_state()``
  so snapshots and scrub-back replay can capture/restore its hidden state;
- it loads weights from a checkpoint file, passed to the eval CLI as
  ``--policy learned_gru --checkpoint PATH``.

Torch is imported lazily (only when a policy instance is created), so
importing ``simlab.core`` or ``simlab.cli`` never loads torch. This module
must never be imported by the core or the GUI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.training.encoding import encode_observation, obs_dim

HIDDEN = 64


def _build_net(torch, in_dim: int, hidden: int):
    nn = torch.nn

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru_ih = nn.Linear(in_dim, hidden)
            self.gru_hh = nn.Linear(hidden, hidden)
            self.policy_head = nn.Linear(hidden, 3)
            self.value_head = nn.Linear(hidden, 1)
            self.log_std_turn = nn.Parameter(torch.zeros(()))
            self.log_std_thrust = nn.Parameter(torch.zeros(()))

        def forward(self, x, h):
            # x: (B, in_dim), h: (B, hidden)
            f = torch.tanh(self.gru_ih(x) + self.gru_hh(h))
            a = self.policy_head(f)  # [turn_raw, thrust_raw, eat_raw] per row
            v = self.value_head(f).squeeze(-1)
            return a, v, f

    return _Net()


class LearnedGRU:
    needs_observations = True

    def __init__(
        self,
        seed: int = 0,
        config: Any = None,
        with_drives: bool | None = None,
    ) -> None:
        import torch  # lazy on import

        self._t = torch
        self._rng = np.random.default_rng(int(seed) if seed is not None else 0)
        ckpt = None
        if config is not None:
            ckpt = getattr(config, "policy_checkpoint", None)
        if ckpt:
            self._load_checkpoint(Path(ckpt))
        else:
            use_drives = bool(with_drives) if with_drives is not None else False
            self._in_dim = obs_dim(with_drives=use_drives)
            self._with_drives = use_drives
            self._hidden = HIDDEN
            self._net = _build_net(torch, self._in_dim, HIDDEN)
        self.n_agents = 0
        self._hidden_t = None
        self._call_index = 0

    # ------------------------------------------------------------------ init
    def _load_checkpoint(self, path: Path) -> None:
        t = self._t
        data = t.load(path, map_location="cpu", weights_only=False)
        self._in_dim = int(data["in_dim"])
        self._hidden = int(data["hidden"])
        self._with_drives = bool(data.get("with_drives", False))
        self._net = _build_net(t, self._in_dim, self._hidden)
        self._net.load_state_dict(data["state_dict"])

    def save_checkpoint(self, path: str | Path) -> None:
        data = {
            "in_dim": self._in_dim,
            "hidden": self._hidden,
            "with_drives": self._with_drives,
            "state_dict": self._net.state_dict(),
        }
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        self._t.save(data, tmp)
        if hasattr(tmp, "open"):
            with tmp.open("ab") as fh:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, path)

    # --------------------------------------------------------------- lifecycle
    def reset(self, n_agents: int, rng: Any = None) -> None:
        self.n_agents = int(n_agents)
        self._hidden_t = self._t.zeros((self.n_agents, self._hidden))
        self._call_index = 0
        if rng is not None:
            self._rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(0)

    def get_state(self) -> dict[str, Any]:
        hidden = (
            self._hidden_t.detach().cpu().numpy().copy()
            if self._hidden_t is not None
            else np.zeros((0, self._hidden), dtype=np.float32)
        )
        return {
            "hidden": hidden,
            "n_agents": self.n_agents,
            "call_index": self._call_index,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        hidden = np.asarray(state["hidden"], dtype=np.float32)
        self._hidden_t = self._t.from_numpy(hidden).float()
        self.n_agents = int(state["n_agents"])
        self._call_index = int(state["call_index"])

    # ------------------------------------------------------------- inference
    def _encode(self, obs: Any, drive_state: Any) -> Any:
        drives = drive_state if self._with_drives else None
        x = encode_observation(obs, drives)
        return self._t.from_numpy(x).float().unsqueeze(0)

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if self._hidden_t is None:
            self.reset(1, self._rng)
        idx = self._call_index % self.n_agents
        x = self._encode(obs, drive_state)
        h_prev = self._hidden_t[idx: idx + 1]
        t = self._t
        with t.no_grad():
            a, _v, f = self._net(x, h_prev)
        turn = float(t.tanh(a[0, 0]))
        thrust = float(t.sigmoid(a[0, 1]))
        eat = float(t.sigmoid(a[0, 2])) >= 0.5
        self._hidden_t[idx] = f[0]
        self._call_index += 1
        return Action(turn, thrust, eat, "learned_gru")

    # --------------------------------------------------- batched (training)
    def sample_actions_batch(self, x, h, sample: bool, rng=None):
        """x: (B, D), h: (B, hidden). Returns actions, value, new hidden."""
        t = self._t
        with t.no_grad():
            a, v, f = self._net(x, h)
        turn_m = t.tanh(a[:, 0])
        thrust_m = t.sigmoid(a[:, 1])
        eat_p = t.sigmoid(a[:, 2])
        if sample:
            rng_t = t.Generator().manual_seed(int(rng.integers(0, 2**31)) if rng is not None else 0)
            eps = t.randn_like(turn_m, generator=rng_t)
            turn = (turn_m + t.exp(self._net.log_std_turn) * eps).clamp(-1.0, 1.0)
            eps2 = t.randn_like(thrust_m, generator=rng_t)
            thrust = (thrust_m + t.exp(self._net.log_std_thrust) * eps2).clamp(0.0, 1.0)
            eat = (t.rand_like(eat_p, generator=rng_t) < eat_p)
        else:
            turn, thrust, eat = turn_m, thrust_m, eat_p >= 0.5
        return turn, thrust, eat, v, f

    def evaluate_batch(self, x, h, turn, thrust, eat):
        """Log-probability and value for PPO updates. Gradients enabled."""
        t = self._t
        a, v, _f = self._net(x, h)
        turn_m = t.tanh(a[:, 0])
        thrust_m = t.sigmoid(a[:, 1])
        eat_p = t.sigmoid(a[:, 2])
        std_t = t.exp(self._net.log_std_turn)
        std_th = t.exp(self._net.log_std_thrust)
        from torch.distributions import Bernoulli, Normal

        logp = (
            Normal(turn_m, std_t).log_prob(turn)
            + Normal(thrust_m, std_th).log_prob(thrust)
            + Bernoulli(eat_p).log_prob(eat.float())
        )
        return logp, v


register("policy", "learned_gru")(LearnedGRU)
