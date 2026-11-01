"""TorchPolicy adapter.

Wraps any learner-side Backbone behind the sim ``Policy`` port so trainer and
eval depend only on this adapter. Torch is imported lazily (on construction),
matching the existing learned-policy isolation contract.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.registry import register
from simlab.core.types import Action
from simlab.training.encoding import encode_observation, obs_dim

POLICY_NAME = "learned_torch"
DEFAULT_HIDDEN = 64


def _infer_obs_mode(in_dim: int) -> str:
    return "obs_drives" if int(in_dim) > obs_dim(with_drives=False) else "obs_only"


def _policy_in_dim(backbone: str, fusion: str, with_drives: bool) -> int:
    """Width of the observation consumed by ``backbone``'s entry layer.

    RDA backbones slice the instantaneous drive vector from the trailing obs
    dims, so they always read the full ``obs_drives`` width regardless of
    fusion. Non-RDA backbones consume ``x`` as-is; film fusion bypasses the
    drive argument at train time (``drive=None``), so the net is sized to the
    actual observation width (drives included when ``obs_drives``).
    """
    if backbone.startswith("rda_") or backbone == "rda":
        return obs_dim(with_drives=True)
    return obs_dim(with_drives=with_drives)


class TorchPolicy:
    needs_observations = True

    def __init__(
        self, seed: int = 0, config: Any = None, cfg: dict[str, Any] | None = None
    ) -> None:
        import torch  # lazy on import

        self._t = torch
        self._rng = np.random.default_rng(int(seed) if seed is not None else 0)
        self.n_agents = 0
        self._hidden_t: Any = None
        self._call_index = 0
        self._backbone: Any = None
        self._in_dim = 0
        self._with_drives = False
        self._fusion = "concat"
        self._embed_drives = False
        self._load_error: str | None = None
        self._checkpoint_path: Path | None = None
        self._trained = False
        self._train_step: int | None = None
        if config is not None:
            ckpt = getattr(config, "policy_checkpoint", None)
            if ckpt:
                try:
                    self._load_checkpoint(Path(ckpt))
                    return
                except Exception as exc:  # noqa: BLE001 - keep previous/fresh policy
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    logging.getLogger(__name__).warning(
                        "checkpoint %s unreadable, using fresh policy: %s",
                        ckpt,
                        self._load_error,
                    )
        self._build_from_cfg(dict(cfg or {}), int(seed))

    # ------------------------------------------------------------- construction
    def _refresh_embed(self) -> None:
        """Drives are embedded in ``x`` when the net expects the full width.

        RDA backbones always read the drive vector from the trailing obs dims
        (``_policy_in_dim`` returns ``obs_dim(with_drives=True)``), so their
        encoder must append drives. Non-RDA film backbones receive drives via
        a separate film port and expect ``x`` at the plain observation width.
        """
        self._embed_drives = self._in_dim == obs_dim(with_drives=True)

    def _build_from_cfg(self, cfg: dict[str, Any], seed: int) -> None:
        from simlab.training.backbones import BACKBONES

        torch = self._t
        backbone_name = cfg.get("backbone", "gru")
        obs_mode = cfg.get("obs_mode", "obs_only")
        self._with_drives = obs_mode == "obs_drives"
        self._fusion = cfg.get("fusion", "concat")
        self._n_drives = 8
        self._in_dim = _policy_in_dim(backbone_name, self._fusion, self._with_drives)
        kwargs = dict(cfg.get("kwargs", {}))
        torch.manual_seed(seed)
        self._backbone = BACKBONES[backbone_name](
            obs_dim=self._in_dim,
            fusion=self._fusion,
            n_drives=self._n_drives,
            **kwargs,
        )
        self._refresh_embed()

    def _build_from_arch(self, arch: dict[str, Any]) -> None:
        from simlab.training.backbones import BACKBONES

        self._fusion = arch.get("fusion", "concat")
        self._n_drives = int(arch.get("n_drives", 8))
        backbone_name = arch.get("backbone", "gru")
        self._in_dim = _policy_in_dim(backbone_name, self._fusion, self._with_drives)
        kwargs = {
            k: v for k, v in arch.items() if k not in ("backbone", "fusion", "n_drives")
        }
        self._backbone = BACKBONES[backbone_name](
            obs_dim=self._in_dim,
            fusion=self._fusion,
            n_drives=self._n_drives,
            **kwargs,
        )
        self._refresh_embed()

    # ------------------------------------------------------------- checkpoint
    def _load_checkpoint(self, path: Path) -> None:
        t = self._t
        data = t.load(path, map_location="cpu", weights_only=False)
        arch = data.get("arch")
        if arch is not None:
            self._with_drives = bool(data.get("with_drives", False))
            self._build_from_arch(dict(arch))
        else:
            self._in_dim = int(data["in_dim"])
            self._with_drives = bool(data.get("with_drives", False))
            self._fusion = "none"
            hidden = int(data["hidden"])
            self._build_from_arch(
                {"backbone": "gru_legacy", "hidden": hidden, "fusion": "none"}
            )
        self._backbone._net.load_state_dict(data["state_dict"])
        retention = getattr(self._backbone, "_retention", None)
        if retention is not None and data.get("retention_state_dict") is not None:
            retention.load_state_dict(data["retention_state_dict"])
        self._checkpoint_path = path
        self._trained = True
        self._train_step = data.get("step") if isinstance(data.get("step"), int) else None
        if self._backbone.retention_mode in ("surprise_soft", "per_drive_soft"):
            self._backbone.eval()

    def save_checkpoint(self, path: str | Path, step: int | None = None) -> None:
        data = {
            "arch": self._backbone.config_dict(),
            "in_dim": self._in_dim,
            "with_drives": self._with_drives,
            "state_dict": self._backbone._net.state_dict(),
        }
        if step is not None:
            data["step"] = int(step)
        retention = getattr(self._backbone, "_retention", None)
        if retention is not None:
            data["retention_state_dict"] = retention.state_dict()
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        self._t.save(data, tmp)
        # Flush to disk before the atomic rename so a crash cannot leave a
        # truncated checkpoint at the final path.
        if hasattr(tmp, "open"):
            with tmp.open("ab") as fh:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp, path)

    def config_dict(self) -> dict[str, Any]:
        cfg = self._backbone.config_dict()
        cfg["obs_mode"] = "obs_drives" if self._with_drives else "obs_only"
        return cfg

    def metadata(self) -> dict[str, Any]:
        """Torch-free summary of the policy for the GUI banners/inspector."""
        cfg = self.config_dict()
        try:
            from simlab.training.arch_spec import arch_hash

            arch = arch_hash(cfg)
        except Exception:  # noqa: BLE001
            arch = ""
        return {
            "name": type(self).__name__,
            "backbone": cfg.get("backbone"),
            "fusion": cfg.get("fusion", self._fusion),
            "obs_mode": "obs_drives" if self._with_drives else "obs_only",
            "param_count": self.param_count(),
            "arch_hash": arch,
            "trained": self._trained,
            "checkpoint_path": (
                str(self._checkpoint_path) if self._checkpoint_path is not None else None
            ),
            "train_step": self._train_step,
        }

    # --------------------------------------------------------------- lifecycle
    def reset(self, n_agents: int, rng: Any = None) -> None:
        self.n_agents = int(n_agents)
        self._hidden_t = self._backbone.initial_state(self.n_agents)
        self._call_index = 0
        if rng is not None:
            self._rng = (
                rng
                if isinstance(rng, np.random.Generator)
                else np.random.default_rng(0)
            )

    def get_state(self) -> dict[str, Any]:
        hidden = (
            self._hidden_t.detach().cpu().numpy().copy()
            if self._hidden_t is not None
            else np.zeros((0, self._backbone.state_dim), dtype=np.float32)
        )
        return {
            "hidden": hidden,
            "n_agents": self.n_agents,
            "call_index": self._call_index,
            "arch": self.config_dict(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        hidden = np.asarray(state["hidden"], dtype=np.float32)
        self._hidden_t = self._t.from_numpy(hidden).float()
        self.n_agents = int(state["n_agents"])
        self._call_index = int(state["call_index"])

    # ------------------------------------------------------------- inference
    def _encode(self, obs: Any, drive_state: Any) -> Any:
        if self._embed_drives:
            x = encode_observation(obs, drive_state)
        elif self._fusion == "concat":
            drives = drive_state if self._with_drives else None
            x = encode_observation(obs, drives)
        else:
            x = encode_observation(obs, None)
        return self._t.from_numpy(x).float().unsqueeze(0)

    def _drive_tensor(self, drive_state: Any) -> Any | None:
        if self._fusion != "film" or not self._with_drives:
            return None
        drive = np.asarray(drive_state, dtype=np.float32)
        if drive.size < self._n_drives:
            drive = np.concatenate(
                [drive, np.zeros(self._n_drives - drive.size, dtype=np.float32)]
            )
        return self._t.from_numpy(drive[: self._n_drives]).float().unsqueeze(0)

    def act(self, obs: Any, drive_state: Any = None) -> Action:
        if self._hidden_t is None:
            self.reset(1, self._rng)
        idx = self._call_index % self.n_agents
        x = self._encode(obs, drive_state)
        drive = self._drive_tensor(drive_state)
        h_prev = self._hidden_t[idx : idx + 1]
        t = self._t
        with t.no_grad():
            a, _v, f = self._backbone.frame_step(x, h_prev, drive)
        turn = float(t.tanh(a[0, 0]))
        thrust = float(t.sigmoid(a[0, 1]))
        eat = float(t.sigmoid(a[0, 2])) >= 0.5
        self._hidden_t[idx] = f[0]
        self._call_index += 1
        return Action(turn, thrust, eat, POLICY_NAME)

    def act_batch(
        self,
        obs_list: list[Any],
        drive_list: list[Any],
    ) -> list[Action]:
        """One batched inference forward for ``len(obs_list)`` agents.

        Keeps the same hidden-state row convention as :meth:`act` (agent ``i``
        occupies row ``i``) so the world-view worker can step every live agent
        in a single forward pass instead of one call per agent.
        """
        if self._hidden_t is None:
            self.reset(len(obs_list), self._rng)
        t = self._t
        xs = [self._encode(obs, drv) for obs, drv in zip(obs_list, drive_list)]
        x = t.cat(xs, dim=0)
        drives = [self._drive_tensor(drv) for drv in drive_list]
        drive = None
        if drives and all(d is not None for d in drives):
            drive = t.cat(drives, dim=0)
        h_prev = self._hidden_t[: len(obs_list)]
        with t.no_grad():
            a, _v, f = self._backbone.frame_step(x, h_prev, drive)
        turns = t.tanh(a[:, 0])
        thrusts = t.sigmoid(a[:, 1])
        eats = t.sigmoid(a[:, 2]) >= 0.5
        self._hidden_t[: len(obs_list)] = f
        return [
            Action(float(turns[i]), float(thrusts[i]), bool(eats[i].item()), POLICY_NAME)
            for i in range(len(obs_list))
        ]

    # --------------------------------------------------- batched (training)
    def frame_step(self, x, h, drive: Any = None):
        """x: (B, D), h: (B, state_dim). Returns actions, value, new state."""
        return self._backbone.frame_step(x, h, drive)

    def sample_actions_batch(self, x, h, sample: bool, rng=None):
        """x: (B, D), h: (B, state_dim). Returns actions, value, new state."""
        t = self._t
        with t.no_grad():
            a, v, f = self._backbone.frame_step(x, h, None)
        turn_m = t.tanh(a[:, 0])
        thrust_m = t.sigmoid(a[:, 1])
        eat_p = t.sigmoid(a[:, 2])
        if sample:
            rng_t = t.Generator().manual_seed(
                int(rng.integers(0, 2**31)) if rng is not None else 0
            )
            eps = t.randn_like(turn_m, generator=rng_t)
            turn = (turn_m + t.exp(self.log_std_turn) * eps).clamp(-1.0, 1.0)
            eps2 = t.randn_like(thrust_m, generator=rng_t)
            thrust = (thrust_m + t.exp(self.log_std_thrust) * eps2).clamp(0.0, 1.0)
            eat = t.rand_like(eat_p, generator=rng_t) < eat_p
        else:
            turn, thrust, eat = turn_m, thrust_m, eat_p >= 0.5
        return turn, thrust, eat, v, f

    def evaluate_batch(self, x, h, turn, thrust, eat):
        """Log-probability and value for PPO updates. Gradients enabled."""
        t = self._t
        a, v, _f = self._backbone.frame_step(x, h, None)
        turn_m = t.tanh(a[:, 0])
        thrust_m = t.sigmoid(a[:, 1])
        eat_p = t.sigmoid(a[:, 2])
        std_t = t.exp(self.log_std_turn)
        std_th = t.exp(self.log_std_thrust)
        from torch.distributions import Bernoulli, Normal

        logp = (
            Normal(turn_m, std_t).log_prob(turn)
            + Normal(thrust_m, std_th).log_prob(thrust)
            + Bernoulli(eat_p).log_prob(eat.float())
        )
        return logp, v

    # -------------------------------------------------------- trainer surface
    @property
    def hidden_size(self) -> int:
        return self._backbone.state_dim

    @property
    def log_std_turn(self) -> Any:
        return self._backbone._net.log_std_turn

    @property
    def log_std_thrust(self) -> Any:
        return self._backbone._net.log_std_thrust

    def reset_aux(self) -> None:
        reset = getattr(self._backbone, "reset_aux", None)
        if reset is not None:
            reset()

    def aux_loss(self):
        fn = getattr(self._backbone, "aux_loss", None)
        if fn is not None:
            return fn()
        return self._t.zeros(())

    def parameters(self):
        yield from self._backbone._net.parameters()
        retention = getattr(self._backbone, "_retention", None)
        if retention is not None:
            yield from retention.parameters()

    def to(self, device: str):
        """Move the underlying net to ``device`` (e.g. ``"cuda"``)."""
        self._backbone._net.to(device)
        return self

    def net_state_dict(self) -> dict[str, Any]:
        return self._backbone._net.state_dict()

    def load_net_state_dict(self, state: dict[str, Any]) -> None:
        self._backbone._net.load_state_dict(state)

    def param_count(self) -> int:
        return self._backbone.param_count()


def make_policy(cfg: dict[str, Any]) -> TorchPolicy:
    """Build a TorchPolicy from an architecture config dict.

    Keys: ``backbone``, ``drive_core``, ``fusion``, ``kwargs``, ``seed``,
    ``obs_mode``, optional ``match_params`` (target parameter count) and
    optional ``checkpoint``.
    """
    cfg = dict(cfg)
    # gru_legacy is the fixed learner-v0 net (hidden 64, no FiLM, 9798
    # params). It must never be size-matched; ignore any sizing request.
    if cfg.get("backbone") == "gru_legacy":
        cfg.pop("match_params", None)
    target = cfg.get("match_params")
    if target is not None:
        from simlab.training.backbones import _is_recurrent, match_params

        backbone = cfg.get("backbone", "gru")
        obs_mode = cfg.get("obs_mode", "obs_only")
        fusion = cfg.get("fusion", "concat")
        with_drives = obs_mode == "obs_drives"
        dim = _policy_in_dim(backbone, fusion, with_drives)
        kwargs = dict(cfg.get("kwargs", {}))
        if _is_recurrent(backbone):
            kwargs.setdefault(
                "hidden", match_params(int(target), backbone, dim, fusion=fusion)
            )
        else:
            kwargs.setdefault(
                "d_model",
                match_params(
                    int(target),
                    backbone,
                    dim,
                    fusion=fusion,
                    context=kwargs.get("context", 16),
                ),
            )
        cfg["kwargs"] = kwargs
    policy = TorchPolicy(seed=cfg.get("seed", 0), cfg=cfg)
    checkpoint = cfg.get("checkpoint")
    if checkpoint is not None:
        policy._load_checkpoint(Path(checkpoint))
    return policy


register("policy", POLICY_NAME)(TorchPolicy)
