"""Learner-side backbone slot.

A backbone is a torch module that turns a sequence of encoded observations
(optionally fused with drive values) into per-step action logits and value
estimates. Backbones are registered in the existing plugin registry under the
``backbone`` type. This module is imported only by training / torch plugin
paths; ``simlab.core`` and ``simlab.cli`` never import it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from simlab.core.registry import register

N_LOGITS = 3
MAX_N_DRIVES = 8


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class Backbone(ABC):
    """Sequence-to-sequence recurrent actor/critic module."""

    state_dim: int
    obs_dim: int
    fusion: str
    n_drives: int

    @abstractmethod
    def initial_state(self, batch: int) -> torch.Tensor:
        """Zero recurrent state, shape ``(batch, state_dim)``."""

    @abstractmethod
    def frame_step(
        self, x: torch.Tensor, h: torch.Tensor, drive: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. ``x (B, obs_dim)``, ``h (B, state_dim)``,
        ``drive (B, n_drives)`` or None. Returns ``(a (B,3), v (B,),
        f (B, state_dim))``."""

    @abstractmethod
    def param_count(self) -> int: ...

    @abstractmethod
    def config_dict(self) -> dict[str, Any]: ...

    def forward(
        self,
        obs_seq: torch.Tensor,
        drive_seq: torch.Tensor | None,
        state: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence version of ``frame_step``.

        ``obs_seq (B,T,obs_dim)``, ``drive_seq (B,T,n_drives)`` or None,
        ``state (B,state_dim)``, ``done_mask (B,T)`` (1 = keep, 0 = reset).
        Returns ``logits (B,T,3)``, ``value (B,T)``, ``new_state (B,state_dim)``.
        """
        _b, t = obs_seq.shape[0], obs_seq.shape[1]
        h = state
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for step in range(t):
            m = done_mask[:, step]
            h = h * m.unsqueeze(-1)
            drive = None if drive_seq is None else drive_seq[:, step]
            a, v, f = self.frame_step(obs_seq[:, step], h, drive)
            logits.append(a)
            values.append(v)
            h = f
        return torch.stack(logits, dim=1), torch.stack(values, dim=1), h


class GRUBackbone(Backbone):
    """The existing GRU cell, reproduced exactly (no behavior change)."""

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.state_dim = self.hidden
        net = nn.Module()
        net.gru_ih = nn.Linear(self.obs_dim, self.hidden)
        net.gru_hh = nn.Linear(self.hidden, self.hidden)
        net.policy_head = nn.Linear(self.hidden, N_LOGITS)
        net.value_head = nn.Linear(self.hidden, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.hidden)
            net.film_shift = nn.Linear(self.n_drives, self.hidden)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        if drive is not None and self.fusion == "film":
            f = torch.tanh(
                net.gru_ih(x)
                + net.film_scale(drive) * net.gru_hh(h)
                + net.film_shift(drive)
            )
        else:
            f = torch.tanh(net.gru_ih(x) + net.gru_hh(h))
        a = net.policy_head(f)
        v = net.value_head(f).squeeze(-1)
        return a, v, f

    def param_count(self) -> int:
        return _param_count(self._net)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": "gru",
            "hidden": self.hidden,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
        }


class LegacyGRUBackbone(GRUBackbone):
    """The learner-v0 net exactly (hidden default 64, no FiLM, drive-free).

    Structurally identical to ``GRUBackbone`` with ``fusion="none"`` at
    ``hidden=64``; kept as a distinct registered preset so a *v0 checkpoint
    loads into a named, documented entry point.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        fusion: str = "none",
        n_drives: int = MAX_N_DRIVES,
        seed: int = 0,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            hidden=hidden,
            fusion="none",
            n_drives=n_drives,
            seed=seed,
        )
        self.backbone_name = "gru_legacy"

    def config_dict(self) -> dict[str, Any]:
        cfg = super().config_dict()
        cfg["backbone"] = "gru_legacy"
        cfg["fusion"] = "none"
        return cfg


class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)


class TransformerBackbone(Backbone):
    """Small causal Transformer over the last ``context`` embedded obs steps.

    The recurrent state is a ring buffer of the last ``context`` embeddings.
    """

    def __init__(
        self,
        obs_dim: int,
        context: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.context = int(context)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.state_dim = self.context * self.d_model
        net = nn.Module()
        net.in_proj = nn.Linear(self.obs_dim, self.d_model)
        net.pos = nn.Parameter(torch.zeros(1, self.context, self.d_model))
        blocks = nn.ModuleList(
            [
                _TransformerBlock(self.d_model, self.n_heads)
                for _ in range(self.n_layers)
            ]
        )
        net.blocks = blocks
        net.policy_head = nn.Linear(self.d_model, N_LOGITS)
        net.value_head = nn.Linear(self.d_model, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.d_model)
            net.film_shift = nn.Linear(self.n_drives, self.d_model)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        b = x.shape[0]
        buf = h.view(b, self.context, self.d_model)
        e = net.in_proj(x)
        if drive is not None and self.fusion == "film":
            e = e * net.film_scale(drive) + net.film_shift(drive)
        buf = torch.cat([buf[:, 1:], e.unsqueeze(1)], dim=1)
        emb = buf + net.pos
        causal = torch.triu(
            torch.full((self.context, self.context), float("-inf"), device=x.device),
            diagonal=1,
        )
        out = emb
        for block in net.blocks:
            attn_out, _ = block.attn(out, out, out, attn_mask=causal)
            out = block.ln1(out + attn_out)
            out = block.ln2(out + block.ff(out))
        last = out[:, -1]
        a = net.policy_head(last)
        v = net.value_head(last).squeeze(-1)
        f = buf.contiguous().view(b, self.state_dim)
        return a, v, f

    def forward(
        self,
        obs_seq: torch.Tensor,
        drive_seq: torch.Tensor | None,
        state: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized causal sequence update.

        Computes the causal attention over the last ``K`` embedded observations
        for every step at once, including ``done_mask`` resets. Equivalent to
        the incremental :meth:`frame_step` ring-buffer rollout.
        """
        net = self._net
        b, t = obs_seq.shape[0], obs_seq.shape[1]
        dev = obs_seq.device
        k = self.context

        e = net.in_proj(obs_seq)  # (B,T,d)
        if drive_seq is not None and self.fusion == "film":
            e = e * net.film_scale(drive_seq) + net.film_shift(drive_seq)

        idx = (
            torch.arange(t, device=dev).unsqueeze(1)
            + torch.arange(k, device=dev).unsqueeze(0)
            - (k - 1)
        )  # (T,K) global index of window position

        reset = (done_mask == 0).float()  # (B,T)
        ar = torch.arange(t, device=dev).reshape(1, -1).expand(b, t)
        r_pos = torch.where(reset > 0, ar, torch.full_like(ar, -1))
        # A reset at r clears history BEFORE r, but the embedding at r is the
        # start of the new episode, so valid history begins at r itself.
        last_r = r_pos.cummax(dim=1).values
        start_i = last_r.clamp(min=0)  # (B,T)

        valid = (idx.unsqueeze(0) >= start_i.unsqueeze(-1)) & (
            idx.unsqueeze(0) >= 0
        )  # (B,T,K)

        gi = idx.clamp(0, t - 1)
        buf = e[:, gi]  # (B,T,K,d)
        buf = buf * valid.unsqueeze(-1).float()
        raw = buf  # stored without positional bias (matches frame_step state)
        buf = buf + net.pos  # (1,K,d)

        causal = torch.triu(
            torch.full((k, k), float("-inf"), device=dev), diagonal=1
        )
        n = b * t
        buf = buf.reshape(n, k, -1)
        out = buf
        for block in net.blocks:
            attn_out, _ = block.attn(out, out, out, attn_mask=causal)
            out = block.ln1(out + attn_out)
            out = block.ln2(out + block.ff(out))
        last = out[:, -1].reshape(b, t, self.d_model)
        a = net.policy_head(last)
        v = net.value_head(last).squeeze(-1)
        new_state = raw.reshape(b, t, k, self.d_model)[:, -1].reshape(
            b, self.state_dim
        )
        return a, v, new_state

    def param_count(self) -> int:
        return _param_count(self._net)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": "transformer",
            "context": self.context,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
        }


class RDABackbone(Backbone):
    """Recurrent Drive/Affective backbone.

    Reads the instantaneous drive vector ``d_t`` from the trailing
    ``n_drives`` dims of each observation (RDA forces ``obs_drives``).
    ``E_t`` is carried in the recurrent state alongside the encoder hidden
    state and updated by the :class:`~simlab.training.retention.Retention`
    module. Fusion is ``film`` (zero-init gains conditioned on ``E_t``) or
    ``concat`` (``E_t`` appended to the feature before the heads).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int = 64,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        retention_mode: str = "fixed",
        rho: float = 0.99,
        baseline: float = 0.9,
        gate_lambda_max: float = 0.9984375,
        m_scale: float = 4.0,
        surprise_loss_weight: float = 1.0,
        trainable_gates: bool = False,
        seed: int = 0,
    ) -> None:
        from simlab.training.retention import Retention

        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.retention_mode = retention_mode
        self.backbone_name = getattr(type(self), "_bb_name", "rda")
        self._retention = Retention(
            mode=retention_mode,
            n_drives=self.n_drives,
            hidden=self.hidden,
            rho=rho,
            baseline=baseline,
            gate_lambda_max=gate_lambda_max,
            m_scale=m_scale,
            surprise_loss_weight=surprise_loss_weight,
            trainable_gates=trainable_gates,
            seed=seed,
        )
        self.state_dim = self.hidden + self._retention.state_dim
        self._last_metrics: dict[str, Any] = {}
        self._aux_loss = torch.zeros(())
        self._aux_scale = float(surprise_loss_weight)

        net = nn.Module()
        net.in_proj = nn.Linear(self.obs_dim - self.n_drives, self.hidden)
        net.gru_ih = nn.Linear(self.hidden, self.hidden)
        net.gru_hh = nn.Linear(self.hidden, self.hidden)
        head_in = self.hidden + (self.n_drives if fusion == "concat" else 0)
        net.policy_head = nn.Linear(head_in, N_LOGITS)
        net.value_head = nn.Linear(head_in, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.hidden)
            net.film_shift = nn.Linear(self.n_drives, self.hidden)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        d = x[:, -self.n_drives :]
        o_raw = x[:, : -self.n_drives]
        o_emb = net.in_proj(o_raw)
        enc_h = h[:, : self.hidden]
        rs = h[:, self.hidden :]

        E, rs_next, metrics = self._retention.step(o_emb, d, rs)
        self._last_metrics = {
            "d": d.detach().cpu(),
            "E": E.detach().cpu(),
        }
        for k in ("lam", "m", "s"):
            if k in metrics:
                self._last_metrics[k] = metrics[k].detach().cpu()

        f = torch.tanh(net.gru_ih(o_emb) + net.gru_hh(enc_h))
        if self.fusion == "film":
            g = torch.tanh(f * net.film_scale(E) + net.film_shift(E))
            self._last_metrics["film_scale"] = (
                net.film_scale.weight.detach().cpu()
            )
            feat = g
        else:
            feat = torch.cat([f, E], dim=1)
        a = net.policy_head(feat)
        v = net.value_head(feat).squeeze(-1)
        h_next = torch.cat(
            [f, rs_next], dim=1
        )
        self._aux_loss = self._aux_loss + metrics.get("aux_loss", torch.zeros(()))
        return a, v, h_next

    def aux_loss(self) -> torch.Tensor:
        """Accumulated auxiliary loss for the current forward chunk."""
        return self._aux_scale * self._aux_loss

    def reset_aux(self) -> None:
        self._aux_loss = torch.zeros(())

    def introspect(self) -> dict[str, Any]:
        """Per-step introspection for the inspector (numpy copies)."""
        out: dict[str, Any] = {}
        for k, v in self._last_metrics.items():
            if hasattr(v, "detach"):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = v
        if "E" in out:
            out["E_norm"] = float(self._last_metrics["E"].norm().item())
        return out

    def param_count(self) -> int:
        return _param_count(self._net) + _param_count(self._retention)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "retention_mode": self.retention_mode,
            "hidden": self.hidden,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
            "rho": self._retention.rho,
            "baseline": self._retention.baseline,
            "gate_lambda_max": self._retention.gate_lambda_max,
            "m_scale": self._retention.m_scale,
            "surprise_loss_weight": self._retention.surprise_loss_weight,
            "trainable_gates": self._retention.trainable_gates,
        }


class RDATransformerBackbone(Backbone):
    """Transformer encoder + retention, per-position FiLM on the tokens.

    Reads ``d_t`` from the trailing drive dims of each observation (RDA
    forces ``obs_drives``). ``E_t`` is carried in the recurrent state with the
    token ring buffer. Fusion is ``film`` (per-token gains conditioned on
    ``E_t``) or ``concat`` (``E_t`` appended before the heads).
    """

    def __init__(
        self,
        obs_dim: int,
        context: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        fusion: str = "concat",
        n_drives: int = MAX_N_DRIVES,
        retention_mode: str = "fixed",
        rho: float = 0.99,
        baseline: float = 0.9,
        gate_lambda_max: float = 0.9984375,
        m_scale: float = 4.0,
        surprise_loss_weight: float = 1.0,
        trainable_gates: bool = False,
        seed: int = 0,
    ) -> None:
        from simlab.training.retention import Retention

        self.obs_dim = int(obs_dim)
        self.context = int(context)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.fusion = fusion
        self.n_drives = int(n_drives)
        self.retention_mode = retention_mode
        self.backbone_name = getattr(type(self), "_bb_name", "rda_tf")
        self._retention = Retention(
            mode=retention_mode,
            n_drives=self.n_drives,
            hidden=self.d_model,
            rho=rho,
            baseline=baseline,
            gate_lambda_max=gate_lambda_max,
            m_scale=m_scale,
            surprise_loss_weight=surprise_loss_weight,
            trainable_gates=trainable_gates,
            seed=seed,
        )
        self._tokens = self.context * self.d_model
        self.state_dim = self._tokens + self._retention.state_dim
        self._last_metrics: dict[str, Any] = {}
        self._aux_loss = torch.zeros(())
        self._aux_scale = float(surprise_loss_weight)

        net = nn.Module()
        net.in_proj = nn.Linear(self.obs_dim - self.n_drives, self.d_model)
        net.pos = nn.Parameter(torch.zeros(1, self.context, self.d_model))
        net.blocks = nn.ModuleList(
            [_TransformerBlock(self.d_model, self.n_heads) for _ in range(self.n_layers)]
        )
        head_in = self.d_model + (self.n_drives if fusion == "concat" else 0)
        net.policy_head = nn.Linear(head_in, N_LOGITS)
        net.value_head = nn.Linear(head_in, 1)
        net.log_std_turn = nn.Parameter(torch.zeros(()))
        net.log_std_thrust = nn.Parameter(torch.zeros(()))
        if fusion == "film":
            net.film_scale = nn.Linear(self.n_drives, self.d_model)
            net.film_shift = nn.Linear(self.n_drives, self.d_model)
        self._net = net

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    def frame_step(self, x, h, drive=None):
        net = self._net
        b = x.shape[0]
        d = x[:, -self.n_drives :]
        o_raw = x[:, : -self.n_drives]
        e = net.in_proj(o_raw)  # (B,d)
        buf = h[:, : self._tokens].view(b, self.context, self.d_model)
        rs = h[:, self._tokens :]

        E, rs_next, metrics = self._retention.step(e, d, rs)
        self._last_metrics = {"d": d.detach().cpu(), "E": E.detach().cpu()}
        for k in ("lam", "m", "s"):
            if k in metrics:
                self._last_metrics[k] = metrics[k].detach().cpu()

        if self.fusion == "film":
            e = e * net.film_scale(E) + net.film_shift(E)
            self._last_metrics["film_scale"] = net.film_scale.weight.detach().cpu()
        buf = torch.cat([buf[:, 1:], e.unsqueeze(1)], dim=1)
        emb = buf + net.pos
        causal = torch.triu(
            torch.full((self.context, self.context), float("-inf"), device=x.device),
            diagonal=1,
        )
        out = emb
        for block in net.blocks:
            attn_out, _ = block.attn(out, out, out, attn_mask=causal)
            out = block.ln1(out + attn_out)
            out = block.ln2(out + block.ff(out))
        last = out[:, -1]
        if self.fusion == "concat":
            last = torch.cat([last, E], dim=1)
        a = net.policy_head(last)
        v = net.value_head(last).squeeze(-1)
        h_next = torch.cat([buf.contiguous().view(b, self._tokens), rs_next], dim=1)
        self._aux_loss = self._aux_loss + metrics.get("aux_loss", torch.zeros(()))
        return a, v, h_next

    def aux_loss(self) -> torch.Tensor:
        return self._aux_scale * self._aux_loss

    def reset_aux(self) -> None:
        self._aux_loss = torch.zeros(())

    def introspect(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._last_metrics.items():
            if hasattr(v, "detach"):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = v
        if "E" in out:
            out["E_norm"] = float(self._last_metrics["E"].norm().item())
        return out

    def param_count(self) -> int:
        return _param_count(self._net) + _param_count(self._retention)

    def config_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "retention_mode": self.retention_mode,
            "context": self.context,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "fusion": self.fusion,
            "n_drives": self.n_drives,
            "rho": self._retention.rho,
            "baseline": self._retention.baseline,
            "gate_lambda_max": self._retention.gate_lambda_max,
            "m_scale": self._retention.m_scale,
            "surprise_loss_weight": self._retention.surprise_loss_weight,
            "trainable_gates": self._retention.trainable_gates,
        }


RETENTION_MODES = (
    "fixed",
    "constant",
    "surprise",
    "surprise_rand",
    "per_drive",
    "per_drive_tied",
)


def _combo_name(kind: str, mode: str) -> str:
    return f"rda_{kind}_{mode}"


def _rda_cls(base: type, kind: str, retention_mode: str):
    def _init(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("retention_mode", retention_mode)
        base.__init__(self, *args, **kwargs)

    return type(
        f"{base.__name__}_{retention_mode}",
        (base,),
        {
            "__init__": _init,
            "_bb_name": _combo_name(kind, retention_mode),
        },
    )


BACKBONES: dict[str, type[Backbone]] = {
    "gru": GRUBackbone,
    "transformer": TransformerBackbone,
    "gru_legacy": LegacyGRUBackbone,
}

for _mode in RETENTION_MODES:
    BACKBONES[_combo_name("gru", _mode)] = _rda_cls(RDABackbone, "gru", _mode)
    BACKBONES[_combo_name("tf", _mode)] = _rda_cls(
        RDATransformerBackbone, "tf", _mode
    )

# Legacy aliases -> the rda_gru_* combos (keeps saved TOML loading).
_RDA_ALIASES = {
    "rda": _combo_name("gru", "fixed"),
    "rda_const": _combo_name("gru", "constant"),
    "rda_g1": _combo_name("gru", "surprise"),
    "rda_g1_rand": _combo_name("gru", "surprise_rand"),
    "rda_g3": _combo_name("gru", "per_drive"),
    "rda_g3_tied": _combo_name("gru", "per_drive_tied"),
}
for _alias, _target in _RDA_ALIASES.items():
    BACKBONES[_alias] = BACKBONES[_target]


def _is_recurrent(backbone: str) -> bool:
    return not (backbone == "transformer" or backbone.startswith("rda_tf"))


def match_params(
    target: int,
    backbone: str,
    obs_dim: int,
    fusion: str = "concat",
    n_drives: int = MAX_N_DRIVES,
    context: int = 16,
) -> int:
    """Pick a hidden size (hidden / d_model) whose parameter count is within 5%
    of ``target``."""
    cls = BACKBONES[backbone]
    target = int(target)
    best: tuple[int, int] | None = None
    for size in range(32, 1025, 16):
        if _is_recurrent(backbone):
            bb = cls(obs_dim=obs_dim, hidden=size, fusion=fusion, n_drives=n_drives)
        else:
            bb = cls(
                obs_dim=obs_dim,
                d_model=size,
                fusion=fusion,
                n_drives=n_drives,
                context=context,
            )
        count = bb.param_count()
        if best is None or abs(count - target) < abs(best[0] - target):
            best = (count, size)
        if abs(count - target) / target <= 0.05:
            return size
    assert best is not None
    return best[1]


for _name, _cls in BACKBONES.items():
    register("backbone", _name)(_cls)
