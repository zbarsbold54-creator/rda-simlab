"""Learner-side retention (torch). Holds the persistent drive state ``E_t``.

``E_t = lam_t * E_{t-1} + (1 - lam_t) * d_t``

The gate (``lam_t``) at step ``t`` may use only information up to ``t-1``
(past-only rule). The write at ``t`` may use ``d_t``. See ``docs/RDA_SIM.md``
for the full accounting of where each quantity is computed.

This module is imported only by training paths; ``simlab.core`` and
``simlab.cli`` never import it.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

EPS = 1e-6

# Gate tunables (placeholders; the user edits these).
DEFAULT_BASELINE = 0.9          # lam floor
DEFAULT_GATE_LAMBDA_MAX = 0.9984375  # lam ceiling, rising from baseline
DEFAULT_RHO = 0.99              # trace decay for m_t
DEFAULT_M_SCALE = 4.0           # normalizer for the trace before the clamp
DEFAULT_SOFT_GATE_K = 1.0


class Retention(nn.Module):
    """Carries the persistent retention state for one of the RDA variants.

    Modes
    -----
    fixed          scalar or per-drive constant decay (lam fixed).
    constant       E_t is a learned constant vector (control), independent of obs.
    surprise       G1 shared gate from an aux-head prediction error.
    surprise_soft  G1 gate with running-standardized m and a sigmoid mapping.
    surprise_rand  control: lam sampled i.i.d. from the empirical marginal of G1.
    per_drive      G3 per-drive gates, each with its own trace and predictor.
    per_drive_soft G3 gate with running-standardized m and a sigmoid mapping.
    per_drive_tied G3 but all gates tied to the shared G1 gate.
    """

    def __init__(
        self,
        mode: str,
        n_drives: int,
        hidden: int,
        rho: float = DEFAULT_RHO,
        baseline: float = DEFAULT_BASELINE,
        gate_lambda_max: float = DEFAULT_GATE_LAMBDA_MAX,
        m_scale: float = DEFAULT_M_SCALE,
        surprise_loss_weight: float = 1.0,
        trainable_gates: bool = False,
        n_hist_bins: int = 64,
        seed: int = 0,
        soft_gate_k: float = DEFAULT_SOFT_GATE_K,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_drives = int(n_drives)
        self.hidden = int(hidden)
        self.rho = float(rho)
        self.baseline = float(baseline)
        self.gate_lambda_max = float(gate_lambda_max)
        self.m_scale = float(m_scale)
        self.surprise_loss_weight = float(surprise_loss_weight)
        self.trainable_gates = bool(trainable_gates)
        self.n_hist_bins = int(n_hist_bins)
        self.soft_gate_k = float(soft_gate_k)

        base = nn.Module()
        # Shared submodules are created for all modes; only the relevant ones
        # are used per mode. This keeps the architecture uniform.
        base.lam_floor = (
            nn.Parameter(torch.full((self.n_drives,), float(self.baseline)))
            if self.trainable_gates
            else torch.full((self.n_drives,), float(self.baseline))
        )
        # Learned constant vector for "constant" mode (control).
        base.e_const = nn.Parameter(torch.zeros((self.n_drives,)))

        # Aux heads are created for every mode so all variants share the same
        # parameter budget (size-matching). They contribute loss only in the
        # modes that actually use them; elsewhere they are inert.
        base.pred_o = nn.Linear(self.hidden, self.hidden)
        base.pred_d = nn.Linear(self.n_drives, self.n_drives)

        self._base = base

        # Running statistics for normalizing the surprise signal (stop-grad).
        self.register_buffer("s_mean", torch.zeros(1))
        self.register_buffer("s_var", torch.ones(1))
        self.register_buffer("d_mean", torch.zeros(self.n_drives))
        self.register_buffer("d_var", torch.ones(self.n_drives))
        if mode in ("surprise_soft", "per_drive_soft"):
            m_width = 1 if mode == "surprise_soft" else self.n_drives
            self.register_buffer("m_mean", torch.zeros(m_width))
            self.register_buffer("m_var", torch.ones(m_width))

        # Empirical histogram of the G1 gate (surprise_rand control).
        self.register_buffer(
            "hist_counts", torch.zeros(self.n_hist_bins, dtype=torch.float64)
        )
        self.register_buffer(
            "hist_edges",
            torch.linspace(
                float(self.baseline), float(self.gate_lambda_max), self.n_hist_bins + 1
            ),
        )

    # ------------------------------------------------------------------ state
    @property
    def state_dim(self) -> int:
        mode = self.mode
        n = self.n_drives
        h = self.hidden
        if mode in ("fixed", "constant"):
            return n + h + n
        if mode in ("surprise", "surprise_rand", "surprise_soft", "surprise_norm"):
            return n + 1 + h + n
        if mode == "per_drive_tied":
            return n * 3 + h + 1
        # per_drive
        return n * 3 + h

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((int(batch), self.state_dim))

    # ------------------------------------------------------------- internals
    def _split(self, rs: torch.Tensor) -> dict[str, torch.Tensor]:
        mode = self.mode
        n = self.n_drives
        h = self.hidden
        if mode in ("fixed", "constant"):
            return {
                "E": rs[:, :n],
                "prev_e": rs[:, n : n + h],
                "prev_d": rs[:, n + h : n + h + n],
            }
        if mode in ("surprise", "surprise_rand", "surprise_soft", "surprise_norm"):
            return {
                "E": rs[:, :n],
                "m": rs[:, n : n + 1],
                "prev_e": rs[:, n + 1 : n + 1 + h],
                "prev_d": rs[:, n + 1 + h : n + 1 + h + n],
            }
        if mode == "per_drive_tied":
            return {
                "E": rs[:, :n],
                "m": rs[:, n : n + n],
                "prev_e": rs[:, n + n : n + n + h],
                "prev_d": rs[:, n + n + h : n + n + h + n],
                "m_g": rs[:, n + n + h + n : n + n + h + n + 1],
            }
        return {
            "E": rs[:, :n],
            "m": rs[:, n : n + n],
            "prev_e": rs[:, n + n : n + n + h],
            "prev_d": rs[:, n + n + h : n + n + h + n],
        }

    def _assemble(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(parts.values()), dim=1)

    # ------------------------------------------------------------- gate math
    def _lam_from_m(self, m_prev: torch.Tensor) -> torch.Tensor:
        """Past-only lam: function of m_{t-1}. Monotone rising toward max."""
        u = torch.clamp(m_prev / self.m_scale, 0.0, 1.0)
        return self.baseline + (self.gate_lambda_max - self.baseline) * u

    def _soft_lam_from_m(
        self, m_prev: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if training:
            sample_mean = m_prev.detach().mean(dim=0)
            sample_var = (
                m_prev.detach().var(dim=0, unbiased=False)
                + (sample_mean - self.m_mean).square()
            )
            self.m_mean = self._ema(self.m_mean, sample_mean)
            self.m_var = self._ema(self.m_var, sample_var)
        m_hat = (m_prev - self.m_mean) / torch.sqrt(self.m_var + EPS)
        u = torch.sigmoid(self.soft_gate_k * m_hat)
        lam = self.baseline + (self.gate_lambda_max - self.baseline) * u
        return lam, u, m_hat

    def _sample_hist(self, batch: int, device: Any) -> torch.Tensor:
        """Sample lam i.i.d. from the empirical marginal of the G1 gate."""
        counts = self.hist_counts.to(device)
        total = float(counts.sum().item())
        if total <= 0.0:
            return (
                self.baseline
                + (self.gate_lambda_max - self.baseline)
                * torch.rand((batch, 1), device=device)
            )
        probs = counts / counts.sum()
        idx = torch.multinomial(probs, batch, replacement=True)
        lo = self.hist_edges[idx].to(device).unsqueeze(-1)
        hi = self.hist_edges[idx + 1].to(device).unsqueeze(-1)
        return lo + torch.rand((batch, 1), device=device) * (hi - lo)

    def _update_hist(self, lam: torch.Tensor) -> None:
        edges = self.hist_edges
        idx = torch.bucketize(lam.detach().reshape(-1), edges) - 1
        idx = torch.clamp(idx, 0, self.n_hist_bins - 1)
        self.hist_counts.index_add_(
            0, idx, torch.ones(idx.numel(), dtype=torch.float64, device=lam.device)
        )

    # ------------------------------------------------------------------ step
    def step(
        self,
        o_emb: torch.Tensor,
        d: torch.Tensor,
        rs: torch.Tensor,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """One step (incremental / inference + rollout).

        ``o_emb (B, hidden)``, ``d (B, n_drives)``, ``rs (B, state_dim)``.
        Returns ``(E, rs_next, metrics)`` where metrics carries lam/m/s for the
        inspector and the current auxiliary loss value.
        """
        if training is None:
            training = self.training
        mode = self.mode
        n = self.n_drives
        B = o_emb.shape[0]
        dev = o_emb.device
        parts = self._split(rs)
        E = parts["E"]
        prev_e = parts["prev_e"]
        prev_d = parts["prev_d"]
        metrics: dict[str, torch.Tensor] = {}

        # Uniform auxiliary-head loss across ALL variants (only the gate
        # differs). pred_o predicts the observation embedding, pred_d predicts
        # the drive vector; both share the same weight in every mode.
        pred_o_err = self._base.pred_o(prev_e) - o_emb  # (B,h)
        pred_d_err = self._base.pred_d(prev_d) - d  # (B,n)
        aux_loss = self.surprise_loss_weight * (
            torch.mean(pred_o_err**2) + torch.mean(pred_d_err**2)
        )
        metrics["aux_loss"] = aux_loss

        next_prev = {"prev_e": o_emb.detach(), "prev_d": d.detach()}

        if mode == "constant":
            E_next = self._base.e_const.expand(B, n)
            lam = torch.zeros((B, n), device=dev)
            rs_next = self._assemble({"E": E_next, **next_prev})
            metrics.update({"lam": lam, "E": E_next})
            return E_next, rs_next, metrics

        if mode == "fixed":
            lam = self._base.lam_floor  # scalar / per-drive constant
            if lam.dim() == 0:
                lam = lam.reshape(1).expand(B, n)
            else:
                lam = lam.expand(B, n)
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble({"E": E_next, **next_prev})
            metrics.update({"lam": lam, "E": E_next})
            return E_next, rs_next, metrics

        if mode == "surprise_soft":
            m_prev = parts["m"]
            s_raw = (self._base.pred_o(prev_e) - o_emb).norm(dim=1, keepdim=True)
            s_raw_d = s_raw.detach()
            if training:
                self.s_mean = self._ema(self.s_mean, s_raw_d.mean())
                self.s_var = self._ema(self.s_var, s_raw_d.var(unbiased=False))
            s = (s_raw - self.s_mean) / torch.sqrt(self.s_var + EPS)
            s = s.detach()
            m = self.rho * m_prev + s
            lam, u, m_hat = self._soft_lam_from_m(m_prev, training)
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble({"E": E_next, "m": m, **next_prev})
            metrics.update(
                {"lam": lam, "m": m, "s": s, "E": E_next, "u": u, "m_hat": m_hat}
            )
            return E_next, rs_next, metrics

        if mode == "surprise_norm":
            m_prev = parts["m"]
            s_raw = (self._base.pred_o(prev_e) - o_emb).norm(dim=1, keepdim=True)
            s_raw_d = s_raw.detach()
            if training:
                self.s_mean = self._ema(self.s_mean, s_raw_d.mean())
                self.s_var = self._ema(self.s_var, s_raw_d.var(unbiased=False))
            s = (s_raw - self.s_mean) / torch.sqrt(self.s_var + EPS)
            s = s.detach()
            s_c = torch.clamp(s, -3.0, 3.0)
            m = self.rho * m_prev + (1.0 - self.rho) * s_c
            m_std = ((1.0 - self.rho) / (1.0 + self.rho)) ** 0.5
            m_hat = m_prev / m_std
            pre_act = self.soft_gate_k * m_hat
            u = torch.sigmoid(pre_act)
            lam = self.baseline + (self.gate_lambda_max - self.baseline) * u
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble({"E": E_next, "m": m, **next_prev})
            metrics.update({"lam": lam, "m": m, "s": s, "E": E_next, "u": u, "m_hat": m_hat, "pre_act": pre_act})
            return E_next, rs_next, metrics

        # --- surprise-family modes (shared G1 gate)
        if mode in ("surprise", "surprise_rand"):
            m_prev = parts["m"]
            s_raw = (
                (self._base.pred_o(prev_e) - o_emb).norm(dim=1, keepdim=True)
            )  # (B,1)
            s_raw_d = s_raw.detach()
            if training:
                self.s_mean = self._ema(self.s_mean, s_raw_d.mean())
                self.s_var = self._ema(self.s_var, s_raw_d.var(unbiased=False))
            s = (s_raw - self.s_mean) / torch.sqrt(self.s_var + EPS)
            s = s.detach()  # stop-gradient on the gate path
            m = self.rho * m_prev + s
            lam = self._lam_from_m(m_prev)  # uses m_{t-1}
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble(
                {"E": E_next, "m": m, **next_prev}
            )
            metrics.update({"lam": lam, "m": m, "s": s, "E": E_next})
            if mode == "surprise_rand":
                lam_s = self._sample_hist(B, dev)
                if training:
                    self._update_hist(lam)
                E_next = lam_s * E + (1.0 - lam_s) * d
                rs_next = self._assemble(
                    {"E": E_next, "m": m, **next_prev}
                )
                metrics.update({"lam": lam_s.expand(B, n), "m": m, "s": s, "E": E_next})
            return E_next, rs_next, metrics

        # --- per-drive G3 (untied / tied to shared G1)
        m_prev = parts["m"]
        d_pred = self._base.pred_d(prev_d)  # (B,n)
        d_err = (d_pred - d).abs()
        d_err_d = d_err.detach()
        if training:
            self.d_mean = self._ema(
                self.d_mean, d_err_d.mean(dim=0), dim=self.n_drives
            )
            self.d_var = self._ema(
                self.d_var, d_err_d.var(dim=0, unbiased=False), dim=self.n_drives
            )
        s = (d_err - self.d_mean) / torch.sqrt(self.d_var + EPS)
        s = s.detach()
        m = self.rho * m_prev + s

        if mode == "per_drive_soft":
            lam, u, m_hat = self._soft_lam_from_m(m_prev, training)
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble({"E": E_next, "m": m, **next_prev})
            metrics.update(
                {"lam": lam, "m": m, "s": s, "E": E_next, "u": u, "m_hat": m_hat}
            )
            return E_next, rs_next, metrics

        if mode == "per_drive_norm":
            s_c = torch.clamp(s, -3.0, 3.0)
            m = self.rho * m_prev + (1.0 - self.rho) * s_c
            m_std = ((1.0 - self.rho) / (1.0 + self.rho)) ** 0.5
            m_hat = m_prev / m_std
            pre_act = self.soft_gate_k * m_hat
            u = torch.sigmoid(pre_act)
            lam = self.baseline + (self.gate_lambda_max - self.baseline) * u
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble({"E": E_next, "m": m, **next_prev})
            metrics.update({"lam": lam, "m": m, "s": s, "E": E_next, "u": u, "m_hat": m_hat, "pre_act": pre_act})
            return E_next, rs_next, metrics

        if mode == "per_drive_tied":
            s_raw = (
                (self._base.pred_o(prev_e) - o_emb).norm(dim=1, keepdim=True)
            )
            s_raw_d = s_raw.detach()
            if training:
                self.s_mean = self._ema(self.s_mean, s_raw_d.mean())
                self.s_var = self._ema(self.s_var, s_raw_d.var(unbiased=False))
            s_g = (s_raw - self.s_mean) / torch.sqrt(self.s_var + EPS)
            s_g = s_g.detach()
            m_g_prev = parts["m_g"]
            m_g = self.rho * m_g_prev + s_g
            lam = self._lam_from_m(m_g_prev).expand(B, n)
            E_next = lam * E + (1.0 - lam) * d
            rs_next = self._assemble(
                {"E": E_next, "m": m, **next_prev, "m_g": m_g}
            )
            metrics.update(
                {"lam": lam, "m": m, "s": s, "m_g": m_g, "E": E_next}
            )
            return E_next, rs_next, metrics

        lam = self._lam_from_m(m_prev)
        E_next = lam * E + (1.0 - lam) * d
        rs_next = self._assemble({"E": E_next, "m": m, **next_prev})
        metrics.update({"lam": lam, "m": m, "s": s, "E": E_next})
        return E_next, rs_next, metrics

    def _ema(
        self,
        cur: torch.Tensor,
        sample: torch.Tensor,
        dim: int | None = None,
    ) -> torch.Tensor:
        """Exponential moving average of a running statistic."""
        return 0.99 * cur + 0.01 * sample.detach()
