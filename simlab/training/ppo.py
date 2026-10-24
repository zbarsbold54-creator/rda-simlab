"""Recurrent PPO trainer for the learned GRU policy.

Pure PyTorch (CPU). The learner runs in one process; rollout workers are N
separate processes, each owning its own World and a CPU copy of the policy.
Workers collect fixed-length rollouts and send trajectories to the learner,
which updates and broadcasts weights. Torch is imported only here (a training
module), never by simlab.core or simlab.cli unless the train command runs.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.types import Action

logger = logging.getLogger(__name__)

# Environment constants mirroring configs/balanced.toml.
TRAIN_SEED_BASE = 100000

@dataclass
class PPOConfig:
    obs_mode: str = "obs_only"  # "obs_only" | "obs_drives"
    total_steps: int = 300000
    workers: int = 4
    seed: int = 0
    env_steps_per_rollout: int = 256
    n_agents: int = 5
    max_episode_len: int = 2000
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr0: float = 3e-4
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    seq_len: int = 32
    minibatch_chunks: int = 32
    eval_every: int = 20
    config_path: str = "configs/balanced.toml"
    # Optional architecture selection (as emitted by the GUI panel / train
    # --arch TOML). When set, it overrides obs_mode/backbone/fusion.
    arch: dict[str, Any] | None = None
    # When True, the trainer records wall-clock shares per phase and prints a
    # profile summary at the end. Off by default; timing overhead is excluded
    # from the default training path.
    profile: bool = False
    # Optional torch intra-op thread count for the learner process during the
    # PPO update only. When None (default), the torch default is left
    # untouched (behavior-identical). Rollout workers always run with 1 thread.
    update_threads: int | None = None
    # Device used for the PPO update phase. Rollouts always run on CPU and
    # workers keep CPU copies of the policy; only forward/backward/optimizer
    # run on the update device. Defaults to "cpu" (behavior-identical).
    update_device: str = "cpu"  # "cpu" | "cuda"


def _build_net(cfg: PPOConfig):
    from simlab.training.arch_spec import build_policy_cfg
    from simlab.training.torch_policy import make_policy

    arch = getattr(cfg, "arch", None) or {}
    policy_cfg = build_policy_cfg(arch) if arch else {
        "backbone": "gru",
        "fusion": "concat",
        "obs_mode": cfg.obs_mode,
        "seed": cfg.seed,
    }
    return make_policy(policy_cfg)


class _WorkerEnv:
    """Per-worker environment wrapper: its own World, drive core and policy."""

    def __init__(self, cfg: PPOConfig, rank: int) -> None:
        from dataclasses import replace

        from simlab.core.config import load_config
        from simlab.training.encoding import encode_observation

        self.cfg = cfg
        self.rank = rank
        self.seed = TRAIN_SEED_BASE + rank * 1000
        self._prof = cfg.profile
        if self._prof:
            self._t_env = 0.0
            self._t_inf = 0.0
            self._t_oth = 0.0
            self._n_steps = 0
        self._encode = encode_observation
        base = load_config(cfg.config_path)
        self.max_energy = float(base.agent.initial_energy)
        c = replace(base, agent=replace(base.agent, count=cfg.n_agents))
        arch = getattr(cfg, "arch", None) or {}
        if arch.get("drive_core"):
            c = replace(
                c,
                drive=arch["drive_core"],
                drive_config=arch.get("drive_config") or None,
            )
        self._build_world(c)
        self.policy = _build_net(cfg)
        self.policy.reset(cfg.n_agents, np.random.default_rng(self.seed))
        self._hidden = {aid: self.policy._t.zeros(self.policy.hidden_size) for aid in self.world._agents}
        self._init_prev()

    def _build_world(self, c) -> None:
        from simlab.core.runner import Runner

        self.runner = Runner(c, seed=self.seed, agents=self.cfg.n_agents)
        self.runner.setup()
        self.world = self.runner._world
        self.runner._ensure_components()
        self.drive = self.runner._drive_core

    def _init_prev(self) -> None:
        self._prev_alive = {aid: st["is_alive"] for aid, st in self.world._agents.items()}
        self._prev_energy = {aid: st["energy"] for aid, st in self.world._agents.items()}

    def _reset_episode(self) -> None:
        from dataclasses import replace

        from simlab.core.config import load_config

        self.seed += 1
        base = load_config(self.cfg.config_path)
        c = replace(base, agent=replace(base.agent, count=self.cfg.n_agents))
        arch = getattr(self.cfg, "arch", None) or {}
        if arch.get("drive_core"):
            c = replace(
                c,
                drive=arch["drive_core"],
                drive_config=arch.get("drive_config") or None,
            )
        self._build_world(c)
        self.policy.reset(self.cfg.n_agents, np.random.default_rng(self.seed))
        self._hidden = {aid: self.policy._t.zeros(self.policy.hidden_size) for aid in self.world._agents}
        self._init_prev()

    def collect(self, torch, rng) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        cur: dict[str, dict[str, list]] = {}

        def _finalize(aid: str) -> None:
            seg = cur.pop(aid, None)
            if seg and seg["x"]:
                segments.append(
                    {
                        "aid": aid,
                        "x": np.stack(seg["x"]).astype(np.float32),
                        "turn": np.asarray(seg["turn"], dtype=np.float32),
                        "thrust": np.asarray(seg["thrust"], dtype=np.float32),
                        "eat": np.asarray(seg["eat"], dtype=np.float32),
                        "logp": np.asarray(seg["logp"], dtype=np.float32),
                        "v": np.asarray(seg["v"], dtype=np.float32),
                        "r": np.asarray(seg["r"], dtype=np.float32),
                        "done": np.asarray(seg["done"], dtype=np.float32),
                    }
                )

        prof = self._prof
        for _ in range(self.cfg.env_steps_per_rollout):
            if prof:
                t0 = time.perf_counter()
            obs_batch = self.world._observation_batch()
            live = [aid for aid, st in self.world._agents.items() if st["is_alive"]]
            if live:
                live_obs = [obs_batch[aid] for aid in live]
                drive_vals = self.drive.step(live_obs, {"rng": self.world._rng})
            xs: list[np.ndarray] = []
            for i, aid in enumerate(live):
                drives_x = tuple(float(v) for v in drive_vals[i]) if self.cfg.obs_mode == "obs_drives" else None
                xs.append(self._encode(obs_batch[aid], drives_x))
            if prof:
                self._t_env += time.perf_counter() - t0
                t0 = time.perf_counter()
            if live:
                X = torch.from_numpy(np.stack(xs)).float()
                Hb = torch.stack([self._hidden[aid] for aid in live])
                turn, thrust, eat, v, f = self.policy.sample_actions_batch(X, Hb, sample=True, rng=rng)
                with torch.no_grad():
                    logp, _ = self.policy.evaluate_batch(X, Hb, turn, thrust, eat.float())
                for j, aid in enumerate(live):
                    self._hidden[aid] = f[j]
            if prof:
                self._t_inf += time.perf_counter() - t0
                t0 = time.perf_counter()
            if not live:
                continue
            actions = {
                aid: Action(
                    turn[j].detach().item(),
                    thrust[j].detach().item(),
                    bool(eat[j].detach().item()),
                )
                for j, aid in enumerate(live)
            }
            self.world.step(
                actions,
                include_observations=False,
                include_snapshot=False,
                include_agent_states=False,
            )
            truncated = self.world.get_step_count() % self.cfg.max_episode_len == 0

            for j, aid in enumerate(live):
                st = self.world._agents[aid]
                e_now = st["energy"]
                alive_now = st["is_alive"]
                e_prev = self._prev_energy.get(aid, e_now)
                r = 0.01 if alive_now else 0.0
                if (not alive_now) and self._prev_alive.get(aid, True):
                    r -= 1.0
                if e_now > e_prev:
                    r += 0.05 * (e_now - e_prev) / self.max_energy
                done = 0.0 if alive_now else 1.0
                seg = cur.setdefault(
                    aid,
                    {
                        "x": [],
                        "turn": [],
                        "thrust": [],
                        "eat": [],
                        "logp": [],
                        "v": [],
                        "r": [],
                        "done": [],
                    },
                )
                seg["x"].append(xs[j])
                seg["turn"].append(turn[j].detach().item())
                seg["thrust"].append(thrust[j].detach().item())
                seg["eat"].append(float(eat[j].detach().item()))
                seg["logp"].append(float(logp[j].detach().item()))
                seg["v"].append(float(v[j].detach().item()))
                seg["r"].append(r)
                seg["done"].append(done)
                self._prev_energy[aid] = e_now
                self._prev_alive[aid] = alive_now
                if not alive_now:
                    _finalize(aid)

            if truncated:
                for aid in list(cur):
                    _finalize(aid)
                self._reset_episode()
            if prof:
                self._t_oth += time.perf_counter() - t0
                self._n_steps += 1

        for aid in list(cur):
            _finalize(aid)
        return segments

    def _profile_dict(self) -> dict[str, float] | None:
        if not self._prof:
            return None
        return {
            "env": self._t_env,
            "inference": self._t_inf,
            "other": self._t_oth,
            "n_steps": self._n_steps,
        }


def _rollout_seed(seed: int, rank: int, iteration: int) -> int:
    """Deterministic rollout RNG seed as a pure function of (training seed,
    worker id, iteration). Same config + same seed reproduces identical
    per-worker rollout streams across separate runs and processes."""
    return (
        int(seed) * 2654435761 ^ rank * 40503 ^ iteration * 17011
    ) & 0x7FFFFFFF


def _rollout_worker(rank: int, cfg: PPOConfig, task_q: mp.Queue, result_q: mp.Queue) -> None:
    import torch

    torch.set_num_threads(1)
    env = _WorkerEnv(cfg, rank)
    while True:
        job = task_q.get()
        if job is None:
            break
        env.policy.load_net_state_dict(job["weights"])
        rng = np.random.default_rng(
            _rollout_seed(cfg.seed, rank, job.get("iteration", 0))
        )
        try:
            segs = env.collect(torch, rng)
        except Exception as exc:  # noqa: BLE001
            result_q.put({"error": f"{type(exc).__name__}: {exc}"})
            continue
        result_q.put(
            {"rank": rank, "segments": segs, "profile": env._profile_dict()}
        )


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, out_dir: str | Path) -> None:
        import torch

        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.policy = _build_net(cfg)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr0)
        self.iterations = max(1, math.ceil(cfg.total_steps / (cfg.workers * cfg.env_steps_per_rollout)))
        self._task_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._procs: list[mp.Process] = []
        self._csv = None
        self._start = time.time()
        self._broadcast_it = -1

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._task_q = mp.Queue()
        self._result_q = mp.Queue()
        for rank in range(self.cfg.workers):
            p = mp.Process(target=_rollout_worker, args=(rank, self.cfg, self._task_q, self._result_q))
            p.start()
            self._procs.append(p)

    def stop(self) -> None:
        if self._task_q is not None:
            for _ in self._procs:
                self._task_q.put(None)
            for p in self._procs:
                p.join(timeout=10)
        self._procs = []

    def _broadcast(self) -> None:
        self._broadcast_it += 1
        sd = {k: v.detach().clone() for k, v in self.policy.net_state_dict().items()}
        job = {"weights": sd, "iteration": self._broadcast_it}
        for _ in self._procs:
            self._task_q.put(job)

    def _collect(self) -> list[dict[str, Any]]:
        results = [self._result_q.get() for _ in self._procs]
        # Concatenate worker rollouts in fixed worker-id order so the segment
        # ordering is independent of mp.Queue arrival timing (as-completed
        # order would make the concatenation nondeterministic across runs).
        results.sort(key=lambda res: res.get("rank", 0))
        segs: list[dict[str, Any]] = []
        prof = self.cfg.profile
        t_env = t_inf = t_oth = 0.0
        for res in results:
            if "error" in res:
                raise RuntimeError(f"worker error: {res['error']}")
            segs.extend(res["segments"])
            if prof:
                p = res.get("profile")
                if p:
                    t_env += p["env"]
                    t_inf += p["inference"]
                    t_oth += p["other"]
        if prof:
            self._collect_prof = {
                "env": t_env,
                "inference": t_inf,
                "other": t_oth,
            }
        return segs

    def _schedule_lr(self, it: int) -> None:
        frac = 1.0 - it / max(1, self.iterations)
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr0 * frac

    # --------------------------------------------------------------- PPO
    def _ppo_update(self, segments: list[dict[str, Any]]) -> dict[str, float]:
        t = self.torch
        xs = np.concatenate([s["x"] for s in segments])
        turn = np.concatenate([s["turn"] for s in segments])
        thrust = np.concatenate([s["thrust"] for s in segments])
        eat = np.concatenate([s["eat"] for s in segments])
        old_logp = np.concatenate([s["logp"] for s in segments])

        prof = self.cfg.profile
        if prof:
            t_gae0 = time.perf_counter()
        adv = np.zeros_like(old_logp)
        ret = np.zeros_like(old_logp)
        idx = 0
        for s in segments:
            L = len(s["x"])
            v = np.asarray(s["v"], dtype=np.float64)
            r = np.asarray(s["r"], dtype=np.float64)
            done = np.asarray(s["done"], dtype=np.float64)
            gae = 0.0
            for i in range(L - 1, -1, -1):
                next_v = 0.0 if (i == L - 1 and done[i]) else (v[i] if i == L - 1 else v[i + 1])
                delta = r[i] + self.cfg.gamma * next_v - v[i]
                gae = delta + self.cfg.gamma * self.cfg.lam * gae * (1.0 - done[i])
                adv[idx + i] = gae
                ret[idx + i] = gae + v[i]
            idx += L
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        ret_t = t.from_numpy(ret).float()
        if prof:
            t_gae = time.perf_counter() - t_gae0
            t_up0 = time.perf_counter()
            t_build = 0.0
            t_fwd = 0.0
            t_bwd = 0.0
            t_opt = 0.0
            n_minibatches = 0

        chunk_starts = []
        idx = 0
        for s in segments:
            L = len(s["x"])
            for st in range(0, L, self.cfg.seq_len):
                chunk_starts.append(idx + st)
            idx += L
        chunk_starts = np.asarray(chunk_starts)
        n_chunks = len(chunk_starts)

        use_cuda = self.cfg.update_device == "cuda"
        dev = "cuda" if use_cuda else "cpu"
        opt = self.opt
        if use_cuda:
            self.policy.to("cuda")
            opt = t.optim.Adam(self.policy.parameters(), lr=self.cfg.lr0)

        eps = self.cfg.clip
        S = self.cfg.seq_len

        total_pol = 0.0
        total_val = 0.0
        total_ent = 0.0
        total_kl = 0.0
        total_n = 0

        for _ in range(self.cfg.ppo_epochs):
            perm = self.torch.randperm(n_chunks)
            for b in range(0, n_chunks, self.cfg.minibatch_chunks):
                cids = perm[b : b + self.cfg.minibatch_chunks].tolist()
                if not cids:
                    continue
                B = len(cids)
                if prof:
                    t0 = time.perf_counter()
                xs_b = np.zeros((B, S, xs.shape[1]), dtype=np.float32)
                turn_b = np.zeros((B, S), dtype=np.float32)
                thrust_b = np.zeros((B, S), dtype=np.float32)
                eat_b = np.zeros((B, S), dtype=np.float32)
                old_b = np.zeros((B, S), dtype=np.float32)
                adv_b = np.zeros((B, S), dtype=np.float32)
                ret_b = np.zeros((B, S), dtype=np.float32)
                mask_b = np.zeros((B, S), dtype=np.float32)
                for bi, cid in enumerate(cids):
                    start = chunk_starts[cid]
                    L = min(S, len(xs) - start)
                    sl = slice(start, start + L)
                    xs_b[bi, :L] = xs[sl]
                    turn_b[bi, :L] = turn[sl]
                    thrust_b[bi, :L] = thrust[sl]
                    eat_b[bi, :L] = eat[sl]
                    old_b[bi, :L] = old_logp[sl]
                    adv_b[bi, :L] = adv[sl]
                    ret_b[bi, :L] = ret[sl]
                    mask_b[bi, :L] = 1.0
                Xb = t.from_numpy(xs_b)
                turnb = t.from_numpy(turn_b)
                thrustb = t.from_numpy(thrust_b)
                eatb = t.from_numpy(eat_b)
                oldb = t.from_numpy(old_b)
                advb = t.from_numpy(adv_b)
                retb = t.from_numpy(ret_b)
                maskb = t.from_numpy(mask_b)
                if use_cuda:
                    Xb, turnb, thrustb, eatb, oldb, advb, retb, maskb = (
                        x.to(dev) for x in (Xb, turnb, thrustb, eatb, oldb, advb, retb, maskb)
                    )
                if prof:
                    t_build += time.perf_counter() - t0
                    t0 = time.perf_counter()

                if hasattr(self.policy, "reset_aux"):
                    self.policy.reset_aux()
                h = t.zeros((B, self.policy.hidden_size), device=dev)
                pol_loss = t.zeros((), dtype=t.float32, device=dev)
                val_loss = t.zeros((), dtype=t.float32, device=dev)
                ent_loss = t.zeros((), dtype=t.float32, device=dev)
                kl_loss = t.zeros((), dtype=t.float32, device=dev)
                std_t = t.exp(self.policy.log_std_turn)
                std_th = t.exp(self.policy.log_std_thrust)
                for si in range(S):
                    m = maskb[:, si]
                    h = h * m.unsqueeze(-1)
                    a, v, f = self.policy.frame_step(Xb[:, si], h, None)
                    turn_m = t.tanh(a[:, 0])
                    thrust_m = t.sigmoid(a[:, 1])
                    eat_p = t.sigmoid(a[:, 2])
                    from torch.distributions import Bernoulli, Normal

                    logp = (
                        Normal(turn_m, std_t).log_prob(turnb[:, si])
                        + Normal(thrust_m, std_th).log_prob(thrustb[:, si])
                        + Bernoulli(eat_p).log_prob(eatb[:, si])
                    )
                    ratio = t.exp(logp - oldb[:, si])
                    adv_m = advb[:, si]
                    surr1 = ratio * adv_m
                    surr2 = t.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv_m
                    pol_loss = pol_loss + (-t.min(surr1, surr2) * m).sum()
                    val_loss = val_loss + (((retb[:, si] - v) ** 2) * m).sum()
                    ent_turn = 0.5 * (math.log(2 * math.pi * math.e) + 2 * self.policy.log_std_turn)
                    ent_thrust = 0.5 * (math.log(2 * math.pi * math.e) + 2 * self.policy.log_std_thrust)
                    ent_eat = -(eat_p * t.log(eat_p + 1e-8) + (1 - eat_p) * t.log(1 - eat_p + 1e-8))
                    ent_loss = ent_loss + ((ent_turn + ent_thrust + ent_eat) * m).sum()
                    kl_loss = kl_loss + ((oldb[:, si] - logp) * m).sum()
                    h = f
                if prof:
                    t_fwd += time.perf_counter() - t0
                n = int(maskb.sum().item())
                total_n += n
                total_pol += float(pol_loss.item())
                total_val += float(val_loss.item())
                total_ent += float(ent_loss.item())
                total_kl += float(kl_loss.item())
                aux_loss = (
                    self.policy.aux_loss()
                    if hasattr(self.policy, "aux_loss")
                    else t.zeros((), dtype=t.float32, device=dev)
                )
                loss = (
                    pol_loss
                    + 0.5 * val_loss
                    - self.cfg.entropy_coef * ent_loss
                    + aux_loss
                )
                opt.zero_grad()
                if prof:
                    t_bwd0 = time.perf_counter()
                loss.backward()
                if prof:
                    t_bwd += time.perf_counter() - t_bwd0
                    t_opt0 = time.perf_counter()
                opt.step()
                if prof:
                    t_opt += time.perf_counter() - t_opt0
                    n_minibatches += 1

        if use_cuda:
            self.policy.to("cpu")

        if prof:
            t_up = time.perf_counter() - t_up0
            self._update_prof = {
                "gae": t_gae,
                "update": t_up,
                "build": t_build,
                "forward": t_fwd,
                "backward": t_bwd,
                "optimizer": t_opt,
                "n_minibatches": n_minibatches,
                "epochs": self.cfg.ppo_epochs,
                "seq_len": S,
            }

        return {
            "mean_return": float(ret_t.mean().item()),
            "entropy": total_ent / max(1, total_n),
            "approx_kl": total_kl / max(1, total_n),
            "value_loss": total_val / max(1, total_n),
        }

    # ------------------------------------------------------------ training loop
    def _open_csv(self) -> None:
        self._csv = open(self.out_dir / "learning.csv", "w", encoding="utf-8", newline="")  # noqa: SIM115
        self._csv.write("iteration,env_steps,mean_episode_len,mean_return,entropy,approx_kl,value_loss\n")
        self._csv.flush()

    def _csv_row(self, row: dict[str, Any]) -> None:
        self._csv.write(
            "{iteration},{env_steps},{mean_episode_len:.4f},{mean_return:.4f},{entropy:.4f},{approx_kl:.4f},{value_loss:.4f}\n".format(
                **row
            )
        )
        self._csv.flush()

    def _save_checkpoint(self, path: Path) -> None:
        self.policy.save_checkpoint(path)

    def _quick_eval(self, it: int) -> None:
        from simlab.eval_lib import aggregate, run_episode_metrics

        seeds = list(range(5))
        ckpt = str(self.out_dir / "latest.pt")
        self.policy.save_checkpoint(ckpt)
        try:
            results = [
                run_episode_metrics(
                    "learned_torch", s, self.cfg.n_agents, 2000, self.cfg.config_path,
                    "simple", ckpt,
                )
                for s in seeds
            ]
            agg = aggregate(results, 0.0, 2000)
            cap = agg.get("cap_pct", float("nan"))
            logger.info("quick-eval it=%d cap_pct=%.1f", it, cap)
            self._csv.write(f"# quick_eval it={it} cap_pct={cap:.1f}\n")
            self._csv.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("quick-eval failed at it=%d: %s", it, exc)

    def train(self) -> None:
        self.start()
        self._open_csv()
        config_path = self.out_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "obs_mode": self.cfg.obs_mode,
                    "total_steps": self.cfg.total_steps,
                    "workers": self.cfg.workers,
                    "seed": self.cfg.seed,
                    "env_steps_per_rollout": self.cfg.env_steps_per_rollout,
                    "n_agents": self.cfg.n_agents,
                    "max_episode_len": self.cfg.max_episode_len,
                    "gamma": self.cfg.gamma,
                    "lam": self.cfg.lam,
                    "clip": self.cfg.clip,
                    "lr0": self.cfg.lr0,
                    "entropy_coef": self.cfg.entropy_coef,
                    "ppo_epochs": self.cfg.ppo_epochs,
                    "seq_len": self.cfg.seq_len,
                    "eval_every": self.cfg.eval_every,
                    "seed_base": TRAIN_SEED_BASE,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        env_steps = 0
        prof = self.cfg.profile
        if prof:
            acc = {
                "broadcast": 0.0,
                "collect": 0.0,
                "gae": 0.0,
                "update": 0.0,
                "build": 0.0,
                "forward": 0.0,
                "backward": 0.0,
                "optimizer": 0.0,
                "schedule": 0.0,
                "env": 0.0,
                "inference": 0.0,
                "worker_other": 0.0,
                "updates": 0,
                "minibatches": 0,
            }
            t_wall = time.perf_counter()
        for it in range(self.iterations):
            if prof:
                t0 = time.perf_counter()
            self._broadcast()
            if prof:
                acc["broadcast"] += time.perf_counter() - t0
                t0 = time.perf_counter()
            trajs = self._collect()
            if prof:
                acc["collect"] += time.perf_counter() - t0
                cpf = getattr(self, "_collect_prof", None)
                if cpf:
                    acc["env"] += cpf["env"]
                    acc["inference"] += cpf["inference"]
                    acc["worker_other"] += cpf["other"]
                t0 = time.perf_counter()
            env_steps += self.cfg.workers * self.cfg.env_steps_per_rollout
            if self.cfg.update_threads is not None:
                self.torch.set_num_threads(self.cfg.update_threads)
            info = self._ppo_update(trajs)
            if prof:
                upf = getattr(self, "_update_prof", None)
                if upf:
                    acc["gae"] += upf["gae"]
                    acc["update"] += upf["update"]
                    acc["build"] += upf["build"]
                    acc["forward"] += upf["forward"]
                    acc["backward"] += upf["backward"]
                    acc["optimizer"] += upf["optimizer"]
                    acc["updates"] += 1
                    acc["minibatches"] += upf["n_minibatches"]
                t0 = time.perf_counter()
            self._schedule_lr(it)
            if prof:
                acc["schedule"] += time.perf_counter() - t0
            row = {
                "iteration": it,
                "env_steps": env_steps,
                "mean_episode_len": env_steps / max(1, self.cfg.workers),
                "mean_return": info["mean_return"],
                "entropy": info["entropy"],
                "approx_kl": info["approx_kl"],
                "value_loss": info["value_loss"],
            }
            self._csv_row(row)
            logger.info(
                "it=%d env=%d return=%.3f entropy=%.3f kl=%.4f vloss=%.3f",
                it, env_steps, row["mean_return"], row["entropy"], row["approx_kl"], row["value_loss"],
            )
            if (it + 1) % self.cfg.eval_every == 0:
                self._quick_eval(it)
        self._save_checkpoint(self.out_dir / "final.pt")
        self.stop()
        if prof:
            serial = acc["broadcast"] + acc["collect"] + acc["gae"] + acc["update"] + acc["schedule"]
            acc["other"] = max(0.0, (time.perf_counter() - t_wall) - serial)
            self._print_profile(acc, env_steps)
        logger.info("training finished at iteration=%d env_steps=%d", self.iterations - 1, env_steps)

    def _print_profile(
        self, acc: dict[str, float], env_steps: int
    ) -> None:
        total = sum(acc[k] for k in ("broadcast", "collect", "gae", "update", "schedule", "other"))
        total = total if total > 0 else 1e-9
        print("=== PPO profile (wall clock) ===")
        rows = [
            ("broadcast", acc["broadcast"]),
            ("IPC/collect wait", acc["collect"]),
            ("GAE (CPU)", acc["gae"]),
            ("PPO update (torch)", acc["update"]),
            ("lr schedule", acc["schedule"]),
            ("other", acc["other"]),
        ]
        for name, secs in rows:
            print(f"{name:<22} {secs:8.2f}s  {100.0 * secs / total:5.1f}%")
        update_total = acc["update"] if acc["update"] > 0 else 1e-9
        print("=== PPO update breakdown (within update) ===")
        urows = [
            ("tensor build/copy", acc["build"]),
            ("forward (GRU unroll)", acc["forward"]),
            ("backward", acc["backward"]),
            ("optimizer step", acc["optimizer"]),
        ]
        for name, secs in urows:
            print(f"{name:<22} {secs:8.2f}s  {100.0 * secs / update_total:5.1f}%")
        print(
            f"updates={acc['updates']} minibatches={acc['minibatches']} "
            f"epochs={self.cfg.ppo_epochs} seq_len={self.cfg.seq_len} "
            f"minibatch_chunks={self.cfg.minibatch_chunks}"
        )
        print(f"torch.get_num_threads(learner)={self.torch.get_num_threads()}")
        print(
            "env OMP_NUM_THREADS="
            f"{__import__('os').environ.get('OMP_NUM_THREADS', '<unset>')} "
            f"MKL_NUM_THREADS={__import__('os').environ.get('MKL_NUM_THREADS', '<unset>')}"
        )
        print(f"total run wall-time         {total:8.2f}s  env steps/s = {env_steps / total:.1f}")
        # Worker-side split (parallel across N workers) within the rollout window.
        wsum = acc["env"] + acc["inference"] + acc["worker_other"]
        wsum = wsum if wsum > 0 else 1e-9
        print("=== worker rollout split (sum over workers) ===")
        print(f"env stepping / obs prep      {100.0 * acc['env'] / wsum:5.1f}%")
        print(f"policy inference             {100.0 * acc['inference'] / wsum:5.1f}%")
        print(f"bookkeeping / other         {100.0 * acc['worker_other'] / wsum:5.1f}%")
